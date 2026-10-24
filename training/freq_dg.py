"""Frequency-split, multi-level-aligned detector on top of Ultralytics YOLO11.

Pieces
  FreqSplit     : per-level rfft2 -> learnable radial low-pass mask -> structure C (low freq) and style S = f - C
  GradReverse   : identity forward, -lambda * gradient backward (DANN)
  FreqDetect    : Detect head whose input is C (S is only used by the training losses)
  FreqDetectionModel.loss : style-perturbs half of every batch (pseudo-domain label 1, original = 0),
                   adds a * L_align + b * L_orth to the usual box/cls/dfl loss
  FreqTrainer   : DetectionTrainer that swaps the standard model/head for the two classes above

modes (run 1): "base"     = plain YOLO11n, no perturbation (the baseline of the first experiment)
       "control"  = style-perturbation augmentation only (no new modules)
       "split"    = control + frequency split (no alignment / decorrelation losses)
       "freq"     = control + split + L_align + L_orth   (the proposed method)
modes (run 2): "geo" = random 90-degree rotations only;  "ctrlgeo" = style perturbation + rotations;
       "freqgeo" = ctrlgeo + split with FIXED cut-off 0.20 + L_align (weight 1.0) + L_orth
"""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Detect


# ------------------------------------------------------------------ style perturbation (pseudo-domain generator)
def style_perturb(x, gen=None):
    """x: B x 3 x H x W in [0,1].  Changes appearance only (colour, brightness, contrast, noise, sharpness), never geometry,
    so the box labels stay valid.  Each image gets its own random draw."""
    B = x.shape[0]
    r = lambda lo, hi: torch.empty(B, 1, 1, 1, device=x.device).uniform_(lo, hi)
    gamma = torch.exp(r(math.log(0.45), math.log(2.2)))           # dark (night-like) ... washed out
    x = x.clamp(1e-4, 1) ** gamma
    gain = torch.empty(B, 3, 1, 1, device=x.device).uniform_(0.7, 1.3)  # colour cast (different sensor / season / terrain tint)
    x = x * gain
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * r(0.5, 1.5) + m                                  # contrast
    gray = (torch.rand(B, 1, 1, 1, device=x.device) < 0.3).float()  # panchromatic sensor
    x = gray * x.mean(1, keepdim=True).expand_as(x) + (1 - gray) * x
    blur = (torch.rand(B, 1, 1, 1, device=x.device) < 0.3).float()  # lower-resolution optics
    x = blur * F.avg_pool2d(x, 3, 1, 1) + (1 - blur) * x
    x = x + torch.randn_like(x) * r(0.0, 0.05)                     # sensor noise
    return x.clamp(0, 1)


def rot90_batch(img, bboxes, bidx):
    """Random 90-degree rotation of every image (k in 0..3) with the boxes turned to match.
    img: B,3,H,W (H == W);  bboxes: N,4 normalised (cx, cy, w, h);  bidx: N image index of each box.
    One counter-clockwise quarter turn maps a point (x, y) to (y, 1 - x) and swaps w and h (matches torch.rot90 on dims (2,3))."""
    B, _, H, W = img.shape
    ks = torch.randint(0, 4, (B,), device=img.device) if H == W else 2 * torch.randint(0, 2, (B,), device=img.device)
    out = img.clone()
    for k in (1, 2, 3):
        idx = (ks == k).nonzero().flatten()
        if len(idx): out[idx] = torch.rot90(img[idx], k, dims=(2, 3))
    bb = bboxes.clone(); kb = ks[bidx.long()]
    for step in (1, 2, 3):
        m = kb >= step
        if m.any():
            x, y, w, h = bb[m, 0].clone(), bb[m, 1].clone(), bb[m, 2].clone(), bb[m, 3].clone()
            bb[m, 0], bb[m, 1], bb[m, 2], bb[m, 3] = y, 1 - x, h, w
    return out, bb


# ------------------------------------------------------------------ building blocks
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class FreqSplit(nn.Module):
    """f (B,c,H,W) -> C, S (both B,c,H,W) with C + S = f.
    C keeps spectrum components with radial frequency below a learnable per-channel cut-off tau (soft sigmoid edge)."""

    def __init__(self, c, tau0=0.30, sharp=24.0, gscale=10.0, learn=True):
        super().__init__()
        self.gscale, self.sharp = gscale, sharp
        t = torch.tensor(tau0 / 0.5).clamp(1e-3, 1 - 1e-3)
        th = torch.full((c,), float(torch.logit(t)) / gscale)                      # tau = 0.5*sigmoid(gscale*theta)
        if learn:
            self.theta = nn.Parameter(th)
        else:                                                                         # a buffer, so the trainer cannot re-enable gradients
            self.register_buffer("theta", th)

    def tau(self):
        return 0.5 * torch.sigmoid(self.gscale * self.theta)       # cycles / pixel, Nyquist = 0.5

    def forward(self, f):
        B, c, H, W = f.shape
        dt = f.dtype
        spec = torch.fft.rfft2(f.float(), norm="ortho")             # B,c,H,W//2+1 complex
        fy = torch.fft.fftfreq(H, device=f.device).view(H, 1)
        fx = torch.fft.rfftfreq(W, device=f.device).view(1, -1)
        rad = torch.sqrt(fy ** 2 + fx ** 2)                         # 0 ... 0.707
        mask = torch.sigmoid(self.sharp * (self.tau().view(1, c, 1, 1) - rad.view(1, 1, H, -1)))
        C = torch.fft.irfft2(spec * mask, s=(H, W), norm="ortho").to(dt)
        return C, f - C


class DomainHead(nn.Module):
    """patch-level domain classifier (1x1 convs): one logit per spatial position"""

    def __init__(self, c, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, 1, 1))

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ detection head with the split in front
class FreqDetect(Detect):
    def setup_freq(self, ch, mode, tau0=0.30, learn=True):
        self.fmode = mode
        self.splits = nn.ModuleList(FreqSplit(c, tau0=tau0, learn=learn) for c in ch)
        self.dom = nn.ModuleList(DomainHead(c) for c in ch) if mode == "freq" else None
        self.lam = 0.0
        self.aux = None
        return self

    def forward(self, x):
        Cs, Ss = [], []
        for s, f in zip(self.splits, x):
            C, S = s(f)
            Cs.append(C); Ss.append(S)
        if self.training:
            self.aux = (Cs, Ss)
        return super().forward(Cs)                                  # head reads structure only

    # ---- training-only losses
    def aux_losses(self, dlabel):
        Cs, Ss = self.aux
        la = lo = torch.zeros((), device=Cs[0].device)
        if self.fmode != "freq":
            return la, lo
        for C, S, d in zip(Cs, Ss, self.dom):
            logit = d(_GRL.apply(C, self.lam))                     # B,1,h,w  (gradient reversed into C)
            tgt = dlabel.view(-1, 1, 1, 1).expand_as(logit)
            la = la + F.binary_cross_entropy_with_logits(logit.float(), tgt.float())
            # decorrelation: cross-correlation between every structure channel and every style channel
            B, c, H, W = C.shape
            Cn = C.float().permute(1, 0, 2, 3).reshape(c, -1); Sn = S.float().permute(1, 0, 2, 3).reshape(c, -1)
            Cn = (Cn - Cn.mean(1, keepdim=True)) / (Cn.std(1, keepdim=True) + 1e-5)
            Sn = (Sn - Sn.mean(1, keepdim=True)) / (Sn.std(1, keepdim=True) + 1e-5)
            rho = Cn @ Sn.T / Cn.shape[1]
            lo = lo + (rho ** 2).mean()
        n = len(Cs)
        return la / n, lo / n


STYLE_MODES = {"control", "split", "freq", "ctrlgeo", "freqgeo"}      # modes that style-perturb half of each batch
GEO_MODES = {"geo", "ctrlgeo", "freqgeo"}                              # modes that randomly rotate each image by k * 90 degrees


class FreqDetectionModel(DetectionModel):
    mode = "control"
    w_align, w_orth = float(os.environ.get("DG_ALIGN", 0.1)), float(os.environ.get("DG_ORTH", 10.0))
    total_steps = 1
    step = 0
    last_aux = (0.0, 0.0)

    def loss(self, batch, preds=None):
        if not (self.training and preds is None):
            return super().loss(batch, preds)
        img = batch["img"]
        B = img.shape[0]
        if self.mode in GEO_MODES:
            img, batch["bboxes"] = rot90_batch(img, batch["bboxes"], batch["batch_idx"])
        d = torch.zeros(B, device=img.device)
        if self.mode in STYLE_MODES:
            d[torch.randperm(B, device=img.device)[: B // 2]] = 1.0   # half of the batch becomes pseudo-domain 1
            img = torch.where(d.view(-1, 1, 1, 1) > 0, style_perturb(img), img)
        batch["img"] = img
        head = self.model[-1]
        if isinstance(head, FreqDetect):
            p = self.step / max(self.total_steps, 1)
            head.lam = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0        # DANN ramp 0 -> 1
        self.step += 1
        preds = self.forward(batch["img"])
        loss, items = super().loss(batch, preds)
        if isinstance(head, FreqDetect) and head.fmode == "freq":
            la, lo = head.aux_losses(d)
            self.last_aux = (float(la), float(lo))
            extra = (self.w_align * la + self.w_orth * lo) * B
            if loss.ndim == 0:
                loss = loss + extra
            else:                                                   # per-term vector (box, cls, dfl): add to one slot only
                loss = loss + extra * torch.eye(loss.numel(), device=loss.device)[0]
        return loss, items


def to_freq_model(model, mode):
    """class-swap a freshly built DetectionModel (same weights) into the frequency-split variant.  mode "base" = untouched YOLO11"""
    if mode == "base":
        return model
    model.__class__ = FreqDetectionModel
    model.mode = mode
    head = model.model[-1]
    if mode in ("split", "freq", "freqgeo"):
        ch = [m[0].conv.in_channels if hasattr(m[0], "conv") else m[0].in_channels for m in head.cv2]
        head.__class__ = FreqDetect
        if mode == "freqgeo":      # run-2 proposed model: cut-off FIXED low (0.20), strong alignment (a = 1.0)
            head.setup_freq(ch, "freq", tau0=0.20, learn=False); model.w_align = 1.0
        else:
            head.setup_freq(ch, mode)
    return model


class FreqTrainer(DetectionTrainer):
    mode = "control"

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        return to_freq_model(model, FreqTrainer.mode)

    def _setup_train(self, *a, **k):
        super()._setup_train(*a, **k)
        self.model.total_steps = self.epochs * len(self.train_loader)
