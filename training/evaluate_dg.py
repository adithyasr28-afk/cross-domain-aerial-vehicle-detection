"""usage: evaluate_dg.py RUN_NAME OUT_JSON  -> evaluates runs/RUN_NAME/weights/last.pt on VisDrone-val and DOTA-val tiles and MERGES the
entries into OUT_JSON (so partial results survive a stopped session).  env: DG_ROOT, DG_DEVICE, DG_IMGSZ, DG_EVAL_MAX (limit tiles, for smoke tests)"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, freq_dg
from ultralytics import YOLO

ROOT = os.environ.get("DG_ROOT", "/home/claude/project"); DEV = os.environ.get("DG_DEVICE", "cpu"); IMGSZ = int(os.environ.get("DG_IMGSZ", 640))
MAXN = int(os.environ.get("DG_EVAL_MAX", 0))
RUN, OUT = sys.argv[1], sys.argv[2]
BUCKETS = [("<16 px", 0, 16), ("16-32 px", 16, 32), ("32-64 px", 32, 64), (">64 px", 64, 1e9)]


def load_gt(dom):
    items = []
    for ip in sorted(glob.glob(f"{ROOT}/data_shared/{dom}/images/val/*.jpg"))[: MAXN or None]:
        lp = ip.replace("/images/", "/labels/")[:-4] + ".txt"
        H, W = cv2.imread(ip).shape[:2]
        g = np.loadtxt(lp, ndmin=2) if os.path.getsize(lp) else np.zeros((0, 5))
        items.append((ip, np.stack([g[:, 0], (g[:, 1] - g[:, 3] / 2) * W, (g[:, 2] - g[:, 4] / 2) * H, (g[:, 1] + g[:, 3] / 2) * W, (g[:, 2] + g[:, 4] / 2) * H], 1)))
    return items


def iou(a, B):
    ix1 = np.maximum(a[0], B[:, 0]); iy1 = np.maximum(a[1], B[:, 1]); ix2 = np.minimum(a[2], B[:, 2]); iy2 = np.minimum(a[3], B[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def match(preds, gt):
    used = np.zeros(len(gt), bool); tp = np.zeros(len(preds), bool)
    for i in (np.argsort(-preds[:, 5]) if len(preds) else []):
        p = preds[i]; cand = np.where((gt[:, 0] == p[0]) & ~used)[0]
        if len(cand) == 0: continue
        ious = iou(p[1:5], gt[cand, 1:5]); j = ious.argmax()
        if ious[j] >= 0.5: used[cand[j]] = True; tp[i] = True
    return used, tp


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
tok = RUN.split("_"); mode, seed = tok[1], tok[3]
model = YOLO(f"{ROOT}/runs/{RUN}/weights/last.pt")
for sn, dom in {"VisDrone": "visdrone", "DOTA": "dota"}.items():
    gt = load_gt(dom)
    r = model.val(data=f"{ROOT}/data_shared/{dom}.yaml", imgsz=IMGSZ, batch=16, device=DEV, conf=0.001, iou=0.6, plots=False, workers=2, verbose=False)
    P, R = float(r.box.mp), float(r.box.mr)
    rec = {b[0]: [0, 0] for b in BUCKETS}; tp_all = fp_all = n_gt = 0
    for ip, g in gt:
        pr = model.predict(ip, imgsz=IMGSZ, conf=0.25, iou=0.6, device=DEV, verbose=False)[0].boxes
        p = np.concatenate([pr.cls.cpu().numpy()[:, None], pr.xyxy.cpu().numpy(), pr.conf.cpu().numpy()[:, None]], 1) if len(pr) else np.zeros((0, 6))
        used, tp = match(p, g); tp_all += tp.sum(); fp_all += (~tp).sum(); n_gt += len(g)
        side = np.sqrt((g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2]))
        for name, lo, hi in BUCKETS:
            m = (side >= lo) & (side < hi); rec[name][0] += int(used[m].sum()); rec[name][1] += int(m.sum())
    res[f"{RUN}|{sn}"] = dict(mode=mode, seed=seed, test=sn, P=P, R=R, F1=2 * P * R / (P + R + 1e-9), AP50=float(r.box.map50), AP50_95=float(r.box.map),
                              AP50_small=float(r.box.ap50[0]), AP50_large=float(r.box.ap50[1]),
                              size_recall={k: (v[0] / v[1] if v[1] else None) for k, v in rec.items()}, size_counts={k: v[1] for k, v in rec.items()},
                              recall_at_0p25=float(tp_all / max(n_gt, 1)), precision_at_0p25=float(tp_all / max(tp_all + fp_all, 1)), fp_per_tile=float(fp_all / max(len(gt), 1)))
    print(RUN, sn, {k: round(v, 3) for k, v in res[f"{RUN}|{sn}"].items() if isinstance(v, float)}, flush=True)
head = model.model.model[-1]
if hasattr(head, "splits"):
    res[f"{RUN}|tau"] = [float(s.tau().mean()) for s in head.splits]
json.dump(res, open(OUT, "w"), indent=1)
