"""Build scale-matched, shared-class (small-vehicle / large-vehicle) YOLO datasets from VisDrone and DOTA.

Both domains are cut into <=640 px tiles at NATIVE resolution (no resizing), so a 40 px car in the source is a
40 px car in the model input. This isolates viewpoint / sensor / context shift from simple scale shift.

Usage: python3 -I src/build_shared.py N_TRAIN N_VAL
"""
import glob, os, random, shutil, sys
import numpy as np
import cv2

ROOT = os.environ.get("DG_ROOT", "/home/claude/project")
VD = f"{ROOT}/datasets/VisDrone"
DT = f"{ROOT}/datasets/DOTAv1/DOTAv1"
OUT = f"{ROOT}/data_shared"
T = 640
N_TRAIN, N_VAL = int(sys.argv[1]), int(sys.argv[2])
VD_MAP = {4: 0, 5: 0, 6: 1, 9: 1}          # car, van -> small ; truck, bus -> large
DT_MAP = {"small-vehicle": 0, "large-vehicle": 1}
random.seed(0)


def starts(L):
    if L <= T: return [0]
    s = list(range(0, L - T + 1, T))
    if L - (s[-1] + T) > T // 4: s.append(L - T)
    return s


def clip_boxes(boxes, x0, y0, w, h):
    """boxes: (cls, x1, y1, x2, y2) in image px. Keep boxes with >=50% area inside tile; return yolo rows."""
    rows = []
    for c, a, b, e, f in boxes:
        ix1, iy1, ix2, iy2 = max(a, x0), max(b, y0), min(e, x0 + w), min(f, y0 + h)
        if ix2 - ix1 < 3 or iy2 - iy1 < 3: continue
        if (ix2 - ix1) * (iy2 - iy1) < 0.5 * (e - a) * (f - b): continue
        rows.append(f"{c} {((ix1+ix2)/2-x0)/w:.6f} {((iy1+iy2)/2-y0)/h:.6f} {(ix2-ix1)/w:.6f} {(iy2-iy1)/h:.6f}")
    return rows


def vis_items(split):
    items = []
    for ap in sorted(glob.glob(f"{VD}/VisDrone2019-DET-{split}/annotations/*.txt")):
        stem = os.path.basename(ap)[:-4]
        boxes = []
        for ln in open(ap).read().strip().splitlines():
            x, y, w, h, score, cat = [int(v) for v in ln.split(",")[:6]]
            if score == 1 and cat in VD_MAP: boxes.append((VD_MAP[cat], x, y, x + w, y + h))
        if boxes: items.append((f"{VD}/VisDrone2019-DET-{split}/images/{stem}.jpg", stem, boxes))
    return items


def dota_items(split):
    items = []
    for lp in sorted(glob.glob(f"{DT}/labels/{split}_original/*.txt")):
        stem = os.path.basename(lp)[:-4]
        boxes = []
        for ln in open(lp).read().strip().splitlines():
            t = ln.split()
            if len(t) >= 10 and t[8] in DT_MAP:
                p = np.array(t[:8], float).reshape(4, 2)
                boxes.append((DT_MAP[t[8]], *p.min(0), *p.max(0)))
        if boxes: items.append((f"{DT}/images/{split}/{stem}.jpg", stem, boxes))
    return items


def make(items, n_target, tag, split, per_img):
    odir = f"{OUT}/{tag}/images/{split}"; ldir = f"{OUT}/{tag}/labels/{split}"
    os.makedirs(odir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    random.shuffle(items)
    n = 0; cls_n = [0, 0]
    for path, stem, boxes in items:
        if n >= n_target: break
        im = cv2.imread(path)
        if im is None: continue
        H, W = im.shape[:2]
        cand = []
        for y0 in starts(H):
            for x0 in starts(W):
                w, h = min(T, W - x0), min(T, H - y0)
                rows = clip_boxes(boxes, x0, y0, w, h)
                if rows: cand.append((len(rows), x0, y0, w, h, rows))
        random.shuffle(cand)
        for _, x0, y0, w, h, rows in cand[:per_img]:
            name = f"{stem}_{x0}_{y0}"
            cv2.imwrite(f"{odir}/{name}.jpg", im[y0:y0 + h, x0:x0 + w], [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(f"{ldir}/{name}.txt", "w").write("\n".join(rows) + "\n")
            for r in rows: cls_n[int(r[0])] += 1
            n += 1
            if n >= n_target: break
    print(f"{tag}/{split}: {n} tiles, instances small={cls_n[0]} large={cls_n[1]}")
    return n, cls_n


shutil.rmtree(OUT, ignore_errors=True)
res = {}
res["vd_train"] = make(vis_items("train"), N_TRAIN, "visdrone", "train", per_img=2)
res["vd_val"] = make(vis_items("val"), N_VAL, "visdrone", "val", per_img=2)
res["dt_train"] = make(dota_items("train"), N_TRAIN, "dota", "train", per_img=3)
res["dt_val"] = make(dota_items("val"), N_VAL, "dota", "val", per_img=3)
for tag in ["visdrone", "dota"]:
    open(f"{OUT}/{tag}.yaml", "w").write(f"path: {OUT}/{tag}\ntrain: images/train\nval: images/val\nnames:\n  0: small-vehicle\n  1: large-vehicle\n")
import json; json.dump(res, open(f"{OUT}/build_summary.json", "w"), indent=1)
