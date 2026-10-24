"""usage: train_dg.py MODE EPOCHS SEED [FRACTION]
MODE: base | control | split | freq | geo | ctrlgeo | freqgeo.   env: DG_ROOT, DG_DEVICE, DG_BATCH, DG_IMGSZ (default 640),
DG_SRC (visdrone | dota, default visdrone), DG_WEIGHTS (default yolo11n.pt)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
import freq_dg

ROOT = os.environ.get("DG_ROOT", "/home/claude/project")
DEV = os.environ.get("DG_DEVICE", "cpu"); BATCH = int(os.environ.get("DG_BATCH", 8)); IMGSZ = int(os.environ.get("DG_IMGSZ", 640))
SRC = os.environ.get("DG_SRC", "visdrone"); W = os.environ.get("DG_WEIGHTS", "yolo11n.pt")
mode, epochs, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
frac = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
freq_dg.FreqTrainer.mode = mode
pre = "vd" if SRC == "visdrone" else "dt"
name = f"{pre}_{mode}_e{epochs}_s{seed}" + (f"_i{IMGSZ}" if IMGSZ != 640 else "") + (f"_{W.split('.')[0]}" if W != "yolo11n.pt" else "")
m = YOLO(f"{ROOT}/{W}" if os.path.exists(f"{ROOT}/{W}") else W)
t0 = time.time()
m.train(trainer=freq_dg.FreqTrainer, data=f"{ROOT}/data_shared/{SRC}.yaml", epochs=epochs, imgsz=IMGSZ, batch=BATCH, workers=2,
        device=DEV, fraction=frac, project=f"{ROOT}/runs", name=name, exist_ok=True, plots=False, val=False, patience=1000,
        seed=seed, deterministic=False, amp=False, cache=False, verbose=False, close_mosaic=max(1, epochs // 10), warmup_epochs=1.0)
print(f"TRAIN_SECONDS {name} {time.time() - t0:.0f}")
