"""RUN 2 on Kaggle (GPU, Internet ON).   !python /kaggle/working/kaggle_bundle2/run_kaggle2.py
env:  SMOKE=1  -> 2-minute smoke test
      STAGE=geo (default) : rotation-augmentation study, 640 px, YOLO11n
      STAGE=up            : same variants at a larger input size (DG_IMGSZ, default 960) -- run only if time allows
Every run is evaluated right after it trains and merged into /kaggle/working/dg/results_run2_<STAGE>.json (partial results survive)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.environ.get("DG_WORK", "/kaggle/working/dg"); os.makedirs(W, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"; STAGE = os.environ.get("STAGE", "geo")
IMGSZ = "640" if STAGE == "geo" else os.environ.get("DG_IMGSZ", "960")
env = dict(os.environ, DG_ROOT=W, DG_DEVICE=os.environ.get("DG_DEVICE", "0"), DG_BATCH="16" if IMGSZ == "640" else "8", DG_IMGSZ=IMGSZ, PYTHONPATH=HERE)
if SMOKE: env["DG_EVAL_MAX"] = "24"
EPOCHS = 1 if SMOKE else 30
OUT = f"{W}/results_run2_{STAGE}.json" + (".smoke" if SMOKE else "")
sh = lambda c, e=env: subprocess.run(c, shell=True, check=True, env=e)

if not os.path.exists(f"{W}/datasets/VisDrone/VisDrone2019-DET-val"):
    B = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
    for s in ["train", "val"]:
        sh(f"mkdir -p {W}/datasets/VisDrone && curl -sL --retry 5 -o {W}/z.zip {B}/VisDrone2019-DET-{s}.zip && unzip -q -o {W}/z.zip -d {W}/datasets/VisDrone && rm {W}/z.zip")
    sh(f"mkdir -p {W}/datasets/DOTAv1 && curl -sL --retry 5 -o {W}/z.zip {B}/DOTAv1.zip && unzip -q -o {W}/z.zip -d {W}/datasets/DOTAv1 && rm {W}/z.zip")
import json
NTR, NVA = (64, 32) if SMOKE else (3000, 600)
try: have = json.load(open(f"{W}/data_shared/build_summary.json"))["vd_train"][0]
except Exception: have = -1
if have != NTR:        # (re)build when missing or built at another size, e.g. left over from a smoke test
    sh(f"cd {W} && python {HERE}/build_shared.py {NTR} {NVA}")   # same tiles & seed as run 1 -> comparable
if not os.path.exists(f"{W}/yolo11n.pt") and not os.path.exists("/home/claude/project/weights/yolo11n.pt"):
    sh(f"cd {W} && curl -sL -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt")

# priority order: seed-major, so a stopped session still leaves complete seeds
if STAGE == "geo":
    plan = [("geo", s) for s in (0,)] + [("ctrlgeo", 0), ("freqgeo", 0)]
    plan += [(m, 1) for m in ("geo", "ctrlgeo", "freqgeo")] + [(m, 2) for m in ("geo", "ctrlgeo", "freqgeo", "base", "control")]
    extra_oracle = True
else:
    plan = [("base", 0), ("ctrlgeo", 0), ("base", 1), ("ctrlgeo", 1)]; extra_oracle = False
if SMOKE: plan = [("geo", 0), ("freqgeo", 0)]; extra_oracle = True

def run(mode, seed, src="visdrone"):
    e = dict(env, DG_SRC=src); t0 = time.time()
    pre = "vd" if src == "visdrone" else "dt"; name = f"{pre}_{mode}_e{EPOCHS}_s{seed}" + (f"_i{IMGSZ}" if IMGSZ != "640" else "")
    sh(f"cd {W} && python {HERE}/train_dg.py {mode} {EPOCHS} {seed} 2>&1 | tr '\\r' '\\n' | grep -E 'epochs completed|TRAIN_SECONDS|Error|Traceback' ; true", e)
    sh(f"cd {W} && python {HERE}/evaluate_dg.py {name} {OUT} 2>&1 | tr '\\r' '\\n' | grep -E '^(vd|dt)_|Error|Traceback' ; true", e)
    print(f"== {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)

for mode, seed in plan: run(mode, seed)
if extra_oracle: run("base", 0, "dota")      # DOTA-trained reference at the main scale (the ceiling for a VisDrone-trained model)
print("DONE ->", OUT)
