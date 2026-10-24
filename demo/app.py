"""Offline Gradio demo for the cross-domain aerial vehicle detector.

The app deliberately loads only local checkpoint files.  It never passes a
model name such as ``yolo11n.pt`` to Ultralytics, because that could trigger a
network download.  The three research checkpoints are optional at runtime:
when they are absent, every affected action returns a readable status instead
of a traceback.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from html import escape as html_escape
from pathlib import Path
from typing import Any, Iterable

# These must be set before importing Gradio, Hugging Face helpers, or
# Ultralytics.  The demo is intentionally local-only at runtime.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_AUTOINSTALL"] = "False"
# Keep dependency caches/settings in the portable demo, not the user's profile.
_LOCAL_DIR = Path(__file__).resolve().parent
for _key, _subdir in {"YOLO_CONFIG_DIR": ".cache/ultralytics", "MPLCONFIGDIR": ".cache/matplotlib", "GRADIO_TEMP_DIR": ".cache/gradio", "HF_HOME": ".cache/huggingface"}.items():
    os.environ[_key] = str(_LOCAL_DIR / _subdir)
    (_LOCAL_DIR / _subdir).mkdir(parents=True, exist_ok=True)
_LOCAL_TEMP = _LOCAL_DIR / ".cache" / "tmp"
_LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
for _key in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_key] = str(_LOCAL_TEMP)

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths and fixed research facts
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent
CSS_PATH = DEMO_DIR / "dashboard.css"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))
WEIGHTS_DIR = DEMO_DIR / "weights"
SAMPLES_DIR = DEMO_DIR / "samples"
CLASS_NAMES = {0: "small-vehicle", 1: "large-vehicle"}
CLASS_COLORS = {
    0: (47, 190, 85),   # green, RGB
    1: (245, 160, 50),  # orange, RGB
}
WINDOW = 640
OVERLAP = 64
MERGE_IOU = 0.60
DEFAULT_CONF = 0.25
MODEL_FILES = {
    "base": WEIGHTS_DIR / "base.pt",
    "ctrlgeo": WEIGHTS_DIR / "ctrlgeo.pt",
    "freqgeo": WEIGHTS_DIR / "freqgeo.pt",
}

# Never mix PDF multi-seed averages with the bundled evaluation.
BUILTIN_RESULTS: list[dict[str, Any]] = []
RESULT_HEADERS = [
    "variant", "n", "VisDrone AP50", "DOTA AP50", "DOTA small AP50",
    "DOTA large AP50", "DOTA P", "DOTA R", "DOTA F1",
]
FREQ_LEVELS = {
    "P3 (stride 8)": 8,
    "P4 (stride 16)": 16,
    "P5 (stride 32)": 32,
}


class ModelUnavailable(RuntimeError):
    """Raised for a model that cannot be used in the current checkout."""


def _safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - only used for status text
        return f"unavailable ({exc})"


def runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": _safe_version("torch"),
        "ultralytics": _safe_version("ultralytics"),
        "gradio": _safe_version("gradio"),
        "opencv": _safe_version("cv2"),
    }


def available_sample_names() -> list[str]:
    """Return local sample stems only; never create or download a sample."""
    if not SAMPLES_DIR.exists():
        return []
    paths = sorted(p for p in SAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    return sorted({p.stem for p in paths})


def sample_path(stem: str | None) -> Path | None:
    if not stem:
        return None
    if not SAMPLES_DIR.is_dir():
        return None
    for candidate in sorted(SAMPLES_DIR.iterdir()):
        if candidate.is_file() and candidate.stem == stem and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return candidate
    return None


def _read_image_path(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalise_image(image: Any) -> np.ndarray | None:
    """Convert Gradio/PIL/path input to an RGB uint8 numpy image."""
    if image is None:
        return None
    if isinstance(image, (str, os.PathLike)):
        return _read_image_path(Path(image))
    try:
        if hasattr(image, "convert"):
            image = np.asarray(image.convert("RGB"))
        else:
            image = np.asarray(image)
    except Exception:
        return None
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or min(image.shape[:2]) < 1 or image.shape[2] not in (1, 3, 4):
        return None
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.nan_to_num(image)
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def resolve_input(uploaded: Any, sample: str | None) -> tuple[np.ndarray | None, str]:
    if uploaded is not None:
        image = normalise_image(uploaded)
        if image is not None:
            return image, "uploaded image"
        return None, "cannot read uploaded image; choose a valid PNG or JPEG"
    path = sample_path(sample)
    if path is not None:
        image = _read_image_path(path)
        if image is not None:
            return image, f"bundled sample `{path.name}`"
    if sample:
        return None, f"sample `{sample}` is not available"
    return None, "no image selected"


def tile_starts(length: int, window: int = WINDOW, overlap: int = OVERLAP) -> list[int]:
    if length <= window:
        return [0]
    stride = max(1, window - overlap)
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def tile_windows(height: int, width: int) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, min(WINDOW, width - x), min(WINDOW, height - y))
        for y in tile_starts(height)
        for x in tile_starts(width)
    ]


def padded_tile(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop at native scale and zero-pad edge tiles without resizing."""
    tile = image[y : y + height, x : x + width]
    return cv2.copyMakeBorder(tile, 0, WINDOW - height, 0, WINDOW - width, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def box_iou(box: np.ndarray, other: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], other[:, 0])
    y1 = np.maximum(box[1], other[:, 1])
    x2 = np.minimum(box[2], other[:, 2])
    y2 = np.minimum(box[3], other[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.clip(other[:, 2] - other[:, 0], 0, None) * np.clip(other[:, 3] - other[:, 1], 0, None)
    return intersection / (area_a + area_b - intersection + 1e-9)


def merge_detections(detections: list[dict[str, Any]], iou_threshold: float = MERGE_IOU) -> list[dict[str, Any]]:
    """Class-aware NMS over full-image coordinates (torchvision, vectorised).

    Same rule as before (a box is dropped when its IoU with a higher-scoring box
    of the same class is above the threshold) but fast on thousands of boxes.
    """
    if not detections:
        return []
    import torch
    import torchvision

    boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
    classes = torch.tensor([d["class_id"] for d in detections], dtype=torch.int64)
    keep = torchvision.ops.batched_nms(boxes, scores, classes, float(iou_threshold))
    return [detections[i] for i in keep.tolist()]


def draw_boxes(image: np.ndarray, detections: Iterable[dict[str, Any]]) -> np.ndarray:
    rendered = image.copy()
    for detection in detections:
        cls_id = int(detection["class_id"])
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls_id, f'class-{cls_id}')} {float(detection['score']):.2f}"
        height, width = rendered.shape[:2]
        x1, x2 = [max(0, min(width - 1, x)) for x in (x1, x2)]
        y1, y2 = [max(0, min(height - 1, y)) for y in (y1, y2)]
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
        scale *= min(1.0, max(1, width - 6) / max(1, text_width), max(1, height - 6) / max(1, text_height + baseline))
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
        label_width = min(width, text_width + 6)
        label_height = min(height, text_height + baseline + 6)
        left = max(0, min(x1, width - label_width))
        top = max(0, min(y1 - label_height, height - label_height))
        cv2.rectangle(rendered, (left, top), (left + label_width - 1, top + label_height - 1), (20, 23, 28), -1)
        cv2.putText(rendered, label, (left + 3, top + text_height + 3), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


def make_tile_grid(image: np.ndarray, windows: list[tuple[int, int, int, int]]) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, 960.0 / max(width, 1), 640.0 / max(height, 1))
    preview = (
        cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
        if scale != 1.0
        else image.copy()
    )
    for index, (x, y, tile_width, tile_height) in enumerate(windows, start=1):
        p1 = (int(round(x * scale)), int(round(y * scale)))
        p2 = (int(round((x + tile_width) * scale)), int(round((y + tile_height) * scale)))
        cv2.rectangle(preview, p1, p2, (70, 220, 255), 2)
        cv2.putText(preview, str(index), (p1[0] + 4, max(18, p1[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 220, 255), 2, cv2.LINE_AA)
    return preview


# ---------------------------------------------------------------------------
# Local model loading and inference
DEVICE = "cpu"
MODEL_REGISTRY: dict[str, dict[str, Any]] = {}
FREQ_IMPORT_ERROR: str | None = None
YOLO_IMPORT_ERROR: str | None = None


def _initialise_models() -> None:
    """Import the custom module first, then load only local checkpoints."""
    global DEVICE, FREQ_IMPORT_ERROR, YOLO_IMPORT_ERROR
    try:
        import torch

        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception as exc:
        DEVICE = "cpu"
        YOLO_IMPORT_ERROR = f"PyTorch is unavailable: {exc}"

    # A freqgeo checkpoint can pickle classes from freq_dg.  This import must
    # happen before any YOLO checkpoint is opened.
    try:
        import freq_dg  # noqa: F401
    except Exception as exc:
        FREQ_IMPORT_ERROR = str(exc)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        YOLO_IMPORT_ERROR = str(exc)
        YOLO = None

    for name, path in MODEL_FILES.items():
        record: dict[str, Any] = {
            "name": name,
            "path": path,
            "model": None,
            "status": "missing",
            "error": None,
            "warmup_ms": None,
        }
        if not path.exists():
            record["status"] = "weights not found"
            record["error"] = f"Expected local file: {path.name}"
            MODEL_REGISTRY[name] = record
            continue
        if YOLO is None:
            record["status"] = "Ultralytics unavailable"
            record["error"] = YOLO_IMPORT_ERROR or "unknown import error"
            MODEL_REGISTRY[name] = record
            continue
        if name == "freqgeo" and FREQ_IMPORT_ERROR:
            record["status"] = "custom module unavailable"
            record["error"] = FREQ_IMPORT_ERROR
            MODEL_REGISTRY[name] = record
            continue
        try:
            model = YOLO(str(path))
            start = time.perf_counter()
            # Warm up using a local zero image.  This is intentionally not a
            # model name, so Ultralytics cannot download anything.
            model.predict(np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8), imgsz=WINDOW, conf=DEFAULT_CONF, iou=MERGE_IOU, device=DEVICE, verbose=False)
            record["warmup_ms"] = (time.perf_counter() - start) * 1000.0
            record["model"] = model
            record["status"] = "ready"
        except Exception as exc:
            record["status"] = "load failed"
            record["error"] = str(exc)
        MODEL_REGISTRY[name] = record


def readiness_status() -> str:
    ready = [name for name in MODEL_FILES if MODEL_REGISTRY.get(name, {}).get("status") == "ready"]
    if len(ready) == len(MODEL_FILES):
        return "Models ready — base, ctrlgeo, freqgeo (loaded and warmed up)"
    failed = [f"{name}: {MODEL_REGISTRY.get(name, {}).get('status', 'not initialised')}" for name in MODEL_FILES if name not in ready]
    return f"Models partially ready ({len(ready)}/{len(MODEL_FILES)}). Unavailable: " + "; ".join(failed)


def model_status_lines() -> str:
    lines = [f"**Runtime:** Python {runtime_versions()['python']} · device `{DEVICE}` · offline/local assets only"]
    if FREQ_IMPORT_ERROR:
        lines.append(f"- `freq_dg.py` import warning: `{FREQ_IMPORT_ERROR}`")
    if YOLO_IMPORT_ERROR:
        lines.append(f"- Ultralytics import warning: `{YOLO_IMPORT_ERROR}`")
    for name in MODEL_FILES:
        record = MODEL_REGISTRY.get(name, {})
        state = record.get("status", "not initialised")
        if state == "ready":
            warm = record.get("warmup_ms")
            suffix = f"; warm-up {warm:.1f} ms" if isinstance(warm, (float, int)) else ""
            lines.append(f"- `{name}`: **ready**{suffix}")
        else:
            error = record.get("error")
            suffix = f" — {error}" if error else ""
            lines.append(f"- `{name}`: **{state}**{suffix}")
    if not any(record.get("status") == "ready" for record in MODEL_REGISTRY.values()):
        lines.append("- No checkpoint is bundled in this checkout. Uploading an image still exercises the non-model viewers; detection results remain unavailable until `demo/weights/*.pt` is added.")
    return "\n".join(lines)


def get_loaded_model(name: str) -> Any:
    record = MODEL_REGISTRY.get(name)
    if not record or record.get("model") is None:
        reason = (record or {}).get("error") or "weights are not available"
        raise ModelUnavailable(f"{name}: {reason}")
    return record["model"]


def _result_detections(result: Any, offset_x: int, offset_y: int, valid_width: int, valid_height: int) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    output: list[dict[str, Any]] = []
    for box, score, class_id in zip(xyxy, scores, classes):
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = min(max(x1, 0.0), float(valid_width)) + offset_x
        y1 = min(max(y1, 0.0), float(valid_height)) + offset_y
        x2 = min(max(x2, 0.0), float(valid_width)) + offset_x
        y2 = min(max(y2, 0.0), float(valid_height)) + offset_y
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        output.append({"box": (x1, y1, x2, y2), "score": float(score), "class_id": int(class_id)})
    return output


def run_inference(image: np.ndarray, model_name: str, confidence: float) -> tuple[np.ndarray, list[dict[str, Any]], list[tuple[int, int, int, int]], float]:
    """Run native-scale tiled inference and merge detections."""
    model = get_loaded_model(model_name)
    height, width = image.shape[:2]
    windows = tile_windows(height, width)
    detections: list[dict[str, Any]] = []
    start = time.perf_counter()
    for x, y, tile_width, tile_height in windows:
        tile = padded_tile(image, x, y, tile_width, tile_height)
        # Ultralytics expects numpy input in BGR order (it flips to RGB internally); the app holds RGB.
        tile = np.ascontiguousarray(tile[..., ::-1])
        results = model.predict(source=tile, imgsz=WINDOW, conf=float(confidence), iou=MERGE_IOU, device=DEVICE, max_det=3000, verbose=False)
        if results:
            detections.extend(_result_detections(results[0], x, y, tile_width, tile_height))
    merged = merge_detections(detections, MERGE_IOU)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return draw_boxes(image, merged), merged, windows, elapsed_ms


def detection_counts(detections: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for detection in detections:
        name = CLASS_NAMES.get(int(detection["class_id"]), f"class-{detection['class_id']}")
        counts[name] = counts.get(name, 0) + 1
    return counts


def format_detection_summary(model_name: str, image_label: str, image: np.ndarray, detections: list[dict[str, Any]], windows: list[tuple[int, int, int, int]], elapsed_ms: float, confidence: float = DEFAULT_CONF) -> str:
    counts = detection_counts(detections)
    return (
        f"### `{model_name}`\n"
        f"- Input: {image_label} ({image.shape[1]}×{image.shape[0]} px)\n"
        f"- Native-scale tiles: **{len(windows)}** (640 px window, {OVERLAP} px overlap; edge tiles are padded, not resized)\n"
        f"- Detections at confidence {confidence:.2f}: **{len(detections)}**\n"
        f"  - small-vehicle: **{counts.get('small-vehicle', 0)}**\n"
        f"  - large-vehicle: **{counts.get('large-vehicle', 0)}**\n"
        f"- Inference + tile merge: **{elapsed_ms:.1f} ms** on `{DEVICE}`\n"
        f"- NMS merge IoU: {MERGE_IOU:.2f}"
    )


def detect_action(uploaded: Any, sample: str | None, model_name: str, confidence: float, compare: bool, second_model: str) -> tuple[Any, Any, str, Any, str]:
    image, label = resolve_input(uploaded, sample)
    if image is None:
        return None, None, "### No input\nUpload an image or choose a bundled sample.", None, f"Input unavailable: {label}."
    windows = tile_windows(image.shape[0], image.shape[1])
    grid = make_tile_grid(image, windows)
    summaries: list[str] = []
    statuses: list[str] = []
    primary_output: Any = image
    secondary_output: Any = None
    try:
        rendered, detections, windows, elapsed_ms = run_inference(image, model_name, confidence)
        primary_output = rendered
        counts = detection_counts(detections)
        summaries.append(format_detection_summary(model_name, label, image, detections, windows, elapsed_ms, confidence))
        statuses.append(f"{model_name}: {len(detections)} detections")
    except ModelUnavailable as exc:
        summaries.append(f"### `{model_name}` unavailable\n- {exc}\n- Add `demo/weights/{model_name}.pt` to enable this model.")
        statuses.append(str(exc))
    except Exception as exc:
        summaries.append(f"### `{model_name}` failed safely\n- {exc}")
        statuses.append(f"{model_name} failed: {exc}")

    if compare:
        if model_name == second_model:
            summaries.append("**Comparison:** the same model is selected twice; there is nothing to compare.")
        try:
            rendered, detections, _, elapsed_ms = run_inference(image, second_model, confidence)
            secondary_output = rendered
            summaries.append(format_detection_summary(second_model, label, image, detections, windows, elapsed_ms, confidence))
            statuses.append(f"{second_model}: {len(detections)} detections")
        except ModelUnavailable as exc:
            summaries.append(f"### `{second_model}` unavailable\n- {exc}\n- Add `demo/weights/{second_model}.pt` to enable this model.")
            statuses.append(str(exc))
        except Exception as exc:
            summaries.append(f"### `{second_model}` failed safely\n- {exc}")
            statuses.append(f"{second_model} failed: {exc}")
    else:
        statuses.append("Comparison disabled")
    return primary_output, secondary_output, "\n\n".join(summaries), grid, " · ".join(statuses)


def rotation_interpretation(model_one: str, model_two: str) -> str:
    """Compare only the selected checkpoints using the bundled s0 metrics."""
    if model_one == model_two:
        return f"**Interpretation:** `{model_one}` is selected twice; there is nothing to compare."
    try:
        data = json.loads((DEMO_DIR / "results_run2_geo.json").read_text(encoding="utf-8"))
        metrics = {name: {domain: data[f"vd_{name}_e30_s0|{domain}"]["AP50"]
                          for domain in ("DOTA", "VisDrone")} for name in (model_one, model_two)}
        comparisons = []
        for domain in ("DOTA", "VisDrone"):
            first, second = metrics[model_one][domain], metrics[model_two][domain]
            comparisons.append(f"{domain} AP50: `{model_one}` **{first:.3f}** versus `{model_two}` **{second:.3f}** ({first - second:+.3f}, first minus second)")
        return "**Interpretation — 1 seed (s0), results_run2_geo.json:** " + "; ".join(comparisons) + ". These rotations report counts on this image only, not held-out AP50."
    except (OSError, ValueError, KeyError, TypeError):
        return "**Interpretation:** bundled benchmark metrics are unavailable; no comparison is claimed."


def rotation_action(
    uploaded: Any,
    sample: str | None,
    confidence: float = DEFAULT_CONF,
    model_one: str = "base",
    model_two: str = "ctrlgeo",
) -> tuple[list[tuple[np.ndarray, str]], str, str]:
    image, label = resolve_input(uploaded, sample)
    if image is None:
        return [], "### No input\nUpload an image or choose a bundled sample.", f"Input unavailable: {label}."
    selected_models = tuple(name if name in MODEL_FILES else fallback for name, fallback in ((model_one, "base"), (model_two, "ctrlgeo")))
    gallery: list[tuple[np.ndarray, str]] = []
    rows: list[str] = [f"Input: {label}. The gallery is ordered as two rows: **{selected_models[0]}**, then **{selected_models[1]}**; columns are 0°, 90°, 180°, 270°."]
    statuses: list[str] = []
    for model_name in selected_models:
        count_text: list[str] = []
        for quarter_turns, angle in enumerate((0, 90, 180, 270)):
            rotated = np.ascontiguousarray(np.rot90(image, quarter_turns))
            try:
                rendered, detections, _, elapsed_ms = run_inference(rotated, model_name, confidence)
                counts = detection_counts(detections)
                label_text = f"{model_name} · {angle}° · {len(detections)} total"
                count_text.append(f"{angle}°: {len(detections)} ({counts.get('small-vehicle', 0)} small, {counts.get('large-vehicle', 0)} large; {elapsed_ms:.1f} ms)")
                statuses.append(f"{model_name} {angle}° ran")
            except ModelUnavailable as exc:
                rendered = rotated
                label_text = f"{model_name} · {angle}° · weights missing"
                count_text.append(f"{angle}°: not available")
                statuses.append(str(exc))
            except Exception as exc:
                rendered = rotated
                label_text = f"{model_name} · {angle}° · failed safely"
                count_text.append(f"{angle}°: failed safely")
                statuses.append(f"{model_name} {angle}° failed: {exc}")
            gallery.append((rendered, label_text))
        rows.append(f"**{model_name}:** " + " · ".join(count_text))
    rows.append(rotation_interpretation(*selected_models))
    return gallery, "\n\n".join(rows), " · ".join(statuses)


FREQ_MAX_SIDE = 1280


def frequency_action(uploaded: Any, sample: str | None, cutoff: float, level: str = "P3 (stride 8)") -> tuple[Any, Any, str, str]:
    """Image-level illustration of the freqgeo split.

    In the model the cut-off is in cycles per FEATURE CELL (0.20 for freqgeo) and the
    mask edge is sigmoid(24 * (cutoff - radius)).  One cell covers `stride` image pixels,
    so the equivalent image-level cut-off is cutoff / stride cycles per pixel and the
    equivalent edge sharpness is 24 * stride.
    """
    image, label = resolve_input(uploaded, sample)
    if image is None:
        return None, None, "### No input\nUpload an image or choose a bundled sample.", f"Input unavailable: {label}."
    stride = FREQ_LEVELS.get(level, 8)
    crop_note = ""
    full_h, full_w = image.shape[:2]
    if max(full_h, full_w) > FREQ_MAX_SIDE:
        y0 = max(0, (full_h - FREQ_MAX_SIDE) // 2)
        x0 = max(0, (full_w - FREQ_MAX_SIDE) // 2)
        image = np.ascontiguousarray(image[y0 : y0 + FREQ_MAX_SIDE, x0 : x0 + FREQ_MAX_SIDE])
        crop_note = f"- Large input: centre crop of {image.shape[1]}×{image.shape[0]} px at native scale (not resized)\n"
    source = image.astype(np.float32) / 255.0
    height, width = source.shape[:2]
    spectrum = np.fft.rfft2(source, axes=(0, 1), norm="ortho")
    fy = np.fft.fftfreq(height).reshape(height, 1, 1)
    fx = np.fft.rfftfreq(width).reshape(1, -1, 1)
    radius = np.sqrt(fy**2 + fx**2)
    cutoff_px = float(cutoff) / stride
    sharp_px = 24.0 * stride
    mask = 1.0 / (1.0 + np.exp(np.clip(-sharp_px * (cutoff_px - radius), -60.0, 60.0)))
    structure = np.fft.irfft2(spectrum * mask, s=(height, width), axes=(0, 1), norm="ortho").real
    residual = source - structure
    structure_view = np.clip(structure, 0.0, 1.0)
    scale = max(float(np.percentile(np.abs(residual), 99.0)), 1e-4)
    residual_view = np.clip(0.5 + residual / (2.0 * scale), 0.0, 1.0)
    period_px = stride / max(float(cutoff), 1e-6)
    caption = (
        "### Image-level illustration of the frequency split\n"
        f"- Input: {label} ({width}×{height} px)\n"
        f"{crop_note}"
        f"- Level **{level}**, cut-off **{float(cutoff):.2f} cycles per feature cell** (freqgeo uses 0.20) = {cutoff_px:.4f} cycles per image pixel\n"
        f"- Detail with a period shorter than about **{period_px:.0f} px** goes to the style residual; coarser structure stays. The 35 px vehicle example refers only to **P3 at cutoff 0.20** (about a 40 px period), not P4 or P5. Loss of fine detail is a possible explanation, not a proven cause of the recall gap.\n"
        "- **Structure** = low-frequency reconstruction; **style residual** = signed residual, contrast-stretched for viewing.\n"
        "- This is an analogy applied to the input image. The real split acts on learned feature maps inside `freqgeo`, so it is not the exact internal split."
    )
    return (structure_view * 255).astype(np.uint8), (residual_view * 255).astype(np.uint8), caption, "Frequency split computed locally; no model or network call used."


# ---------------------------------------------------------------------------
# Results loading and presentation
def _result_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _expected_values_by_variant() -> dict[str, dict[str, float]]:
    return {row["variant"]: {key: float(row[key]) for key in RESULT_HEADERS if key not in {"variant", "n"}} for row in BUILTIN_RESULTS}


def _json_candidate_paths() -> list[Path]:
    return _run2_json_paths()


def _rows_from_simple_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("variants"), list):
        data = data["variants"]
    if not isinstance(data, list):
        return []
    rows: list[dict[str, Any]] = []
    aliases = {
        "visdrone": "VisDrone AP50", "visdrone_ap50": "VisDrone AP50", "AP50_VisDrone": "VisDrone AP50",
        "dota": "DOTA AP50", "dota_ap50": "DOTA AP50", "AP50_DOTA": "DOTA AP50",
    }
    for item in data:
        if not isinstance(item, dict):
            continue
        variant = item.get("variant") or item.get("mode")
        if not variant:
            continue
        row: dict[str, Any] = {"variant": str(variant), "n": int(item.get("n", 1))}
        for target_key in RESULT_HEADERS[2:]:
            value = item.get(target_key)
            if value is None:
                for alias, canonical in aliases.items():
                    if canonical == target_key and alias in item:
                        value = item[alias]
                        break
            number = _result_number(value)
            if number is not None:
                row[target_key] = number
        if "VisDrone AP50" in row and "DOTA AP50" in row:
            rows.append(row)
    return rows


def _rows_from_run2_json(data: Any) -> list[dict[str, Any]]:
    """Extract aggregate table rows from evaluate_dg.py's per-test schema."""
    if not isinstance(data, dict):
        return []
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not (key.endswith("|VisDrone") or key.endswith("|DOTA")):
            continue
        if not isinstance(value, dict):
            continue
        mode = value.get("mode")
        seed = value.get("seed")
        test = value.get("test")
        if not mode or seed is None or test not in {"VisDrone", "DOTA"}:
            continue
        if _result_number(value.get("AP50")) is None:
            continue
        records[(str(mode), str(seed), str(test))] = value

    rows: list[dict[str, Any]] = []
    for mode in ("base", "ctrlgeo", "freqgeo"):
        vis = [record for (record_mode, _, test), record in records.items() if record_mode == mode and test == "VisDrone"]
        dota = [record for (record_mode, _, test), record in records.items() if record_mode == mode and test == "DOTA"]
        if not vis or not dota:
            continue

        def mean_field(items: list[dict[str, Any]], key: str) -> float | None:
            values = [_result_number(item.get(key)) for item in items]
            numeric = [value for value in values if value is not None]
            return None if not numeric else float(np.mean(numeric))

        seeds = {
            str(seed)
            for record_mode, seed, test in records
            if record_mode == mode and test in {"VisDrone", "DOTA"}
        }
        row: dict[str, Any] = {
            "variant": mode,
            "n": len(seeds),
            "VisDrone AP50": mean_field(vis, "AP50"),
            "DOTA AP50": mean_field(dota, "AP50"),
            "DOTA small AP50": mean_field(dota, "AP50_small"),
            "DOTA large AP50": mean_field(dota, "AP50_large"),
            "DOTA P": mean_field(dota, "P"),
            "DOTA R": mean_field(dota, "R"),
            "DOTA F1": mean_field(dota, "F1"),
        }
        if row["VisDrone AP50"] is not None and row["DOTA AP50"] is not None:
            rows.append(row)
    return rows


def load_results() -> tuple[list[dict[str, Any]], str]:
    """Read only the bundled evaluation; metadata lists such as |tau are ignored."""
    path = DEMO_DIR / "results_run2_geo.json"
    try:
        rows = _rows_from_run2_json(json.loads(path.read_text(encoding="utf-8")))
        return rows, "results_run2_geo.json · 1 seed (s0) · 30 epochs · YOLO11n"
    except (OSError, ValueError, TypeError):
        return [], "Bundled results unavailable; no fallback metrics are claimed."


RUN2_BUCKETS = ["<16 px", "16-32 px", "32-64 px", ">64 px"]
RUN2_PANEL_HEADERS = ["mode", "n seeds"] + [item for bucket in RUN2_BUCKETS for item in (f"{bucket} recall", f"{bucket} objects")]


def _run2_json_paths() -> list[Path]:
    """Never search parent folders or test fixtures for metrics."""
    path = DEMO_DIR / "results_run2_geo.json"
    return [path] if path.is_file() else []


def parse_run2_json(data: Any) -> list[dict[str, Any]]:
    """Validate and extract DOTA records from evaluate_dg.py's exact schema."""
    if not isinstance(data, dict):
        raise ValueError("top level must be an object")
    records: list[dict[str, Any]] = []
    for key, value in data.items():
        if not isinstance(key, str) or not key.endswith("|DOTA"):
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{key}: record must be an object")
        mode = value.get("mode")
        seed = value.get("seed")
        test = value.get("test")
        size_recall = value.get("size_recall")
        size_counts = value.get("size_counts")
        if not mode or seed is None or test != "DOTA":
            raise ValueError(f"{key}: expected mode, seed, and test='DOTA'")
        if not isinstance(size_recall, dict) or not isinstance(size_counts, dict):
            raise ValueError(f"{key}: size_recall and size_counts must be objects")
        recall_values: dict[str, float | None] = {}
        count_values: dict[str, int] = {}
        for bucket in RUN2_BUCKETS:
            if bucket not in size_recall or bucket not in size_counts:
                raise ValueError(f"{key}: missing size bucket {bucket!r}")
            recall = size_recall[bucket]
            recall_values[bucket] = None if recall is None else _result_number(recall)
            if recall is not None and recall_values[bucket] is None:
                raise ValueError(f"{key}: non-numeric recall for {bucket!r}")
            count = _result_number(size_counts[bucket])
            if count is None or count < 0:
                raise ValueError(f"{key}: invalid object count for {bucket!r}")
            count_values[bucket] = int(round(count))
        records.append({"mode": str(mode), "seed": str(seed), "recall": recall_values, "counts": count_values, "key": key})
    if not records:
        raise ValueError("no DOTA records were found")
    return records


def aggregate_run2_recall(records: list[dict[str, Any]]) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    """Mean DOTA recall over unique seeds per mode, with bucket counts."""
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        deduplicated[(record["mode"], record["seed"])] = record
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in deduplicated.values():
        grouped.setdefault(record["mode"], []).append(record)
    order = {name: index for index, name in enumerate(("base", "control", "geo", "ctrlgeo", "freqgeo"))}
    modes = sorted(grouped, key=lambda name: (order.get(name, 99), name))
    panel_groups: list[dict[str, Any]] = []
    table: list[list[Any]] = []
    for mode in modes:
        mode_records = grouped[mode]
        recalls: dict[str, float | None] = {}
        counts: dict[str, int] = {}
        for bucket in RUN2_BUCKETS:
            values = [record["recall"][bucket] for record in mode_records if record["recall"][bucket] is not None]
            recalls[bucket] = float(np.mean(values)) if values else None
            count_values = [record["counts"][bucket] for record in mode_records]
            counts[bucket] = int(round(float(np.mean(count_values)))) if count_values else 0
        panel_groups.append({"mode": mode, "n_seeds": len(mode_records), "recall": recalls, "counts": counts})
        row: list[Any] = [mode, len(mode_records)]
        for bucket in RUN2_BUCKETS:
            value = recalls[bucket]
            row.extend(["not available" if value is None else f"{value:.3f}", counts[bucket]])
        table.append(row)
    return table, panel_groups


def make_run2_recall_plot(groups: list[dict[str, Any]]):
    if not groups:
        return None
    names = [group["mode"] for group in groups]
    x = np.arange(len(names))
    width = 0.8 / len(RUN2_BUCKETS)
    colors = ("#7c3aed", "#0ea5e9", "#10b981", "#f59e0b")
    figure, axis = _new_chart((11, 5.3))
    for index, bucket in enumerate(RUN2_BUCKETS):
        values = [np.nan if group["recall"][bucket] is None else group["recall"][bucket] for group in groups]
        positions = x - 0.4 + width / 2 + index * width
        axis.bar(positions, values, width, label=bucket, color=colors[index])
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Recall at confidence 0.25")
    axis.set_title("Run 2 DOTA recall by object size — 1 seed (s0)")
    axis.set_xticks(x, names)
    axis.grid(axis="y", alpha=0.20, color="#385579")
    axis.legend(title="Object size")
    figure.tight_layout()
    return figure


def build_run2_recall_panel() -> tuple[list[list[Any]], Any, str]:
    paths = _run2_json_paths()
    if not paths:
        return [], None, "**Run 2 JSON only; base and control have one seed here.** No `results_run2_geo.json` was found; recall by object size is unavailable."
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records = parse_run2_json(data)
            table, groups = aggregate_run2_recall(records)
            source = path.relative_to(PROJECT_DIR) if path.is_relative_to(PROJECT_DIR) else path
            return table, make_run2_recall_plot(groups), f"**All three models: 1 seed (s0).** Source: `{source}`. DOTA recall at confidence 0.25; object columns are evaluated object counts in each bucket."
        except Exception as exc:
            return [], None, f"**Run 2 JSON only; base and control have one seed here.** Could not parse `{path.name}`: `{exc}`. The panel is unavailable; the report table above is unchanged."
    return [], None, "**Run 2 JSON only; base and control have one seed here.** No usable Run 2 JSON was found."


def make_results_plot(rows: list[dict[str, Any]]):
    names = [row["variant"] for row in rows]
    vis = [row["VisDrone AP50"] for row in rows]
    dota = [row["DOTA AP50"] for row in rows]
    x = np.arange(len(names))
    width = 0.38
    figure, axis = _new_chart((11, 5.3))
    axis.bar(x - width / 2, vis, width, label="VisDrone test", color="#3b82f6")
    axis.bar(x + width / 2, dota, width, label="DOTA test", color="#e5a11b")
    axis.set_ylim(0, 0.95)
    axis.set_ylabel("AP50")
    axis.set_title("Cross-domain AP50: local metrics where available")
    axis.set_xticks(x, names, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.20, color="#385579")
    axis.legend()
    for values, offset in ((vis, -width / 2), (dota, width / 2)):
        for position, value in zip(x, values):
            axis.text(position + offset, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#dce6f4")
    figure.tight_layout()
    return figure


def _new_chart(size: tuple[float, float]):
    """Create a matplotlib figure that remains legible inside the dark dashboard."""
    figure, axis = plt.subplots(figsize=size, dpi=120)
    figure.patch.set_facecolor("#0f1b2d")
    axis.set_facecolor("#0f1b2d")
    axis.tick_params(colors="#91a3ba", labelsize=9)
    axis.xaxis.label.set_color("#b8c6d8")
    axis.yaxis.label.set_color("#b8c6d8")
    axis.title.set_color("#f5f7fb")
    for spine in axis.spines.values():
        spine.set_color("#263b58")
    return figure, axis


def make_prf_plot(rows: list[dict[str, Any]]):
    """Plot the supplied DOTA precision, recall, and F1 values."""
    names = [row["variant"] for row in rows]
    x = np.arange(len(names))
    width = 0.24
    figure, axis = _new_chart((11, 5.3))
    for index, (label, key, color) in enumerate(
        (("Precision", "DOTA P", "#55d68a"), ("Recall", "DOTA R", "#5fa8ff"), ("F1", "DOTA F1", "#ff8a32"))
    ):
        values = [float(row[key]) for row in rows]
        offset = (index - 1) * width
        axis.bar(x + offset, values, width, label=label, color=color)
        for position, value in zip(x, values):
            axis.text(position + offset, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=7, color="#dce6f4")
    axis.set_ylim(0, 0.95)
    axis.set_ylabel("DOTA score")
    axis.set_title("DOTA precision, recall, and F1 by variant")
    axis.set_xticks(x, names, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.20, color="#385579")
    axis.legend(ncol=3)
    figure.tight_layout()
    return figure


def _row_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("variant")) == variant:
            return row
    return {}


def _format_metric(value: Any, fallback: str = "—") -> str:
    number = _result_number(value)
    return fallback if number is None else f"{number:.3f}"


def make_metric_cards(rows: list[dict[str, Any]]) -> str:
    base = _row_by_variant(rows, "base")
    ctrlgeo = _row_by_variant(rows, "ctrlgeo")
    base_dota = _format_metric(base.get("DOTA AP50"))
    ctrlgeo_dota = _format_metric(ctrlgeo.get("DOTA AP50"))
    base_vis = _format_metric(base.get("VisDrone AP50"))
    ctrlgeo_vis = _format_metric(ctrlgeo.get("VisDrone AP50"))
    dota_gain = _result_number(ctrlgeo.get("DOTA AP50"))
    base_dota_number = _result_number(base.get("DOTA AP50"))
    vis_change = _result_number(ctrlgeo.get("VisDrone AP50"))
    base_vis_number = _result_number(base.get("VisDrone AP50"))
    dota_delta = "—" if dota_gain is None or base_dota_number is None else f"{dota_gain - base_dota_number:+.3f} vs base"
    vis_delta = "—" if vis_change is None or base_vis_number is None else f"{vis_change - base_vis_number:+.3f} vs base"
    return f"""
    <section class="metric-strip" aria-label="Benchmark summary">
      <article class="metric-card metric-primary">
        <span class="metric-label">DOTA AP50 · ctrlgeo</span>
        <strong class="metric-value">{html_escape(ctrlgeo_dota)}</strong>
        <span class="metric-note positive">{html_escape(dota_delta)}</span>
      </article>
      <article class="metric-card">
        <span class="metric-label">DOTA AP50 · base</span>
        <strong class="metric-value">{html_escape(base_dota)}</strong>
        <span class="metric-note neutral">unseen target domain</span>
      </article>
      <article class="metric-card">
        <span class="metric-label">VisDrone AP50 · base → ctrlgeo</span>
        <strong class="metric-value">{html_escape(base_vis)} → {html_escape(ctrlgeo_vis)}</strong>
        <span class="metric-note negative">{html_escape(vis_delta)}</span>
      </article>
      <article class="metric-card">
        <span class="metric-label">Runtime</span>
        <strong class="metric-value">LOCAL</strong>
        <span class="metric-note positive">offline guarantee · no downloads</span>
      </article>
    </section>
    """


def make_takeaways(rows: list[dict[str, Any]]) -> str:
    base = _row_by_variant(rows, "base")
    ctrlgeo = _row_by_variant(rows, "ctrlgeo")
    freqgeo = _row_by_variant(rows, "freqgeo")
    base_dota = _result_number(base.get("DOTA AP50"))
    ctrlgeo_dota = _result_number(ctrlgeo.get("DOTA AP50"))
    base_vis = _result_number(base.get("VisDrone AP50"))
    ctrlgeo_vis = _result_number(ctrlgeo.get("VisDrone AP50"))
    freqgeo_dota = _result_number(freqgeo.get("DOTA AP50"))
    freqgeo_vis = _result_number(freqgeo.get("VisDrone AP50"))
    dota_gain = "—" if base_dota is None or ctrlgeo_dota is None else f"{ctrlgeo_dota - base_dota:+.3f}"
    vis_change = "—" if base_vis is None or ctrlgeo_vis is None else f"{ctrlgeo_vis - base_vis:+.3f}"
    freqgeo_gap = "—" if freqgeo_dota is None or ctrlgeo_dota is None else f"{freqgeo_dota - ctrlgeo_dota:+.3f}"
    freqgeo_vis_gap = "—" if freqgeo_vis is None or ctrlgeo_vis is None else f"{freqgeo_vis - ctrlgeo_vis:+.3f}"
    return f"""
    <div class="takeaway-box">
      <strong>Key takeaways</strong>
      <ul>
        <li>Rotation/geometric augmentation is the useful intervention in the reported aggregate benchmark.</li>
        <li><b>ctrlgeo</b> improves DOTA AP50 from <b>{_format_metric(base_dota)}</b> to <b>{_format_metric(ctrlgeo_dota)}</b> ({dota_gain}), while VisDrone AP50 changes from <b>{_format_metric(base_vis)}</b> to <b>{_format_metric(ctrlgeo_vis)}</b> ({vis_change}).</li>
        <li>The frequency split did not improve cross-domain performance over <b>ctrlgeo</b>: freqgeo DOTA AP50 is <b>{_format_metric(freqgeo_dota)}</b> ({freqgeo_gap}) and VisDrone AP50 is <b>{_format_metric(freqgeo_vis)}</b> ({freqgeo_vis_gap}).</li>
        <li>Base is best on VisDrone; freqgeo has the weakest VisDrone tiny-object recall. One seed (s0) does not quantify uncertainty or establish causality.</li>
      </ul>
    </div>
    """


def make_results_table(rows: list[dict[str, Any]]) -> str:
    labels = {
        "variant": "Variant", "n": "Seeds", "VisDrone AP50": "VisDrone AP50", "DOTA AP50": "DOTA AP50",
        "DOTA small AP50": "DOTA small AP50", "DOTA large AP50": "DOTA large AP50",
        "DOTA P": "DOTA P", "DOTA R": "DOTA R", "DOTA F1": "DOTA F1",
    }
    head = "".join(f"<th>{html_escape(labels[key])}</th>" for key in RESULT_HEADERS)
    body: list[str] = []
    for row in rows:
        variant = str(row.get("variant", ""))
        classes = []
        if variant == "ctrlgeo":
            classes.append("best-row")
        if variant == "DOTA reference":
            classes.append("reference-row")
        cells: list[str] = []
        for key in RESULT_HEADERS:
            value = row.get(key, "not available")
            display = ("1 seed (s0)" if value == 1 else f"mean of {value} seeds") if key == "n" else (str(value) if key == "variant" else _format_metric(value, "not available"))
            cells.append(f"<td>{html_escape(display)}</td>")
        body.append(f"<tr class=\"{' '.join(classes)}\">{''.join(cells)}</tr>")
    return f"""
    <div class="benchmark-table-wrap" role="region" aria-label="Exact benchmark table" tabindex="0">
      <table class="benchmark-table">
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    """


def load_dashboard_css() -> str:
    try:
        return CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def dashboard_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="slate",
        font=("Inter", "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=("ui-monospace", "SFMono-Regular", "Consolas", "monospace"),
    )


RESULT_ROWS, RESULTS_SOURCE = load_results()
RUN2_PANEL_TABLE, RUN2_PANEL_PLOT, RUN2_PANEL_STATUS = build_run2_recall_panel()


_initialise_models()


def _result_matrix(rows: list[dict[str, Any]]) -> list[list[Any]]:
    matrix: list[list[Any]] = []
    for row in rows:
        values: list[Any] = []
        for header in RESULT_HEADERS:
            value = row.get(header, "not available")
            if header in {"variant", "n"} or value == "not available":
                values.append(value)
            else:
                values.append(f"{float(value):.3f}")
        matrix.append(values)
    return matrix


LIMITATIONS_MARKDOWN = """### Limitations

- **1 seed (s0)**, **30 epochs**, **YOLO11n**; no confidence intervals or multi-seed certainty claimed.
- Two classes only: **small-vehicle** (car, van) and **large-vehicle** (truck, bus).
- VisDrone recall below **16 px**: base **0.489**, ctrlgeo **0.364**, freqgeo **0.268** (1 seed (s0), bundled JSON). The frequency model is weakest; causality is not established by this comparison.
- DOTA oriented boxes were converted to **axis-aligned** boxes, which can inflate boxes and affect NMS.
- The models were trained from one source domain (VisDrone) and evaluated on an unseen DOTA domain; pseudo-style perturbations are not a real second sensor.
- The live viewer is an inference demonstration, not a replacement for the held-out benchmark. Detection counts on a selected image can be less dramatic than the aggregate AP50 table.
"""


def build_app() -> gr.Blocks:
    sample_choices = available_sample_names()
    version_text = runtime_versions()
    sample_note = (
        f"{len(sample_choices)} bundled sample tile(s) available."
        if sample_choices
        else "No bundled sample tiles found — upload-only mode is active."
    )
    results_source = html_escape(RESULTS_SOURCE)
    hero_html = f"""
    <section class="app-hero">
      <div class="hero-row">
        <div>
          <div class="hero-kicker">Research demo · cross-domain generalization</div>
          <h1 class="hero-title">Cross-domain aerial vehicle detection</h1>
          <p class="hero-copy">YOLO11n models trained on VisDrone tiles are evaluated on unseen DOTA-style imagery. The viewer keeps tiles at native scale, makes the geometry shift visible, and stays local-only at runtime.</p>
        </div>
        <div class="offline-pill">{html_escape(readiness_status())}</div>
      </div>
    </section>
    """
    runtime_html = f"""
    <div class="status-box">
      <strong>Runtime</strong> · Python {html_escape(version_text['python'])} · torch {html_escape(version_text['torch'])} · Ultralytics {html_escape(version_text['ultralytics'])} · Gradio {html_escape(version_text['gradio'])} · OpenCV {html_escape(version_text['opencv'])}<br>
      <span class="helper-text">{html_escape(sample_note)} Checkpoint status is available below. Missing assets are reported explicitly; no model or sample is substituted.</span>
    </div>
    """
    protocol_html = """
    <div class="protocol-row" aria-label="Inference protocol">
      <span class="protocol-chip"><b>Image size</b> 640</span>
      <span class="protocol-chip"><b>Confidence</b> 0.25</span>
      <span class="protocol-chip"><b>NMS IoU</b> 0.60</span>
      <span class="protocol-chip"><b>Classes</b> 2 vehicle groups</span>
    </div>
    """
    legend_html = """
    <div class="legend-box">
      <strong>Bounding-box legend</strong><br>
      <span class="legend-item"><span class="legend-swatch small"></span> small-vehicle</span>
      <span class="legend-item"><span class="legend-swatch large"></span> large-vehicle</span>
      <span class="helper-text">Counts and boxes come from the selected image only.</span>
    </div>
    """
    takeaways_html = make_takeaways(RESULT_ROWS)
    run2_status_class = "status-box" if RUN2_PANEL_TABLE else "empty-state"

    with gr.Blocks(
        title="Cross-domain aerial vehicle detection",
        fill_width=True,
        analytics_enabled=False,
    ) as demo:
        with gr.Column(elem_id="app-shell"):
            gr.HTML(hero_html)
            gr.HTML(make_metric_cards(RESULT_ROWS))
            with gr.Row(elem_classes=["runtime-grid"]):
                with gr.Column(scale=2, elem_classes=["runtime-card"]):
                    gr.HTML(runtime_html)
                with gr.Column(scale=1, elem_classes=["scope-card"]):
                    gr.HTML("<div class=\"status-box\"><strong>Experiment lens</strong><br><span class=\"helper-text\">Source-only VisDrone training · unseen DOTA-style evaluation · results_run2_geo.json · 1 seed (s0) · 30 epochs · YOLO11n.</span></div>")
            with gr.Accordion("Checkpoint and runtime details", open=False, elem_classes=["runtime-card"]):
                gr.Markdown(model_status_lines())

            with gr.Tabs(elem_classes=["app-tabs"]):
                with gr.Tab("Detect", id="detect"):
                    with gr.Column(elem_classes=["page-shell"]):
                        gr.HTML("""
                        <div class="page-heading">
                          <div>
                            <div class="page-kicker">01 · inference workspace</div>
                            <h2 class="page-title">Detect</h2>
                            <p class="page-description">Upload one image or choose a bundled local sample. Large images are processed through overlapping native-scale 640 px tiles and merged back into full-image coordinates.</p>
                          </div>
                        </div>
                        """)
                        with gr.Row(elem_classes=["workspace-grid"]):
                            with gr.Column(scale=7, elem_classes=["panel-card"]):
                                gr.HTML("<div class=\"section-label\">Input image <span>drag and drop or browse</span></div>")
                                detect_upload = gr.Image(show_label=False, type="numpy", image_mode="RGB", height=250, sources=["upload", "clipboard"], elem_classes=["upload-zone"])
                                detect_sample = gr.Dropdown(choices=sample_choices, value=None, label="Bundled sample (optional)", allow_custom_value=False)
                                gr.Markdown(sample_note, elem_classes=["sample-note"])
                            with gr.Column(scale=5, elem_classes=["control-card"]):
                                gr.HTML("<div class=\"section-label\">Model settings <span>preserved while switching tabs</span></div>")
                                detect_model = gr.Dropdown(choices=list(MODEL_FILES), value="base", label="Primary model")
                                detect_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.01, label="Confidence threshold", info="Applied to each native-scale tile")
                                detect_compare = gr.Checkbox(value=False, label="Compare a second model")
                                detect_second = gr.Dropdown(choices=list(MODEL_FILES), value="ctrlgeo", label="Secondary model", visible=False)
                        detect_button = gr.Button("Run Detection", variant="primary", elem_classes=["primary-action"])
                        with gr.Row(elem_classes=["result-grid"]):
                            with gr.Column(elem_classes=["result-card"]):
                                gr.HTML("<div class=\"section-label\">Primary visualization <span>annotated full image</span></div>")
                                detect_output = gr.Image(show_label=False, type="numpy", height=320, elem_classes=["result-image"])
                            with gr.Column(elem_classes=["result-card"], visible=False) as compare_card:
                                gr.HTML("<div class=\"section-label\">Comparison visualization <span>optional second model</span></div>")
                                compare_output = gr.Image(show_label=False, type="numpy", height=320, visible=False, elem_classes=["result-image"])
                        with gr.Row(elem_classes=["result-grid"]):
                            with gr.Column(scale=2, elem_classes=["result-card"]):
                                gr.HTML("<div class=\"section-label\">Detection summary <span>counts · dimensions · latency</span></div>")
                                detect_summary = gr.Markdown(elem_classes=["status-box"])
                            with gr.Column(scale=1, elem_classes=["result-card"]):
                                gr.HTML("<div class=\"section-label\">Native-scale tile grid <span>overlap windows</span></div>")
                                tile_grid = gr.Image(show_label=False, type="numpy", height=220, elem_classes=["tile-image"])
                        detect_status = gr.Markdown(elem_classes=["status-box"])
                        detect_compare.change(
                            lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled)),
                            inputs=detect_compare,
                            outputs=[detect_second, compare_output, compare_card],
                            show_progress="hidden",
                        )
                        detect_button.click(
                            detect_action,
                            inputs=[detect_upload, detect_sample, detect_model, detect_conf, detect_compare, detect_second],
                            outputs=[detect_output, compare_output, detect_summary, tile_grid, detect_status],
                            api_name="detect_action",
                            show_progress="full",
                            trigger_mode="once",
                            concurrency_limit=1,
                        )

                with gr.Tab("Rotation Robustness", id="rotation"):
                    with gr.Column(elem_classes=["page-shell"]):
                        gr.HTML("""
                        <div class="page-heading">
                          <div>
                            <div class="page-kicker">02 · geometric stress test</div>
                            <h2 class="page-title">Rotation Robustness</h2>
                            <p class="page-description">Run the actual inference pipeline on 0°, 90°, 180°, and 270° versions of the selected image. The grid is a sample-level diagnostic; the aggregate benchmark remains on the Results page.</p>
                          </div>
                        </div>
                        """)
                        with gr.Row(elem_classes=["workspace-grid"]):
                            with gr.Column(scale=7, elem_classes=["panel-card"]):
                                gr.HTML("<div class=\"section-label\">Input image <span>one source image · eight outputs</span></div>")
                                rotation_upload = gr.Image(show_label=False, type="numpy", image_mode="RGB", height=220, sources=["upload", "clipboard"], elem_classes=["upload-zone"])
                                rotation_sample = gr.Dropdown(choices=sample_choices, value=None, label="Bundled sample (optional)", allow_custom_value=False)
                            with gr.Column(scale=5, elem_classes=["control-card"]):
                                gr.HTML("<div class=\"section-label\">Comparison settings <span>defaults: base vs ctrlgeo</span></div>")
                                rotation_model_one = gr.Dropdown(choices=list(MODEL_FILES), value="base", label="Model 1")
                                rotation_model_two = gr.Dropdown(choices=list(MODEL_FILES), value="ctrlgeo", label="Model 2")
                                rotation_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.01, label="Confidence threshold")
                        rotation_button = gr.Button("Run 2×4 Rotation Grid", variant="primary", elem_classes=["primary-action"])
                        rotation_gallery = gr.Gallery(show_label=False, columns=4, rows=2, height=560, object_fit="contain", elem_classes=["rotation-gallery"])
                        with gr.Row(elem_classes=["result-grid"]):
                            with gr.Column(scale=2, elem_classes=["result-card"]):
                                rotation_summary = gr.Markdown(elem_classes=["status-box"])
                            with gr.Column(scale=1, elem_classes=["result-card"]):
                                gr.HTML(legend_html)
                        rotation_status = gr.Markdown(elem_classes=["status-box"])
                        rotation_button.click(
                            rotation_action,
                            inputs=[rotation_upload, rotation_sample, rotation_conf, rotation_model_one, rotation_model_two],
                            outputs=[rotation_gallery, rotation_summary, rotation_status],
                            api_name="rotation_action",
                            show_progress="full",
                            trigger_mode="once",
                            concurrency_limit=1,
                        )

                with gr.Tab("Frequency Split Viewer", id="frequency"):
                    with gr.Column(elem_classes=["page-shell"]):
                        gr.HTML("""
                        <div class="page-heading">
                          <div>
                            <div class="page-kicker">03 · Fourier intuition</div>
                            <h2 class="page-title">Frequency Split Viewer</h2>
                            <p class="page-description">An image-level Fourier illustration of structure and residual detail. It uses the model's frequency units and pyramid strides, but it is not the learned feature-map split inside the detector.</p>
                          </div>
                        </div>
                        """)
                        with gr.Row(elem_classes=["workspace-grid"]):
                            with gr.Column(scale=7, elem_classes=["panel-card"]):
                                gr.HTML("<div class=\"section-label\">Input image <span>local-only analysis · no checkpoint required</span></div>")
                                freq_upload = gr.Image(show_label=False, type="numpy", image_mode="RGB", height=230, sources=["upload", "clipboard"], elem_classes=["upload-zone"])
                                freq_sample = gr.Dropdown(choices=sample_choices, value=None, label="Bundled sample (optional)", allow_custom_value=False)
                            with gr.Column(scale=5, elem_classes=["control-card"]):
                                gr.HTML("<div class=\"section-label\">Split controls <span>match the experiment's units</span></div>")
                                freq_cutoff = gr.Slider(0.05, 0.30, value=0.20, step=0.01, label="Cut-off frequency", info="Cycles per feature cell")
                                freq_level = gr.Radio(choices=list(FREQ_LEVELS), value="P3 (stride 8)", label="Pyramid level")
                                gr.Markdown("The selected stride converts the feature-cell cut-off into image-pixel frequency for this illustration.", elem_classes=["helper-text"])
                        freq_button = gr.Button("Show Structure and Residual", variant="primary", elem_classes=["primary-action"])
                        with gr.Row(elem_classes=["result-grid"]):
                            with gr.Column(elem_classes=["result-card"]):
                                gr.HTML("<div class=\"section-label\">Low-frequency structure <span>coarse content retained</span></div>")
                                freq_structure = gr.Image(show_label=False, type="numpy", height=330, elem_classes=["result-image"])
                            with gr.Column(elem_classes=["result-card"]):
                                gr.HTML("<div class=\"section-label\">High-frequency residual <span>contrast-stretched for viewing</span></div>")
                                freq_residual = gr.Image(show_label=False, type="numpy", height=330, elem_classes=["result-image"])
                        freq_caption = gr.Markdown(elem_classes=["status-box"])
                        freq_status = gr.Markdown(elem_classes=["status-box"])
                        freq_button.click(
                            frequency_action,
                            inputs=[freq_upload, freq_sample, freq_cutoff, freq_level],
                            outputs=[freq_structure, freq_residual, freq_caption, freq_status],
                            api_name="frequency_action",
                            show_progress="full",
                            trigger_mode="once",
                            concurrency_limit=1,
                        )

                with gr.Tab("Results", id="results"):
                    with gr.Column(elem_classes=["page-shell"]):
                        gr.HTML("""
                        <div class="page-heading">
                          <div>
                            <div class="page-kicker">04 · held-out benchmark</div>
                            <h2 class="page-title">Results</h2>
                          <p class="page-description">YOLO11n models trained on VisDrone are evaluated on unseen DOTA-style imagery. All metrics come from results_run2_geo.json: 1 seed (s0), 30 epochs, two classes. No PDF averages are mixed in.</p>
                          </div>
                          <span class="source-pill">Source-only evaluation</span>
                        </div>
                        """)
                        gr.HTML(protocol_html)
                        gr.HTML(make_metric_cards(RESULT_ROWS))
                        gr.HTML(f"<div class=\"helper-text\">Results source: <code>{results_source}</code></div>")
                        gr.HTML(make_results_table(RESULT_ROWS))
                        with gr.Row(elem_classes=["chart-grid"]):
                            with gr.Column(elem_classes=["chart-panel"]):
                                gr.HTML("<div class=\"section-label\">AP50 comparison <span>VisDrone versus DOTA</span></div>")
                                gr.Plot(value=make_results_plot(RESULT_ROWS), show_label=False)
                            with gr.Column(elem_classes=["chart-panel"]):
                                gr.HTML("<div class=\"section-label\">DOTA quality metrics <span>precision · recall · F1</span></div>")
                                gr.Plot(value=make_prf_plot(RESULT_ROWS), show_label=False)
                        gr.HTML("<div class=\"section-label\">Recall by object size <span>Run 2 JSON only</span></div>")
                        gr.Markdown(RUN2_PANEL_STATUS, elem_classes=[run2_status_class])
                        gr.Dataframe(
                            value=RUN2_PANEL_TABLE,
                            headers=RUN2_PANEL_HEADERS,
                            interactive=False,
                            wrap=True,
                            show_label=False,
                            visible=bool(RUN2_PANEL_TABLE),
                            elem_classes=["result-card"],
                        )
                        gr.Plot(value=RUN2_PANEL_PLOT, show_label=False, visible=RUN2_PANEL_PLOT is not None, elem_classes=["chart-panel"])
                        with gr.Row(elem_classes=["result-grid"]):
                            with gr.Column(elem_classes=["result-card"]):
                                gr.HTML(takeaways_html)
                            with gr.Column(elem_classes=["result-card"]):
                                gr.Markdown(LIMITATIONS_MARKDOWN, elem_classes=["limitations-box"])

            gr.HTML("<footer class=\"offline-footer\"><b>Offline guarantee</b> · This dashboard reads local files under <code>demo/</code> and the selected upload only. It does not download checkpoints, samples, fonts, or data at runtime.</footer>")
    return demo


APP = build_app()


def choose_port(host: str = "127.0.0.1", preferred: int = 7860) -> int:
    import socket
    for port in [preferred] + [p for p in range(7860, 7871) if p != preferred]:
        try:
            with socket.socket() as probe:
                if os.name == "nt":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind((host, port))
            return port
        except OSError:
            continue
    raise OSError("No free demo port: close an application using ports 7860-7870 and retry.")


def _open_when_ready(url: str) -> None:
    import urllib.request
    import webbrowser
    for _ in range(180):
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(1)
    print(f"Browser not opened: server did not become ready. Check the console, or open {url} manually.", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the offline cross-domain vehicle detector demo.")
    parser.add_argument("--open-browser", action="store_true", help="Open the local URL after the server responds")
    parser.add_argument("--server-name", default=os.environ.get("DEMO_HOST", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.environ.get("DEMO_PORT", "7860")))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    port = choose_port(args.server_name, args.server_port)
    url = f"http://127.0.0.1:{port}"
    print(readiness_status(), flush=True)
    print(f"Demo URL: {url}", flush=True)
    if args.open_browser:
        import threading
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    APP.launch(
        server_name=args.server_name,
        server_port=port,
        share=False,
        show_error=False,
        css=load_dashboard_css(),
        theme=dashboard_theme(),
    )
