"""Repeatable offline verification for the demo.

This test never substitutes generated imagery for detector verification. When
real VisDrone/DOTA tiles are present it runs every loaded checkpoint on them;
when they are absent it reports real-inference verification as blocked while
still checking model loading, pure helpers, and no-input UI error paths.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# freq_dg imports Ultralytics before app; protect this test's early imports too.
DEMO_DIR = Path(__file__).resolve().parent
for key, value in {"GRADIO_ANALYTICS_ENABLED": "False", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "YOLO_OFFLINE": "True", "YOLO_AUTOINSTALL": "False"}.items():
    os.environ[key] = value
for key, subdir in {"YOLO_CONFIG_DIR": ".cache/ultralytics", "MPLCONFIGDIR": ".cache/matplotlib", "GRADIO_TEMP_DIR": ".cache/gradio", "HF_HOME": ".cache/huggingface", "TEMP": ".cache/tmp", "TMP": ".cache/tmp", "TMPDIR": ".cache/tmp"}.items():
    path = DEMO_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)

import numpy as np
PROJECT_DIR = DEMO_DIR.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import freq_dg  # noqa: E402,F401 — must precede app/YOLO checkpoint loading
import app  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXCLUDED_PARTS = {"proof", "proof_redesign", "__pycache__", "tests", "old"}


def real_sample_paths(limit: int = 3) -> list[Path]:
    """Find only plausible local VisDrone/DOTA/sample tiles; never generate one."""
    candidates: list[Path] = []
    for path in sorted(PROJECT_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = {part.lower() for part in path.relative_to(PROJECT_DIR).parts}
        if parts & EXCLUDED_PARTS:
            continue
        relative = str(path.relative_to(PROJECT_DIR)).lower().replace("\\", "/")
        under_demo_samples = path.is_relative_to(DEMO_DIR / "samples") if (DEMO_DIR / "samples").exists() else False
        dataset_named = any(token in relative for token in ("visdrone", "dota", "data_shared"))
        if under_demo_samples or dataset_named:
            candidates.append(path)
    return candidates[:limit]


def available_inputs() -> list[tuple[Path, np.ndarray]]:
    inputs: list[tuple[Path, np.ndarray]] = []
    for path in real_sample_paths():
        image = app._read_image_path(path)
        if image is not None:
            inputs.append((path, image))
    return inputs


def assert_detection_shape(image: np.ndarray, detections: list[dict], windows: list[tuple[int, int, int, int]], latency: float) -> None:
    assert isinstance(image, np.ndarray), "rendered output is not an ndarray"
    assert image.ndim == 3 and image.shape[2] == 3, f"bad rendered shape: {image.shape}"
    assert isinstance(detections, list), "detections is not a list"
    assert windows and all(len(window) == 4 for window in windows), "invalid tile windows"
    assert latency >= 0.0, "negative latency"
    for detection in detections:
        assert np.isfinite(np.asarray(detection["box"], dtype=np.float32)).all(), "non-finite box"
        assert 0.0 <= float(detection["score"]) <= 1.0, "invalid confidence"
        assert int(detection["class_id"]) in (0, 1), "unexpected class id"


def main() -> int:
    print("SELFTEST: app imported")
    print(f"SELFTEST: freq_dg imported from {freq_dg.__file__}")
    print(f"SELFTEST: device={app.DEVICE}")
    print(app.model_status_lines())
    real_inputs = available_inputs()
    if real_inputs:
        print("SELFTEST: real image tiles:")
        for path, image in real_inputs:
            print(f"- {path.relative_to(PROJECT_DIR)} ({image.shape[1]}×{image.shape[0]} px)")
    else:
        print("SELFTEST: REAL INFERENCE BLOCKED — no local VisDrone/DOTA image tiles found under demo/samples or the project workspace")

    # Pure helper boundaries: a small image is one padded tile, a wide image
    # produces multiple overlapping windows, and NMS is class-aware.
    assert app.tile_windows(640, 640) == [(0, 0, 640, 640)]
    assert len(app.tile_windows(721, 1190)) > 1
    duplicate = [
        {"box": (0, 0, 100, 100), "score": 0.9, "class_id": 0},
        {"box": (2, 2, 98, 98), "score": 0.8, "class_id": 0},
        {"box": (2, 2, 98, 98), "score": 0.7, "class_id": 1},
    ]
    assert len(app.merge_detections(duplicate)) == 2
    print("SELFTEST: tiling and class-aware NMS helpers passed")

    # Authoritative metrics only: ignore the tau metadata list and compare all pairs.
    rows, source = app.load_results()
    assert [row["variant"] for row in rows] == ["base", "ctrlgeo", "freqgeo"]
    assert [round(row["DOTA AP50"], 3) for row in rows] == [0.538, 0.648, 0.558]
    assert [round(row["VisDrone AP50"], 3) for row in rows] == [0.609, 0.518, 0.494]
    assert "1 seed (s0)" in source
    for first in app.MODEL_FILES:
        for second in app.MODEL_FILES:
            text = app.rotation_interpretation(first, second)
            assert ("nothing to compare" in text) if first == second else ("1 seed (s0)" in text and first in text and second in text)
    print("SELFTEST: JSON AP50/tau handling and all nine rotation pairings passed")

    # Regression: merge must stay fast on dense input (the old version took minutes at ~5,000 boxes).
    rng = np.random.default_rng(3)
    dense = [
        {"box": (float(x), float(y), float(x + 30), float(y + 30)), "score": float(s), "class_id": int(c)}
        for x, y, s, c in zip(rng.uniform(0, 4000, 5000), rng.uniform(0, 4000, 5000), rng.uniform(0.25, 1, 5000), rng.integers(0, 2, 5000))
    ]
    started = time.perf_counter()
    app.merge_detections(dense)
    merge_seconds = time.perf_counter() - started
    assert merge_seconds < 5.0, f"merge too slow: {merge_seconds:.1f}s for 5000 detections"
    print(f"SELFTEST: merge of 5000 detections took {merge_seconds:.2f}s")

    # Pure regression probe: Ultralytics expects BGR numpy input; this is not a detector result.
    seen: dict = {}

    class _Probe:
        def predict(self, source=None, **kwargs):
            seen["pixel"] = tuple(int(v) for v in source[0, 0])
            return []

    app.MODEL_REGISTRY["__probe"] = {"model": _Probe(), "status": "ready"}
    red_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    red_rgb[..., 0] = 255
    app.run_inference(red_rgb, "__probe", 0.25)
    del app.MODEL_REGISTRY["__probe"]
    assert seen["pixel"] == (0, 0, 255), f"model input is not BGR: {seen['pixel']}"
    print("SELFTEST: model input channel order (BGR) passed")

    # Exercise callbacks with a real tile when available. Without one, verify the
    # readable no-input path instead of substituting generated imagery.
    if real_inputs:
        source = real_inputs[0][1]
        primary, secondary, summary, grid, status = app.detect_action(source, None, "base", 0.25, True, "ctrlgeo")
        assert isinstance(primary, np.ndarray) and isinstance(summary, str) and isinstance(grid, np.ndarray) and isinstance(status, str)
        assert secondary is None or isinstance(secondary, np.ndarray)
        gallery, rotation_summary, rotation_status = app.rotation_action(source, None, 0.25)
        assert len(gallery) == 8 and isinstance(rotation_summary, str) and isinstance(rotation_status, str)
        selectable_gallery, selectable_summary, selectable_status = app.rotation_action(source, None, 0.25, "ctrlgeo", "base")
        assert len(selectable_gallery) == 8 and "ctrlgeo" in selectable_summary and isinstance(selectable_status, str)
        structure, residual, frequency_summary, frequency_status = app.frequency_action(source, None, 0.20)
        assert isinstance(structure, np.ndarray) and isinstance(residual, np.ndarray)
        assert structure.shape == source.shape and residual.shape == source.shape
        assert isinstance(frequency_summary, str) and isinstance(frequency_status, str)
        print("SELFTEST: Detect, Rotation robustness, and Frequency split callbacks ran on a real tile")
    else:
        primary, secondary, summary, grid, status = app.detect_action(None, None, "base", 0.25, True, "ctrlgeo")
        assert primary is None and secondary is None and grid is None and isinstance(summary, str) and isinstance(status, str)
        gallery, rotation_summary, rotation_status = app.rotation_action(None, None, 0.25, "base", "ctrlgeo")
        assert gallery == [] and isinstance(rotation_summary, str) and isinstance(rotation_status, str)
        structure, residual, frequency_summary, frequency_status = app.frequency_action(None, None, 0.20, "P4 (stride 16)")
        assert structure is None and residual is None and isinstance(frequency_summary, str) and isinstance(frequency_status, str)
        print("SELFTEST: no-input Detect, Rotation robustness, and Frequency split error paths passed; no real tile was substituted")

    loaded = 0
    for model_name in app.MODEL_FILES:
        record = app.MODEL_REGISTRY[model_name]
        if record.get("model") is None:
            raise AssertionError(f"{model_name} failed to load: {record.get('error')}")
        loaded += 1
        model = record["model"]
        names = getattr(model, "names", None)
        normalized_names = set(names.values()) if isinstance(names, dict) else set(names or [])
        assert normalized_names == {"small-vehicle", "large-vehicle"}, f"{model_name}: unexpected classes {names!r}"
        model_class = getattr(getattr(model, "model", None), "__class__", type(None)).__name__
        if model_name in {"ctrlgeo", "freqgeo"}:
            assert model_class == "FreqDetectionModel", f"{model_name}: expected FreqDetectionModel, got {model_class}"
        print(f"SELFTEST: {model_name}: loaded; class={model_class}; names={names}")

        if real_inputs:
            latencies: list[float] = []
            for index, (path, source) in enumerate(real_inputs, start=1):
                started = time.perf_counter()
                rendered, detections, windows, latency = app.run_inference(source, model_name, 0.25)
                wall_ms = (time.perf_counter() - started) * 1000.0
                assert_detection_shape(rendered, detections, windows, latency)
                latencies.append(wall_ms)
                counts = app.detection_counts(detections)
                print(
                    f"SELFTEST: {model_name} {path.relative_to(PROJECT_DIR)}: "
                    f"{len(detections)} detections "
                    f"(small={counts.get('small-vehicle', 0)}, large={counts.get('large-vehicle', 0)}), "
                    f"{wall_ms:.1f} ms wall / {latency:.1f} ms reported"
                )
            print(f"SELFTEST: {model_name}: real inference PASS; mean wall latency={np.mean(latencies):.1f} ms")

    if real_inputs:
        print(f"SELFTEST: real model inference PASS for {loaded}/3 loaded models")
    else:
        print("SELFTEST: model loading PASS for 3/3 models; real model inference BLOCKED because no local aerial image tiles were found")
    print("SELFTEST: PASS (with real-inference blocker reported above)" if not real_inputs else "SELFTEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
