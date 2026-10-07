"""Explicit artifact workflows. None runs during webcam startup."""

import json
import platform
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml

from .backends.ultralytics import synchronize, yolo_class
from .config import Config
from .contracts import AppError
from .detector import checksum, load_detector, read_bundle
from .postprocessing import decode_detections


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: version(name)
            for name in (
                "ultralytics",
                "torch",
                "torchvision",
                "opencv-python",
                "numpy",
                "PyYAML",
            )
        },
    }


def resolved_dataset(data: Path, output: Path) -> Path:
    """Resolve YAML paths ourselves; the framework's dataset cache root may differ."""
    raw = yaml.safe_load(data.read_text())
    raw["path"] = str((data.resolve().parent / raw["path"]).resolve())
    output.mkdir(parents=True, exist_ok=True)
    path = output / "resolved_dataset.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def package_model(artifact: Path, output: Path, provenance: dict, input_size: int = 640) -> Path:
    if not artifact.is_file() or artifact.suffix not in (".pt", ".onnx"):
        raise AppError("Package a trusted local .pt or .onnx detection artifact.")
    model = yolo_class()(str(artifact.resolve()), task="detect")
    names = model.names
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    output.mkdir(parents=True, exist_ok=True)
    target = output / artifact.name
    if target.resolve() != artifact.resolve():
        if target.exists():
            raise AppError("Output artifact already exists; choose a new bundle directory.")
        shutil.copy2(artifact, target)
    manifest = output / "bundle.json"
    if manifest.exists():
        raise AppError("Bundle already exists; use a new versioned directory.")
    write_json(
        manifest,
        {
            "schema_version": 1,
            "artifact": target.name,
            "sha256": checksum(target),
            "classes": list(names),
            "input_size": input_size,
            "preprocessing_owner": "ultralytics",
            "postprocessing_owner": "ultralytics",
            "coordinates": "original_xyxy",
            "color_order": "BGR",
            "dtype": "uint8",
            "tensor_input": "backend-owned fixed square letterbox",
            "created_utc": datetime.now(UTC).isoformat(),
            "provenance": provenance,
            "environment": environment(),
            "validated_custom_accuracy": False,
        },
    )
    return manifest


def setup_model(output: Path) -> Path:
    """Fetch one fixed official release asset, then record its identity locally.

    A checksum records reproducibility, not an independent signature/attestation.
    Never download arbitrary user URLs or use model names during normal startup.
    """
    import requests

    url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "yolo26n.pt"
    if (output / "bundle.json").exists():
        read_bundle(output / "bundle.json")
        return output / "bundle.json"
    if artifact.exists():
        raise AppError(
            "Unbundled artifact exists. Remove it or package it explicitly with provenance."
        )
    temporary = output / "yolo26n.pt.part"
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            total = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    total += len(chunk)
                    if total > 100 * 1024 * 1024:
                        raise AppError("Unexpectedly large model download.")
                    handle.write(chunk)
        temporary.replace(artifact)
    except (requests.RequestException, OSError) as exc:
        raise AppError(
            "Model download failed. Check network access and retry setup-model."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return package_model(
        artifact,
        output,
        {
            "source": url,
            "dataset": "COCO pretrained; not project fine-tuned",
            "terms": "Ultralytics AGPL-3.0 / Enterprise; review before redistribution",
        },
    )


def benchmark(config: Config, images: list[Path], warmup: int, iterations: int) -> dict:
    if warmup < 1 or iterations < 1 or not images:
        raise AppError("Benchmark requires images and positive warmup/iterations.")
    detector = load_detector(config)
    elapsed, stage_values = [], []
    try:
        for image in images:
            frame = cv2.imread(str(image))
            if frame is None:
                raise AppError("Cannot decode benchmark image.")
            for index in range(warmup + iterations):
                synchronize(config.device)
                start = perf_counter()
                rows, stages = detector.predict(frame, config.confidence)
                decode_detections(
                    rows, detector.names, frame.shape[1], frame.shape[0], config.confidence
                )
                duration = perf_counter() - start
                if index >= warmup:
                    elapsed.append(duration * 1000)
                    stage_values.append(stages)
        return {
            "environment": environment(),
            "bundle": str(config.bundle),
            "device": config.device,
            "input_size": config.input_size,
            "batch_size": 1,
            "warmup_per_image": warmup,
            "iterations_per_image": iterations,
            "samples": len(elapsed),
            "median_ms": float(np.median(elapsed)),
            "p95_ms": float(np.percentile(elapsed, 95)),
            "backend_stages_median_ms": {
                k: float(np.median([s[k] for s in stage_values]) * 1000) for k in stage_values[0]
            },
            "boundaries": "predict + output validation; excludes image decode, camera and display",
            "image_sha256": [checksum(p) for p in images],
        }
    finally:
        detector.close()


def train(bundle: Path, data: Path, train_config: Path, groups: Path, output: Path):
    from .data import validate_dataset

    audit = validate_dataset(data, groups)
    artifact, _, metadata = read_bundle(bundle)
    if artifact.suffix != ".pt":
        raise AppError("Fine-tuning requires the original .pt checkpoint.")
    raw = yaml.safe_load(train_config.read_text())
    allowed = {"epochs", "batch", "imgsz", "device", "seed", "patience", "workers"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise AppError("Training config contains unsupported settings.")
    for key in ("epochs", "batch", "imgsz", "seed", "patience", "workers"):
        if key in raw and (
            type(raw[key]) is not int
            or raw[key] < (0 if key in ("seed", "patience", "workers") else 1)
        ):
            raise AppError(f"Training setting {key} has an invalid integer value.")
    Config(bundle=bundle, input_size=raw.get("imgsz", 640), device=raw.get("device", "cpu"))
    if output.exists():
        raise AppError("Choose a new training run directory to preserve previous runs.")
    write_json(
        output / "run_manifest.json",
        {
            "data_audit": audit,
            "data": str(data.resolve()),
            "config": raw,
            "source_bundle": metadata,
            "environment": environment(),
            "status": "started",
        },
    )
    model = yolo_class()(str(artifact), task="detect")
    dataset = resolved_dataset(data, output)
    # workers=0 is portable on macOS; external scripts retain a guarded entry point.
    model.train(
        data=str(dataset.resolve()),
        project=str(output.resolve()),
        name="training",
        exist_ok=False,
        **raw,
    )
    best = output / "training" / "weights" / "best.pt"
    manifest = package_model(
        best,
        output / "model",
        {
            "source": str(artifact),
            "dataset_config_sha256": checksum(data),
            "groups_sha256": checksum(groups),
            "training_config": raw,
            "dataset_revision": audit["config_sha256"],
        },
        input_size=raw.get("imgsz", 640),
    )
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    run_manifest.update({"status": "completed", "bundle": str(manifest.resolve())})
    write_json(output / "run_manifest.json", run_manifest)
    return {"bundle": str(manifest), "status": "training completed; evaluation pending"}


def evaluate(bundle: Path, data: Path, groups: Path, split: str, device: str, output: Path):
    from .data import validate_dataset

    audit = validate_dataset(data, groups)
    artifact, names, metadata = read_bundle(bundle)
    if tuple(audit["classes"]) != names:
        raise AppError("Dataset class order must match the evaluated model bundle.")
    if output.exists():
        raise AppError("Choose a new evaluation output directory.")
    model = yolo_class()(str(artifact), task="detect")
    dataset = resolved_dataset(data, output)
    metrics = model.val(
        data=str(dataset.resolve()),
        split=split,
        device=device,
        imgsz=metadata["input_size"],
        project=str(output.resolve()),
        name="evaluation",
        plots=True,
        save_json=False,
        workers=0,
    )
    # AP exists only for classes included in the metric computation; map IDs explicitly.
    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        per_class[names[int(class_id)]] = {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "ap50": float(metrics.box.ap50[index]),
            "ap50_95": float(metrics.box.ap[index]),
        }
    return {
        "split": split,
        "environment": environment(),
        "dataset_audit": audit,
        "bundle_sha256": checksum(bundle),
        "metrics": metrics.results_dict,
        "per_class": per_class,
        "input_size": metadata["input_size"],
        "test_exposure": split == "test",
        "protocol": "Ultralytics detection validation; AP sweeps confidence at IoU .50:.95",
    }


def export_model(bundle: Path, output: Path) -> Path:
    artifact, _, metadata = read_bundle(bundle)
    if artifact.suffix != ".pt":
        raise AppError("Export requires the retained original .pt checkpoint.")
    if output.exists():
        raise AppError("Choose a new export directory.")
    # Copy before export so exporter side effects cannot replace the source artifact.
    output.mkdir(parents=True)
    checkpoint = output / artifact.name
    shutil.copy2(artifact, checkpoint)
    model = yolo_class()(str(checkpoint), task="detect")
    exported = Path(
        model.export(
            format="onnx", imgsz=metadata["input_size"], dynamic=False, simplify=False, device="cpu"
        )
    )
    return package_model(
        exported,
        output,
        {
            "source_checkpoint_sha256": checksum(artifact),
            "format": "onnx",
            "dynamic": False,
            "simplify": False,
            "precision": "FP32",
            "verification": "pending parity and dataset evaluation",
        },
        metadata["input_size"],
    )


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def compare_detections(left, right, score_tolerance: float, box_tolerance: float) -> list[str]:
    """Greedy same-class maximum-IoU matching; require every detection to match once."""
    failures, remaining = [], list(right)
    for detection in sorted(left, key=lambda d: d.confidence, reverse=True):
        candidates = [d for d in remaining if d.class_id == detection.class_id]
        if not candidates:
            failures.append("missing class detection")
            continue
        match = max(candidates, key=lambda d: box_iou(d.xyxy, detection.xyxy))
        remaining.remove(match)
        if abs(match.confidence - detection.confidence) > score_tolerance:
            failures.append("score tolerance exceeded")
        if max(abs(a - b) for a, b in zip(match.xyxy, detection.xyxy, strict=True)) > box_tolerance:
            failures.append("coordinate tolerance exceeded")
    if remaining:
        failures.append("extra detections")
    return failures


def verify_export(
    left: Config, right: Config, images: list[Path], score_tol: float, box_tol: float
):
    if not images or not 0 <= score_tol <= 1 or not np.isfinite(box_tol) or box_tol < 0:
        raise AppError("Provide images and finite nonnegative comparison tolerances.")
    a, b = load_detector(left), None
    try:
        b = load_detector(right)
        if a.names != b.names:
            raise AppError("Export class map differs from checkpoint.")
        failures = {}
        for image in images:
            frame = cv2.imread(str(image))
            if frame is None:
                raise AppError("Cannot decode export verification image.")
            height, width = frame.shape[:2]
            detections = []
            for detector, config in ((a, left), (b, right)):
                rows, _ = detector.predict(frame, config.confidence)
                detections.append(
                    decode_detections(rows, detector.names, width, height, config.confidence)
                )
            errors = compare_detections(*detections, score_tol, box_tol)
            if errors:
                failures[image.name] = errors
        return {
            "passed": not failures,
            "images": len(images),
            "failures": failures,
            "score_tolerance": score_tol,
            "box_tolerance_pixels": box_tol,
            "confidence": left.confidence,
            "input_size": left.input_size,
            "matching": "same-class greedy maximum-IoU, descending checkpoint confidence",
            "image_sha256": [checksum(p) for p in images],
            "environment": environment(),
            "limitations": "Image parity alone does not establish held-out dataset accuracy.",
        }
    finally:
        a.close()
        if b is not None:
            b.close()
