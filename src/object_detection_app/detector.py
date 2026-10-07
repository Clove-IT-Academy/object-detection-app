"""Bundle integrity and the application-facing detector contract."""

import hashlib
import json
from pathlib import Path
from typing import Protocol

from .contracts import AppError


class Detector(Protocol):
    names: tuple[str, ...]

    def predict(self, frame, confidence: float): ...
    def close(self) -> None: ...


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_bundle(path: Path) -> tuple[Path, tuple[str, ...], dict]:
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("schema")
        filename = raw["artifact"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("artifact must be adjacent to the manifest")
        artifact = path.resolve().parent / filename
        if artifact.suffix not in (".pt", ".onnx") or not artifact.is_file():
            raise ValueError("artifact missing or unsupported")
        expected = raw["sha256"]
        if not isinstance(expected, str) or checksum(artifact) != expected:
            raise ValueError("checksum mismatch")
        names = raw["classes"]
        if (
            not isinstance(names, list)
            or not names
            or any(
                not isinstance(name, str) or not name.strip() or len(name) > 100 for name in names
            )
            or len(names) != len(set(names))
        ):
            raise ValueError("invalid class map")
        if raw.get("preprocessing_owner") != "ultralytics":
            raise ValueError("unsupported preprocessing")
        if raw.get("postprocessing_owner") != "ultralytics":
            raise ValueError("unsupported postprocessing")
        if raw.get("coordinates") != "original_xyxy":
            raise ValueError("unsupported coordinates")
        if type(raw.get("input_size")) is not int or not 32 <= raw["input_size"] <= 4096:
            raise ValueError("invalid input size")
        return artifact, tuple(names), raw
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AppError(
            "Invalid or missing model bundle. Run setup-model, or package matching weights "
            "and metadata; check SHA-256 integrity."
        ) from exc


def load_detector(config) -> Detector:
    artifact, names, metadata = read_bundle(config.bundle)
    if artifact.suffix == ".onnx" and config.input_size != metadata["input_size"]:
        raise AppError("ONNX input_size must match the static exported bundle.")
    if artifact.suffix == ".onnx" and config.device != "cpu":
        raise AppError("This ONNX backend is verified for CPU only; select --device cpu.")
    from .backends.ultralytics import UltralyticsBackend

    return UltralyticsBackend(artifact, names, config)
