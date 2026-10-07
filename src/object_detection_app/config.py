"""Strict startup configuration with predictable path resolution."""

import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from .contracts import AppError


@dataclass(frozen=True)
class Config:
    bundle: Path
    camera: int = 0
    device: str = "cpu"
    input_size: int = 640
    confidence: float = 0.35
    iou: float = 0.7
    max_detections: int = 100
    read_retries: int = 3
    window_title: str = "Object Detection App"

    def __post_init__(self) -> None:
        for name, low, high in (
            ("camera", 0, 64),
            ("input_size", 32, 4096),
            ("max_detections", 1, 1000),
            ("read_retries", 0, 100),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise AppError(f"{name} must be an integer between {low} and {high}.")
        if self.input_size % 32:
            raise AppError("input_size must be a multiple of 32.")
        for name in ("confidence", "iou"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise AppError(f"{name} must be finite and between 0 and 1.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise AppError("device must be a nonempty runtime device string.")
        if not isinstance(self.window_title, str) or not 1 <= len(self.window_title) <= 80:
            raise AppError("window_title must contain 1 to 80 characters.")
        if not isinstance(self.bundle, Path):
            raise AppError("bundle must be a local path.")


def load_config(path: Path, overrides: dict | None = None) -> Config:
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise AppError("Cannot read configuration; check the file and YAML syntax.") from exc
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(Config)}:
        raise AppError("Configuration must be a mapping containing only supported settings.")
    raw = dict(raw)
    bundle = raw.get("bundle")
    if not isinstance(bundle, str) or not bundle or "://" in bundle:
        raise AppError("Configuration requires a local model bundle path.")
    raw["bundle"] = (path.resolve().parent / bundle).resolve()
    # CLI paths are relative to the shell, whereas YAML paths belong to the YAML file.
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return Config(**raw)
