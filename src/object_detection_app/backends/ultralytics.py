"""Use the documented high-level API for both checkpoints and supported ONNX exports."""

import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from ..contracts import AppError
from ..paths import PROJECT_ROOT


def yolo_class():
    # Disable runtime dependency installation and telemetry before importing the framework.
    # Keep framework/cache files in the project rather than global user directories.
    cache = PROJECT_ROOT / ".cache"
    (cache / "ultralytics").mkdir(parents=True, exist_ok=True)
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    if find_spec("onnxruntime") is not None:
        import onnxruntime

        onnxruntime.disable_telemetry_events()
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    return YOLO


def synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


class UltralyticsBackend:
    def __init__(self, artifact: Path, names: tuple[str, ...], config):
        self.config = config
        self.names = names
        try:
            self.model = yolo_class()(str(artifact), task="detect")
        except Exception as exc:
            raise AppError(
                "Cannot load model. Check artifact integrity and runtime compatibility."
            ) from exc
        # ONNX metadata is initialized lazily; compare names after every prediction as well.
        if artifact.suffix == ".pt":
            self.check_names(self.model.names)

    def check_names(self, raw_names) -> None:
        if isinstance(raw_names, dict):
            if set(raw_names) != set(range(len(raw_names))):
                raise AppError("Model class IDs must be contiguous from zero.")
            actual = tuple(raw_names[i] for i in range(len(raw_names)))
        else:
            actual = tuple(raw_names)
        if actual != self.names:
            raise AppError("Model class map does not match the bundle. Use matching metadata.")

    def predict(self, frame, confidence: float) -> tuple[np.ndarray, dict[str, float]]:
        try:
            result = self.model.predict(
                source=frame,
                device=self.config.device,
                imgsz=self.config.input_size,
                conf=confidence,
                iou=self.config.iou,
                max_det=self.config.max_detections,
                rect=False,
                agnostic_nms=False,
                save=False,
                save_txt=False,
                show=False,
                verbose=False,
                stream=False,
            )[0]
            synchronize(self.config.device)
            self.check_names(result.names)
            boxes = result.boxes
            if boxes is None:
                raise AppError("Detection model returned no box output contract.")
            # xyxy is already in original-frame coordinates, and the API owns NMS (if needed).
            rows = np.column_stack(
                (
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                )
            )
            speed = {
                k: float(result.speed[k]) / 1000
                for k in (
                    "preprocess",
                    "inference",
                    "postprocess",
                )
            }
            return rows, speed
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                "Model inference failed. Check device, input size, and runtime."
            ) from exc

    def close(self) -> None:
        self.model = None
