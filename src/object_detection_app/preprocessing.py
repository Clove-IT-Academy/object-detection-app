"""Frame validation and geometry helpers; the current adapter owns model preprocessing."""

import cv2
import numpy as np

from .contracts import AppError, Transform


def validate_frame(frame: np.ndarray) -> np.ndarray:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or not 1 <= frame.shape[0] <= 8192
        or not 1 <= frame.shape[1] <= 8192
    ):
        raise AppError("Expected a nonempty uint8 BGR image with three channels.")
    return frame


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, Transform]:
    """Geometry helper for future raw adapters; do not call before Ultralytics predict.

    Separate x/y scales preserve the exact rounded resize rather than a nominal ratio.
    """
    validate_frame(frame)
    if type(size) is not int or not 1 <= size <= 4096:
        raise AppError("Letterbox size must be an integer between 1 and 4096.")
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
    pad_x, pad_y = (size - resized_w) // 2, (size - resized_h) // 2
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, Transform(width, height, resized_w / width, resized_h / height, pad_x, pad_y)
