"""Small shared contracts keep camera, runtime, geometry, and UI independent."""

from dataclasses import dataclass


class AppError(Exception):
    """An operational failure suitable for a concise user-facing message."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Transform:
    original_width: int
    original_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int

    def restore(self, box: tuple[float, float, float, float]) -> tuple[float, ...]:
        x1, y1, x2, y2 = box
        return (
            (x1 - self.pad_x) / self.scale_x,
            (y1 - self.pad_y) / self.scale_y,
            (x2 - self.pad_x) / self.scale_x,
            (y2 - self.pad_y) / self.scale_y,
        )
