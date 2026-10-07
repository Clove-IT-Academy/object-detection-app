"""Validate decoded detections; never repeat the backend's NMS or coordinate restoration."""

import numpy as np

from .contracts import AppError, Detection


def decode_detections(
    rows: np.ndarray,
    names: tuple[str, ...],
    width: int,
    height: int,
    confidence: float,
) -> list[Detection]:
    """Accept original-frame [x1,y1,x2,y2,score,class_id] rows; fail on corrupt output.

    Valid boxes are clipped; zero-area boxes after clipping are intentionally discarded.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise AppError("Detector output must have shape (N, 6).")
    if not np.issubdtype(rows.dtype, np.number) or not np.isfinite(rows).all():
        raise AppError("Detector returned non-finite or non-numeric values.")
    detections = []
    for x1, y1, x2, y2, score, class_id in rows:
        if not 0 <= score <= 1 or class_id != int(class_id) or not 0 <= class_id < len(names):
            raise AppError("Detector returned an invalid score or class ID.")
        if x2 < x1 or y2 < y1:
            raise AppError("Detector returned reversed box coordinates.")
        if score < confidence:
            continue
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        index = int(class_id)
        detections.append(
            Detection(
                index,
                names[index],
                float(score),
                tuple(
                    map(
                        float,
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    )
                ),
            )
        )
    return detections
