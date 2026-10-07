import numpy as np
import pytest

from object_detection_app.contracts import AppError
from object_detection_app.postprocessing import decode_detections
from object_detection_app.preprocessing import letterbox, validate_frame


@pytest.mark.parametrize("shape", [(97, 301, 3), (301, 97, 3), (100, 100, 3)])
def test_letterbox_restores_exact_rounded_geometry(shape):
    frame = np.zeros(shape, dtype=np.uint8)
    canvas, transform = letterbox(frame, 320)
    box = (5, 7, shape[1] - 3, shape[0] - 4)
    x1, y1, x2, y2 = box
    model_box = (
        x1 * transform.scale_x + transform.pad_x,
        y1 * transform.scale_y + transform.pad_y,
        x2 * transform.scale_x + transform.pad_x,
        y2 * transform.scale_y + transform.pad_y,
    )
    assert transform.restore(model_box) == pytest.approx(box)
    assert canvas.shape == (320, 320, 3)
    assert not frame.any()


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3)),
    ],
)
def test_bad_frames_rejected(frame):
    with pytest.raises(AppError):
        validate_frame(frame)


def test_clipping_and_inclusive_confidence_boundary():
    rows = np.array(
        [[-10, -5, 200, 90, 0.35, 0], [1, 2, 5, 6, 0.34, 0], [110, 110, 120, 120, 0.9, 0]]
    )
    result = decode_detections(rows, ("cup",), 100, 80, 0.35)
    assert len(result) == 1
    assert result[0].xyxy == (0, 0, 99, 79)
    assert result[0].name == "cup"


@pytest.mark.parametrize(
    "row",
    [
        [0, 0, 5, 5, float("nan"), 0],
        [0, 0, 5, 5, 1.1, 0],
        [0, 0, 5, 5, 0.5, 3],
        [0, 0, 5, 5, 0.5, 0.1],
        [5, 0, 0, 5, 0.5, 0],
    ],
)
def test_corrupt_outputs_fail_explicitly(row):
    with pytest.raises(AppError):
        decode_detections(np.array([row]), ("cup",), 100, 80, 0.35)


def test_empty_output_is_normal():
    assert decode_detections(np.empty((0, 6)), ("cup",), 100, 80, 0.35) == []
