from pathlib import Path

import numpy as np
import pytest

from object_detection_app.config import Config
from object_detection_app.contracts import AppError
from object_detection_app.controller import run


class FakeCamera:
    def __init__(self, frame=None, failure=False):
        self.frame, self.failure, self.closed = frame, failure, False

    def open(self):
        if self.failure:
            raise AppError("cannot open")

    def read(self):
        return self.frame

    def close(self):
        self.closed = True


class FakeRenderer:
    def __init__(self, keys=None, cleanup_failure=False):
        self.keys = iter(keys or [ord("q")])
        self.closed, self.cleanup_failure = False, cleanup_failure

    def open(self):
        pass

    def draw(self, frame, detections, confidence, fps):
        return frame

    def show(self, canvas):
        return next(self.keys)

    def poll(self):
        return 255

    def close(self):
        self.closed = True
        if self.cleanup_failure:
            raise RuntimeError("cleanup error")


class FakeDetector:
    names = ("cup",)

    def __init__(self, failure=False):
        self.closed, self.failure, self.thresholds = False, failure, []

    def predict(self, frame, confidence):
        self.thresholds.append(confidence)
        if self.failure:
            raise AppError("inference error")
        return np.empty((0, 6)), {"preprocess": 0, "inference": 0, "postprocess": 0}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("case", ["normal", "camera", "reads", "inference", "cleanup"])
def test_every_exit_releases_all_resources(case):
    camera = FakeCamera(np.zeros((100, 100, 3), dtype=np.uint8), case == "camera")
    if case == "reads":
        camera.frame = None
    renderer = FakeRenderer(cleanup_failure=case == "cleanup")
    detector = FakeDetector(case == "inference")
    config = Config(bundle=Path("unused"), read_retries=1)
    if case in ("camera", "reads", "inference"):
        with pytest.raises(AppError):
            run(config, camera=camera, renderer=renderer, detector=detector)
    else:
        assert (
            run(config, camera=camera, renderer=renderer, detector=detector)["processed_frames"]
            == 1
        )
    assert camera.closed and renderer.closed and detector.closed


def test_controls_update_threshold_without_reloading_model():
    camera = FakeCamera(np.zeros((100, 100, 3), dtype=np.uint8))
    renderer = FakeRenderer([ord("+"), ord("-"), ord("q")])
    detector = FakeDetector()
    run(Config(bundle=Path("unused")), camera=camera, renderer=renderer, detector=detector)
    assert detector.thresholds == [0.35, 0.4, 0.35]
