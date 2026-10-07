import socket
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np
import pytest

from object_detection_app.config import load_config
from object_detection_app.detector import load_detector
from object_detection_app.postprocessing import decode_detections
from object_detection_app.renderer import Renderer
from object_detection_app.workflows import compare_detections

pytestmark = pytest.mark.model
ROOT = Path(__file__).resolve().parents[1]


def deny_network(*args, **kwargs):
    raise AssertionError("Inference must not open network connections")


@pytest.mark.parametrize("bundle", ["yolo26n", "yolo26n-onnx"])
def test_real_backend_infers_offline_and_does_not_mutate_frame(bundle, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    config = replace(
        load_config(ROOT / "configs/app.yaml"), bundle=ROOT / "models" / bundle / "bundle.json"
    )
    detector = load_detector(config)
    try:
        frame = np.zeros((361, 701, 3), dtype=np.uint8)
        before = frame.copy()
        rows, stages = detector.predict(frame, config.confidence)
        detections = decode_detections(rows, detector.names, 701, 361, config.confidence)
        Renderer("offline").draw(frame, detections, config.confidence, 0)
        assert np.array_equal(frame, before)
        assert rows.shape[1] == 6
        assert all(value >= 0 for value in stages.values())
    finally:
        detector.close()


def test_public_sample_export_parity(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    sample = Path(find_spec("ultralytics").origin).parent / "assets/bus.jpg"
    frame = cv2.imread(str(sample))
    assert frame is not None
    config = load_config(ROOT / "configs/app.yaml")
    a, b = load_detector(config), None
    try:
        b = load_detector(replace(config, bundle=ROOT / "models/yolo26n-onnx/bundle.json"))
        results = []
        for detector in (a, b):
            rows, _ = detector.predict(frame, config.confidence)
            results.append(
                decode_detections(
                    rows, detector.names, frame.shape[1], frame.shape[0], config.confidence
                )
            )
        assert any(d.name == "bus" for d in results[0])
        assert compare_detections(*results, 0.02, 2) == []
    finally:
        a.close()
        if b is not None:
            b.close()
