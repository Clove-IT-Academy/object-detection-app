import json
from pathlib import Path

import pytest

from object_detection_app.config import Config, load_config
from object_detection_app.contracts import AppError, Detection
from object_detection_app.detector import checksum, read_bundle
from object_detection_app.timing import Timings
from object_detection_app.workflows import compare_detections


@pytest.mark.parametrize(
    "setting",
    [
        {"camera": True},
        {"confidence": float("nan")},
        {"input_size": 333},
        {"read_retries": -1},
        {"iou": 2},
    ],
)
def test_invalid_config(setting):
    with pytest.raises(AppError):
        Config(bundle=Path("unused"), **setting)


def test_config_paths_belong_to_yaml_file(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text("bundle: models/bundle.json\n")
    assert load_config(path).bundle == tmp_path / "models/bundle.json"
    path.write_text("bundle: model.json\nunknown: true\n")
    with pytest.raises(AppError):
        load_config(path)


def test_model_integrity_and_metadata_checked_before_loading(tmp_path):
    artifact = tmp_path / "fake.pt"
    artifact.write_bytes(b"not actually deserialized in this test")
    path = tmp_path / "bundle.json"
    metadata = {
        "schema_version": 1,
        "artifact": "fake.pt",
        "sha256": checksum(artifact),
        "classes": ["cup"],
        "input_size": 640,
        "preprocessing_owner": "ultralytics",
        "postprocessing_owner": "ultralytics",
        "coordinates": "original_xyxy",
    }
    path.write_text(json.dumps(metadata))
    assert read_bundle(path)[1] == ("cup",)
    artifact.write_bytes(b"modified")
    with pytest.raises(AppError):
        read_bundle(path)


def test_export_matching_is_class_specific_and_requires_no_extras():
    a = Detection(0, "cup", 0.8, (0, 0, 10, 10))
    b = Detection(0, "cup", 0.81, (0.5, 0, 10, 10))
    assert compare_detections([a], [b], 0.02, 2) == []
    assert compare_detections([a], [], 0.02, 2)
    assert compare_detections([], [b], 0.02, 2)
    assert compare_detections([a], [b], 0.001, 0.1)


def test_timing_memory_is_bounded():
    timings = Timings(capacity=2)
    for _ in range(20):
        timings.add(total=0.1)
    assert timings.summary()["samples"] == 2
    assert timings.fps == pytest.approx(10)
