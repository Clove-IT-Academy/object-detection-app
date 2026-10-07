import json

import cv2
import numpy as np
import pytest

from object_detection_app.contracts import AppError
from object_detection_app.data import validate_dataset


def dataset(tmp_path):
    groups = {}
    for index, split in enumerate(("train", "val", "test")):
        images, labels = tmp_path / "images" / split, tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / "a.png"), np.full((20, 20, 3), index * 50, dtype=np.uint8))
        (labels / "a.txt").write_text("0 .5 .5 .5 .5\n")
        groups[f"images/{split}/a.png"] = split
    config = tmp_path / "dataset.yaml"
    config.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [cup]\n"
    )
    manifest = tmp_path / "groups.json"
    manifest.write_text(json.dumps(groups))
    return config, manifest


def test_valid_dataset(tmp_path):
    config, groups = dataset(tmp_path)
    assert validate_dataset(config, groups)["images"] == {"train": 1, "val": 1, "test": 1}


@pytest.mark.parametrize("error", ["overlap", "duplicate", "bad_label", "missing_label"])
def test_dataset_rejects_leakage_and_bad_labels(tmp_path, error):
    config, groups = dataset(tmp_path)
    if error == "overlap":
        values = json.loads(groups.read_text())
        values["images/val/a.png"] = "train"
        groups.write_text(json.dumps(values))
    elif error == "duplicate":
        (tmp_path / "images/val/a.png").write_bytes((tmp_path / "images/train/a.png").read_bytes())
    elif error == "bad_label":
        (tmp_path / "labels/train/a.txt").write_text("1 .5 .5 .5 .5\n")
    else:
        (tmp_path / "labels/train/a.txt").unlink()
    with pytest.raises(AppError):
        validate_dataset(config, groups)
