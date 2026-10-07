"""Audit local YOLO labels and group manifests before training or evaluation."""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .contracts import AppError
from .detector import checksum


def validate_dataset(config_path: Path, groups_path: Path) -> dict:
    """Require images/<split> and labels/<split>, with a group per image.

    Empty .txt files explicitly represent reviewed negative images. Missing labels
    are rejected because they could otherwise hide incomplete annotation.
    """
    try:
        config = yaml.safe_load(config_path.read_text())
        groups = json.loads(groups_path.read_text())
        names = config["names"]
        if isinstance(names, dict):
            if set(names) != set(range(len(names))):
                raise ValueError("Class IDs must start at zero and be contiguous.")
            names = [names[i] for i in range(len(names))]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Invalid class names.")
        if not isinstance(groups, dict):
            raise ValueError("Groups must be a mapping of relative image paths to source IDs.")
        root = (config_path.resolve().parent / config["path"]).resolve()
        hashes, group_splits, counts, class_counts = {}, {}, {}, {}
        known_images = set()
        for split in ("train", "val", "test"):
            relative = Path(config[split])
            if relative.parts != ("images", split):
                raise ValueError(f"{split} must use images/{split} in this validator.")
            directory = root / relative
            if not directory.is_dir():
                raise ValueError(f"Missing image directory for {split}.")
            images = sorted(
                p
                for p in directory.rglob("*")
                if p.suffix.lower()
                in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                )
            )
            if not images:
                raise ValueError(f"No images in {split}.")
            seen_labels = set()
            split_counts = [0] * len(names)
            for image in images:
                key = image.relative_to(root).as_posix()
                known_images.add(key)
                group = groups.get(key)
                if not isinstance(group, str) or not group.strip():
                    raise ValueError(f"Missing source group: {key}.")
                previous = group_splits.setdefault(group, split)
                if previous != split:
                    raise ValueError("Source/session group overlaps dataset splits.")
                frame = cv2.imread(str(image))
                if frame is None:
                    raise ValueError(f"Unreadable image: {key}.")
                # Decoded-pixel hashes also catch identical content encoded differently.
                import hashlib

                digest = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError(f"Duplicate image content: {key} and {hashes[digest]}.")
                hashes[digest] = key
                label = root / "labels" / split / image.relative_to(directory).with_suffix(".txt")
                if label in seen_labels:
                    raise ValueError("Multiple images map to the same label filename.")
                seen_labels.add(label)
                rows = set()
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f"Label must contain five fields: {label.name}.")
                    cls, cx, cy, width, height = map(float, parts)
                    if not np.isfinite([cls, cx, cy, width, height]).all():
                        raise ValueError("Non-finite label values.")
                    if cls != int(cls) or not 0 <= cls < len(names):
                        raise ValueError("Label class ID outside the class map.")
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("Invalid normalized box geometry.")
                    if (
                        cx - width / 2 < -1e-6
                        or cx + width / 2 > 1 + 1e-6
                        or cy - height / 2 < -1e-6
                        or cy + height / 2 > 1 + 1e-6
                    ):
                        raise ValueError("Label box extends outside image boundaries.")
                    row = (cls, cx, cy, width, height)
                    if row in rows:
                        raise ValueError("Duplicate annotation box.")
                    rows.add(row)
                    split_counts[int(cls)] += 1
            if set((root / "labels" / split).rglob("*.txt")) != seen_labels:
                raise ValueError(f"Unpaired label files in {split}.")
            if any(count == 0 for count in split_counts):
                raise ValueError(f"Every class requires independent instances in {split}.")
            counts[split], class_counts[split] = len(images), split_counts
        if set(groups) != known_images:
            raise ValueError("Group manifest contains missing or extra image entries.")
        return {
            "images": counts,
            "instances_by_class": class_counts,
            "classes": names,
            "config_sha256": checksum(config_path),
            "groups_sha256": checksum(groups_path),
            "image_pixel_hashes": hashes,
            "limitations": "Decoded duplicates checked; near-duplicates require visual review.",
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as exc:
        raise AppError(f"Dataset audit failed: {exc}") from exc
