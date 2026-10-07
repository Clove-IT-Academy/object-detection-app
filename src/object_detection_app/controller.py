"""A bounded synchronous pipeline with injectable resources for camera-free tests."""

import logging
from time import perf_counter

import cv2

from .camera import Camera
from .contracts import AppError
from .detector import load_detector
from .postprocessing import decode_detections
from .renderer import Renderer
from .timing import Timings

logger = logging.getLogger(__name__)


def run(config, *, camera=None, renderer=None, detector=None, max_frames=None) -> dict:
    camera = camera if camera is not None else Camera(config.camera)
    renderer = renderer if renderer is not None else Renderer(config.window_title)
    timings = Timings()
    confidence, failures, processed = config.confidence, 0, 0
    try:
        detector = detector if detector is not None else load_detector(config)
        camera.open()
        renderer.open()
        while max_frames is None or processed < max_frames:
            start = perf_counter()
            frame = camera.read()
            captured = perf_counter()
            if frame is None:
                failures += 1
                if failures > config.read_retries:
                    raise AppError("Camera stopped returning frames. Reconnect it and restart.")
                # Continue polling during bounded retries, so quit remains responsive.
                if renderer.poll() in (ord("q"), ord("Q"), 27):
                    break
                continue
            failures = 0
            rows, stages = detector.predict(frame, confidence)
            predicted = perf_counter()
            detections = decode_detections(
                rows,
                detector.names,
                frame.shape[1],
                frame.shape[0],
                confidence,
            )
            decoded = perf_counter()
            canvas = renderer.draw(frame, detections, confidence, timings.fps)
            rendered = perf_counter()
            key = renderer.show(canvas)
            ended = perf_counter()
            timings.add(
                capture=captured - start,
                backend=predicted - captured,
                preprocess=stages["preprocess"],
                inference=stages["inference"],
                backend_postprocess=stages["postprocess"],
                validation=decoded - predicted,
                render=rendered - decoded,
                display=ended - rendered,
                total=ended - start,
            )
            processed += 1
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("+"), ord("=")):
                confidence = min(1.0, round(confidence + 0.05, 2))
            elif key in (ord("-"), ord("_")):
                confidence = max(0.0, round(confidence - 0.05, 2))
    except cv2.error as exc:
        raise AppError(
            "Camera/display operation failed. Check desktop access and permissions."
        ) from exc
    finally:
        # Each cleanup is independent: a display failure must not leak the camera/model.
        for resource in (camera, renderer, detector):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    logger.warning("Resource cleanup failed for %s.", type(resource).__name__)
    return {"processed_frames": processed, "rolling_timings": timings.summary()}
