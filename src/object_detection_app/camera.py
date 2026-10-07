"""Own the webcam resource and never return an unchecked frame."""

import cv2

from .contracts import AppError
from .preprocessing import validate_frame


class Camera:
    def __init__(self, index: int):
        self.index = index
        self.capture = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            self.close()
            raise AppError("Cannot open camera. Check permission, camera index, and other apps.")

    def read(self):
        if self.capture is None:
            raise AppError("Camera has not been opened.")
        ok, frame = self.capture.read()
        if not ok:
            return None
        return validate_frame(frame)

    def close(self) -> None:
        capture, self.capture = self.capture, None
        if capture is not None:
            capture.release()
