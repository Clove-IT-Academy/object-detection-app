"""OpenCV preview and controls; render only on a copy of the captured frame."""

import cv2

from .contracts import Detection


class Renderer:
    def __init__(self, title: str):
        self.title = title
        self.opened = False

    def open(self) -> None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        self.opened = True

    def draw(self, frame, detections: list[Detection], confidence: float, fps: float):
        canvas = frame.copy()
        height, width = canvas.shape[:2]
        for detection in detections:
            x1, y1, x2, y2 = map(round, detection.xyxy)
            color = (80, 220, 100)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.name} {detection.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = max(0, min(x1, width - tw - 4))
            ty = min(height - baseline - 2, max(th + 4, y1 - 5))
            cv2.rectangle(
                canvas,
                (tx, max(0, ty - th - 4)),
                (min(width - 1, tx + tw + 4), min(height - 1, ty + baseline)),
                (20, 30, 20),
                -1,
            )
            cv2.putText(
                canvas, text, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
        status = f"{len(detections)} objects | conf {confidence:.2f} | {fps:.1f} FPS"
        cv2.rectangle(canvas, (0, 0), (width - 1, 56), (25, 25, 25), -1)
        cv2.putText(
            canvas, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            "Q/Esc quit | +/- confidence",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def show(self, canvas) -> int:
        # Poll before imshow so a native close does not recreate the preview window.
        if self.is_closed():
            return ord("q")
        cv2.imshow(self.title, canvas)
        return cv2.waitKey(1) & 0xFF

    def poll(self) -> int:
        if self.is_closed():
            return ord("q")
        return cv2.waitKey(10) & 0xFF

    def is_closed(self) -> bool:
        if not self.opened:
            return True
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            # Some GUI backends raise when native close has already destroyed the window.
            # Initialization and actual imshow errors still propagate to the controller.
            return True

    def close(self) -> None:
        if self.opened:
            self.opened = False
            cv2.destroyAllWindows()
