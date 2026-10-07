import cv2
import numpy as np
import pytest

from object_detection_app.renderer import Renderer


@pytest.mark.parametrize("visibility", [0, -1, "error"])
def test_native_close_never_calls_imshow_or_recreates_window(monkeypatch, visibility):
    renderer = Renderer("fake")
    renderer.opened = True

    def property_value(*args):
        if visibility == "error":
            raise cv2.error("Window no longer exists")
        return visibility

    def unexpected(*args):
        raise AssertionError("A closed window must not be displayed again")

    monkeypatch.setattr(cv2, "getWindowProperty", property_value)
    monkeypatch.setattr(cv2, "imshow", unexpected)
    assert renderer.show(np.zeros((10, 10, 3), dtype=np.uint8)) == ord("q")
    assert renderer.poll() == ord("q")
