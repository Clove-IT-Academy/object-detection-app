"""Convenience launcher: `venv/bin/python main.py` starts the webcam preview."""

import sys

from object_detection_app.__main__ import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("run")
    raise SystemExit(main())
