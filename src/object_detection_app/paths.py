"""Use the checkout's assets in editable installs, or the caller's project for wheels."""

from pathlib import Path

_checkout = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _checkout if (_checkout / "configs/app.yaml").is_file() else Path.cwd()
