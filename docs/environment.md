# Development environment

Date: 2026-10-07. Existing `venv` initially contained Python 3.14.7 and pip only. Current
compatible releases were installed into that environment, not global Python. The target is
macOS 27.0.1 arm64, Apple M4 CPU. Camera index 0 was attempted but OS access was denied;
camera make/resolution and live display behavior remain unverified.

Runtime/framework versions and benchmark settings are recorded in reports and model manifests.
All installed package versions are locked in `requirements.lock.txt`; direct app/dev/export
dependencies are pinned in `pyproject.toml`. README contains recreation and execution commands.
The framework cache is project-local `.cache/`; large data/weights/runs are ignored by source control.

Compatibility evidence covers installation, CPU still-image inference, export/runtime, tests,
synthetic training and validation. CUDA/MPS, other OSes, camera devices, and other Python versions
need their own checks before support is claimed. Do not substitute competing OpenCV distributions.
