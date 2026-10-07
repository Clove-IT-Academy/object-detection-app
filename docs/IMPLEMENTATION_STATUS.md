# Implementation status — 2026-10-07

Version 0.1.0 is a development implementation with a runnable pretrained baseline, not a
completed custom-trained project release. Start from README instructions and this status.

Implemented: installable package and CLI; YAML config/overrides; validated camera adapter;
single-load detector for local checkpoint/ONNX; model checksums and matching class metadata;
backend-owned preprocessing/NMS/original-pixel coordinates; safe clipping/filtering; annotated
preview with quit/window-close and threshold controls; bounded timing storage and cleanup;
explicit download/package, data validation, train, evaluation, export, parity and benchmark
workflows. Comments explain non-obvious geometry, ownership, privacy, and cleanup decisions.

Verified: 40 automated tests including three actual-model integration tests, real pretrained
inference, CPU benchmarking, ONNX export and image parity, a one-epoch synthetic training and
validation workflow, lint/format checks and dependency consistency. The synthetic dataset is
only an engineering fixture and its zero metrics are not representative of project accuracy.

The camera smoke run was attempted; macOS denied camera access. Hardware preview, native
window closure, camera disconnect, sustained full-loop performance, and other platforms are
unverified. No camera footage was recorded or saved during the attempted check.

| Plan step | Status |
| --- | --- |
| 1: Scope | Baseline brief exists; custom classes/scenario and target criteria need confirmation |
| 2: Environment | Complete for development venv and source scaffold; release-platform installation remains separate |
| 3: Baseline | CPU benchmark complete; user quality/latency acceptance targets remain unresolved |
| 4–5: Data | Audit tooling and example layout ready; real collection/annotation/splits not started |
| 6: Training | Workflow smoke-tested; real fine-tuning and validation selection pending |
| 7: Runtime modules | Complete with focused synthetic/fake-component tests |
| 8: Live integration | Pretrained still-image integration complete; webcam verification and custom model pending |
| 9: Responsiveness | Checkpoint/ONNX model benchmarks available; full loop/stability targets pending |
| 10: Export | Pretrained export/image parity complete; representative metrics and live integration pending |
| 11: Final evaluation | Requires frozen real candidate and independent test dataset |
| 12–13: Release/ownership | Documentation templates provided; release gates, terms and maintainer pending |

Next work: grant camera permission and complete the manual smoke checklist; agree intended
classes/scenario and measurable criteria; then prepare representative permitted data and follow
the remaining data/model gates. Do not update the remaining plan checkboxes based on a pretrained
demo or synthetic training run alone.
