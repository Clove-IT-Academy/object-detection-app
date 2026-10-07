# Decision log — 2026-10-07

| Decision | Implementation choice | Status / reason |
| --- | --- | --- |
| Baseline classes | Exact official YOLO26n COCO class map | Provisional; custom class choice/data pending |
| Runtime | Python/OpenCV desktop, synchronous loop | Implemented; simple bounded memory pipeline |
| Interpreter | Existing Python 3.14.7 venv | Dependencies installed and checked successfully |
| Device | CPU | Apple M4 inference measured; other devices unverified |
| Dependencies | Latest compatible PyPI versions resolved today, then pinned | Full installed versions in requirements.lock.txt |
| Model acquisition | Explicit fixed official GitHub release download | No implicit downloads on startup |
| Geometry/NMS | Ultralytics owns preprocessing, decode, coordinate restoration and NMS if needed | App validates original-frame decoded output |
| Confidence / IoU | 0.35 / 0.7 starting settings; class-aware backend suppression | Validation selection pending |
| Export | FP32 static 640 ONNX, CPU runtime | Smoke parity passed; full quality gate pending |
| Data paths | Resolve YAML root relative to config before passing to framework | Prevents framework dataset-cache path ambiguity |
| Failures | Three read retries then exit; controls remain polled | Fake lifecycle tests pass |
| Controls | Q/Esc/native close; +/- threshold | Automated controller test; physical GUI checks pending |
| Privacy | Offline framework, no autoinstall, sync disabled, transient frames | Python-network-blocked inference tests included |
| Camera access | macOS denied the attempted smoke run | Requires user OS permission; not changed automatically |
| Release | Development build 0.1.0 | Dataset, target metrics, platform checks, owner/terms pending |

Plan adaptation: independent runtime scaffolding and pretrained feasibility work proceeded while
data-dependent gates remain unresolved, as permitted by the plan. No custom training gate is
marked complete. Synthetic workflow training is engineering verification, not quality evidence.
