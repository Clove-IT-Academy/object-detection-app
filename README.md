# Object Detection App

Local webcam detection with bounding boxes, class names, confidence scores, and a live
preview. The application is a modular Python implementation of the architecture and
development plan. The current runnable baseline uses official pretrained YOLO26n weights
and their 80 COCO classes. Custom collection, annotation, fine-tuning, and final accuracy
evaluation remain pending until project classes and a representative dataset are supplied.

## Run the existing installation

From this project directory, on macOS/Linux:

```bash
venv/bin/python main.py
```

Equivalent command with explicit settings:

```bash
venv/bin/python -m object_detection_app run --camera 0 --confidence 0.35 --device cpu
```

On Windows, after completing the Windows setup below, use PowerShell:

```powershell
.\venv\Scripts\python.exe main.py
.\venv\Scripts\python.exe -m object_detection_app run --camera 0 --confidence 0.35 --device cpu
```

Activation is optional: these commands directly select the project's virtual environment.

Controls: **Q**, **Esc**, or native window close quits; **+**/**=** increases confidence
by 0.05; **-**/**_** decreases it. The threshold stays between 0 and 1. Adjustments last
only for the session. The header shows object count, threshold, and rolling processed FPS.
The first frame has no preceding timing sample, so its displayed FPS is zero.

macOS denied camera access during the development smoke check. If you see that error,
enable camera access in **System Settings → Privacy & Security → Camera** for the app
hosting Python (your terminal or Codex), then restart that app and run again. Close other
applications using the camera and try another `--camera` index if necessary. If a hosting
app cannot request access, run the same command from an interactive terminal.

## Recreate the environment on macOS/Linux

Tested dependency installation and still-image inference: Python **3.14.7**, macOS
**27.0.1**, Apple M4/arm64, CPU. Live camera/display behavior on this host remains pending
because camera permission was denied. Other operating systems and accelerators are unverified.

```bash
python3.14 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.lock.txt
venv/bin/python -m pip install --no-build-isolation -e .
venv/bin/python -m object_detection_app setup-model
venv/bin/python main.py
```

The first command bootstraps the virtual environment; all subsequent Python, pip, scripts,
and tests use its interpreter. The lock records the versions resolved on 2026-10-07 for
this host. Installation may need a separately tested lock on another platform. Do not
install headless/contrib OpenCV alongside `opencv-python` in this environment.

`setup-model` explicitly downloads one official GitHub release asset. Normal startup
requires existing local weights and metadata and does not download, train, export, record,
or upload frames. Model bundles are verified against their recorded SHA-256 before loading.
Checksums provide artifact identity, not independent publisher signature verification.

Current direct runtime versions are pinned in `pyproject.toml`: Ultralytics 8.4.174,
OpenCV 5.0.0.93, NumPy 2.5.3, and PyYAML 6.0.3. ONNX 1.23.2 and ONNX Runtime 1.30.0
are included in the tested environment. These were resolved from current PyPI releases;
upgrades should repeat the affected tests instead of silently changing dependencies.

## Windows setup and troubleshooting

Windows instructions are provided for local development; installation, webcam behavior,
and performance on Windows have **not been verified**. Start with a 64-bit Python 3.14
installation from [Python's official website](https://www.python.org/downloads/windows/)
and open PowerShell in `object-detection-app`. Confirm the selected interpreter:

```powershell
py -3.14 --version
```

If `py` is unavailable, use the installed Python 3.14 executable instead and confirm its
version before creating the environment. Create a fresh Windows venv; do not copy the
macOS `venv` folder, because virtual environments are platform-specific and not portable.
See the [Python venv documentation](https://docs.python.org/3.14/library/venv.html).

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install setuptools==84.0.0
.\venv\Scripts\python.exe -m pip install --no-build-isolation -e ".[dev,export]"
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m object_detection_app setup-model
.\venv\Scripts\python.exe main.py
```

The existing `requirements.lock.txt` records the macOS environment. Windows setup above
uses the project's pinned direct dependencies and lets pip resolve compatible Windows
transitive packages. If pip reports that a pinned version has no compatible wheel, check
the Python version/architecture and package support; do not assume this platform is verified
or silently change versions. After successful installation and verification, save a separate
Windows lock rather than replacing the macOS one:

```powershell
.\venv\Scripts\python.exe -m pip freeze --exclude object-detection-app | Set-Content -Encoding ascii requirements.windows.lock.txt
```

Future installations on the same verified Windows/Python target can install that lock,
then install the application with `--no-deps --no-build-isolation -e .`.

All later commands in this README also work with the Windows interpreter: replace
`venv/bin/python` with `.\venv\Scripts\python.exe` and keep the remaining arguments.
For example, `scripts/manage.py` can be written as `scripts\manage.py`. Quote paths that
contain spaces. In YAML, prefer forward slashes (such as `../models/yolo26n/bundle.json`)
or single-quoted Windows paths so backslashes are not interpreted as escape sequences.

Command Prompt users can use `venv\Scripts\python.exe` without the leading `.\`; the
PowerShell pipeline for saving the lock can be replaced with:

```bat
venv\Scripts\python.exe -m pip freeze --exclude object-detection-app > requirements.windows.lock.txt
```

No PowerShell activation or execution-policy change is needed when calling `python.exe`
directly. Optional activation is `.\venv\Scripts\Activate.ps1` in PowerShell or
`venv\Scripts\activate.bat` in Command Prompt; if PowerShell blocks activation, continue
using the explicit interpreter commands above.

For camera access on Windows 11, open **Settings → Privacy & security → Camera** and
enable **Camera access**, **Let apps access your camera**, and **Let desktop apps access
your camera**. On Windows 10, open **Settings → Privacy → Camera** and enable the
corresponding device/app/desktop camera access settings. The Python/OpenCV app is a desktop
application. If controls are managed by your organization, contact its administrator.
See [Microsoft's camera permission guidance](https://support.microsoft.com/en-us/windows/privacy/manage-app-permissions-for-a-camera-in-windows).

If the camera still fails, check it in the Windows Camera app, then close Camera, Teams,
Zoom, browsers, and other applications using it. Check a physical privacy shutter/USB
connection, try `--camera 1`, and restart the terminal after permission changes. Run the
native Windows interpreter from a desktop session for this guide, rather than assuming
that WSL, a remote session, or a headless environment has webcam/GUI access. Use `--device cpu`
initially; CUDA requires separately checked hardware, drivers, and a compatible PyTorch build.

Verify the Windows setup with camera-free checks first:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m ruff check src tests scripts main.py
.\venv\Scripts\python.exe -m object_detection_app run --max-frames 100
```

The last command opens the actual camera. Manually check Q/Esc/native close, +/- controls,
repeated starts, camera failures, and sustained operation before recording Windows support.
Tests using real model/export artifacts require the additional ONNX bundle described below.

## Configuration and supported classes

Edit `configs/app.yaml`. Bundle paths in YAML resolve relative to the YAML file; CLI paths
resolve relative to the working directory. Invalid settings fail at startup. CPU is the
tested default. The selected input size is 640, maximum detections 100, confidence 0.35,
and applicable IoU/NMS threshold 0.7. These are provisional operating settings, not
validation-selected optimal values. The static ONNX bundle requires its exported input size.

```bash
venv/bin/python -m object_detection_app inspect-model
venv/bin/python -m object_detection_app --help
venv/bin/python scripts/manage.py --help
```

The bundle lists the exact ordered classes. Examples include person, bottle, cup, laptop,
keyboard, chair, and book. A `person` category is object detection; the app does not identify
people. It cannot reliably recognize objects outside its trained classes, and a confidence
score is not a guarantee of correctness.

## Dataset and custom training

Choose classes and annotation rules first. Copy `configs/dataset.example.yaml` to a new
dataset config and replace its placeholder. Dataset root paths resolve relative to that YAML:

```text
data/custom-v1/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
  groups.json
```

Every image needs a matching `.txt` label with one normalized YOLO detection row per object:
`class_id center_x center_y width height`. Use an empty label file for a reviewed negative
image. Nested paths are supported if images and labels have matching relative stems.
`groups.json` maps every image path relative to the dataset root to its source/session ID:

```json
{
  "images/train/a.jpg": "capture-session-01",
  "images/val/b.jpg": "capture-session-02",
  "images/test/c.jpg": "capture-session-03"
}
```

Do not put related frames into different splits. Record sources, consent/rights, annotation
rules, counts, dataset revision, and class/scenario coverage in the dataset card. The audit
checks label geometry, class coverage, pairing, exact decoded-pixel duplicates, and source
overlap. Near-duplicates and label correctness still need visual review. It does not collect,
annotate, or split images for you.

```bash
venv/bin/python -m object_detection_app validate-data --data configs/dataset.yaml --groups data/custom-v1/groups.json --output reports/data-audit.json
venv/bin/python -m object_detection_app train --bundle models/yolo26n/bundle.json --data configs/dataset.yaml --groups data/custom-v1/groups.json --output runs/custom-v1
venv/bin/python -m object_detection_app evaluate --bundle runs/custom-v1/model/bundle.json --data configs/dataset.yaml --groups data/custom-v1/groups.json --output reports/custom-v1-val
```

Adjust `configs/train.yaml` using validation evidence and available compute. Training creates
a run manifest, resolved dataset configuration, logs, checkpoints, and matching model bundle.
Seed/configuration recording aids reproduction; identical results across devices are not promised.
The evaluation output includes per-class precision, recall/AP and aggregate mAP; framework
plots use your explicitly supplied dataset. Keep sensitive datasets and generated plots private.

Freeze the candidate and thresholds before final test evaluation. `--split test --final-test`
is an explicit reminder that the test set will be exposed; the program cannot certify that
you have preserved its independence. If those results inform further tuning, reserve new
independent test data for the revised release.

## Export, verify, and benchmark

An ONNX baseline bundle was created on the development machine. If that bundle is present,
you can use it now:

```bash
venv/bin/python -m object_detection_app run --bundle models/yolo26n-onnx/bundle.json --device cpu
```

A fresh Windows setup downloads only the original checkpoint. Create the ONNX bundle
before selecting it or running the model integration tests:

```powershell
.\venv\Scripts\python.exe -m object_detection_app export --bundle models/yolo26n/bundle.json --output models/yolo26n-onnx
.\venv\Scripts\python.exe -m pytest -q --run-model-tests
```

Export requires a new output directory. If it already exists, inspect its bundle or choose
a new versioned directory; the integration tests expect `models/yolo26n-onnx/bundle.json`.

For a new checkpoint, use a new versioned output directory:

```bash
venv/bin/python -m object_detection_app export --bundle runs/custom-v1/model/bundle.json --output models/custom-v1-onnx
venv/bin/python -m object_detection_app verify-export --bundle runs/custom-v1/model/bundle.json --export-bundle models/custom-v1-onnx/bundle.json --images path/to/validation-image.jpg --output reports/custom-v1-parity.json
venv/bin/python -m object_detection_app benchmark --bundle models/custom-v1-onnx/bundle.json --images path/to/validation-image.jpg --warmup 3 --iterations 20 --output reports/custom-v1-benchmark.json
```

Use identical input sizes/settings and CPU for comparison. Default verification tolerances
are 0.02 in confidence and 2 original-image pixels; agree appropriate tolerances before
examining results. Same-class greedy maximum-IoU matching requires no missing/extra detections.
Use varied validation images, including overlap and aspect-ratio extremes. Final release also
requires exported-runtime evaluation on the held-out dataset and webcam integration.

The framework adapter owns resizing, color/tensor preparation, decoding, coordinate
restoration, and NMS when required by the model. The app validates, clips, and confidence-filters
decoded original-frame boxes. It does not apply NMS or letterbox twice.

`benchmark` measures prediction plus output validation, excluding image decode, capture and
display. The live loop measures capture, backend stages, validation, rendering/display, and
total duration using at most 300 samples. Backend stage timings are subdivisions of backend
time and must not be added to it again. Camera-free timings are not a webcam FPS promise.

## Tests and verification

```bash
venv/bin/python -m pytest -q
venv/bin/python -m pytest -q --run-model-tests
venv/bin/python -m ruff check src tests scripts main.py
venv/bin/python -m ruff format --check src tests scripts main.py
venv/bin/python -m pip check
venv/bin/python -m object_detection_app run --max-frames 100
```

Model integration tests require both provisioned baseline bundles. They block Python socket
connections, use a blank image and the library's public bus sample, verify frame immutability,
and compare exported outputs. Camera-free tests never require a physical webcam. Use the
bounded run as a smoke check, then manually verify Q/Esc/native close, +/- controls, missing
models, camera disconnects, varied scenes, and sustained operation. See `reports/VERIFICATION.md`.

## Privacy, limitations, and maintenance

Captured webcam images stay in memory. The app saves no camera images/video, crops, class
names, or scores to logs. It prints aggregate timings when a session ends. Framework offline
mode and automatic-installation disablement are set before import; framework sync and ONNX
Runtime telemetry events are disabled. Explicit setup downloads weights; explicit data/model
workflows write their requested artifacts. Treat model files as trusted code-bearing artifacts
and do not package arbitrary downloaded checkpoints.

Small objects, occlusion, motion blur, lighting, and domain mismatch can reduce accuracy.
No project-specific mAP, minimum per-class quality, sustained webcam rate, or additional OS
support is claimed. See `docs/IMPLEMENTATION_STATUS.md` for unresolved gates. The implementation
uses current documented public APIs; a dependency may internally use its own legacy code path,
which is distinct from a deprecated application API.

Ultralytics code/weights use AGPL-3.0 or Enterprise terms; review applicable terms before
redistribution. Dataset and other dependency terms need their own review. Weights and datasets
are ignored by source control; preserve manifests/checksums and acquisition instructions.
Report issues with app/model version, OS/device, reproduction steps, and non-sensitive errors.
Assign a maintainer before release and keep the previous verified bundle/config for rollback.

Official references used during implementation:
[prediction](https://docs.ultralytics.com/modes/predict/),
[training](https://docs.ultralytics.com/modes/train/),
[export](https://docs.ultralytics.com/modes/export/), and
[OpenCV package guidance](https://pypi.org/project/opencv-python/).
