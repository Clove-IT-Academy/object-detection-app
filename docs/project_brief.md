# Initial implementation brief

Baseline choice: official pretrained YOLO26n and its bundled 80 COCO classes. This is a
provisional runnable baseline while custom classes and representative project data remain
unresolved. The app does not perform identity recognition. Initial usage is a local webcam
preview on the development computer; target lighting/distances/scenarios need user confirmation.

One camera, local synchronous inference, no recording/uploads, boxes + names + confidence.
Controls are Q/Esc/native close and +/- confidence; camera/device/input size are startup settings.
Failed reads receive at most three retries while controls are polled, then exit with cleanup.
Invalid frame/model/output contracts stop with a controlled error. Model files are loaded once.

Platform evidence: installation and CPU still-image inference on Apple M4, arm64, macOS 27.0.1,
Python 3.14.7. Webcam permissions were denied, so camera/display support is not yet verified.
Release ownership, custom class scope, quantitative quality goals, and usage conditions remain open.
