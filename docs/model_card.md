# Baseline model card

Model: official YOLO26n pretrained detection checkpoint. Source:
`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`.
Original SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Class order and artifact checksums are preserved in local `models/*/bundle.json`.

80 COCO classes; no custom fine-tuning in the app baseline. Framework-owned BGR uint8 input
preparation and original-pixel xyxy decoded outputs. CPU input size 640, class-aware suppression
when required, confidence 0.35, IoU 0.7, max detections 100. These settings require validation
selection for any custom use.

Deployment: FP32 static ONNX, opset 18 observed in exporter output, ONNX Runtime CPU. The
original checkpoint remains separate. Image parity passed at score tolerance 0.02 and box
tolerance 2 pixels on the public bus sample and an empty non-square image. No representative
custom dataset accuracy or held-out test mAP is claimed.

Terms: Ultralytics AGPL-3.0 / Enterprise; verify applicable redistribution obligations before
release, along with dataset/runtime terms. Known limitations: fixed trained class vocabulary,
occlusion, tiny objects, lighting/domain shift, no identity or tracking. See baseline benchmark
and verification reports. Synthetic one-epoch training artifacts are separate smoke fixtures.
