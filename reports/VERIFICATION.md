# Verification — 2026-10-07

Environment: Python 3.14.7 in project `venv`, macOS 27.0.1 arm64, Apple M4 CPU.

- Automated suite with local model tests: **40 passed** (37 focused + 3 actual-model tests).
- Ruff lint and formatting checked; project imports/help require no camera or weights.
- Pip dependency consistency checked; installed versions recorded in `requirements.lock.txt`.
- Missing bundle startup returns controlled error before camera acquisition.
- Public Ultralytics bus sample: five retained detections at confidence 0.35; overlays inspected
  visually. No personal webcam footage used as a fixture.
- Original checkpoint baseline: median **27.31 ms**, p95 **28.58 ms**, ten measured iterations
  after three warm-up iterations. See `pretrained_cpu_baseline.json`.
- ONNX CPU baseline: median **23.08 ms**, p95 **29.69 ms**, same sample/protocol. See
  `onnx_cpu_baseline.json`. Short runs are preliminary and exclude camera/display.
- Export: FP32 static 640 input, ONNX opset 18. Same-class output comparison passed on the
  public scene and blank non-square image at score tolerance 0.02 / coordinate tolerance 2 px.
  See `export_parity.json`. This is not a full dataset accuracy evaluation.
- Real inference integration tests deny Python socket connections and verify input immutability.
- Synthetic workflow smoke: four train/two validation/two test images, one class, one epoch,
  64-pixel model input; checkpoint, bundle and validation report produced successfully. Metrics
  were zero; this verifies workflow execution only. Synthetic final test split was not evaluated.
- Webcam attempt: `run --max-frames 5` failed cleanly because macOS denied access. Actual preview,
  controls, native close, disconnect, sustained operation and webcam FPS are **not verified**.

Pending manual checklist after granting camera access:

- [ ] Preview shows correctly aligned boxes/names/scores and normal empty-scene status.
- [ ] Q, Esc, and native window close exit without recreating a window.
- [ ] +/- adjusts the displayed/applied threshold; restart restores config.
- [ ] Busy/unavailable/disconnected camera and failed reads produce controlled errors.
- [ ] Missing/corrupt model or metadata fails before preview; resources remain released.
- [ ] Sustained run is responsive with stable memory on the intended camera/settings.
- [ ] Repeated starts after normal/error exits can reopen the camera.
- [ ] Capture/display/total-loop timings and target criteria are agreed and recorded.
- [ ] Representative per-class quality and exported-runtime final evaluation pass.
