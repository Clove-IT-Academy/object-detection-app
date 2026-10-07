# Acceptance criteria and current evidence

Functional requirements: valid annotated preview, readable labels, inspectable thresholds,
single model load, bounded frame/timing memory, no default webcam media persistence, controlled
startup/read/inference errors, and independent cleanup on every exit. Synthetic tests cover
geometry, filtering, invalid outputs, controls, data integrity, and lifecycle. Actual camera
controls/window closure and sustained operation require manual validation.

Accuracy: project-specific per-class precision/recall/AP floors and aggregate targets remain
unresolved until classes, dataset, and scenario are agreed. COCO pretraining is not project
acceptance evidence. Select settings on validation; preserve final test evaluation.

Performance: measured camera-free CPU baseline is recorded under `reports/`. Full-loop latency,
acceptable tail latency, sustained test duration, and any webcam FPS target remain unresolved.
Do not treat the model benchmark as application-level acceptance.

Export smoke gate: identical classes/settings, maximum score difference 0.02 and coordinate
difference 2 pixels, no missing/extra detections on supplied images. Public sample + blank parity
passed; representative validation/test dataset metrics and camera integration are still required.

Release gate: all agreed criteria, exported-model validation, clean documented setup, supported
platform camera checks, rights/redistribution review, maintenance owner, and rollback record.
Current version is a development build; this release gate has not been met.
