# Maintenance and rollback

Maintainer: unassigned; resolve before release. Capture non-sensitive issue details: app/model
version, artifact checksum, Python/dependencies, OS/device, camera index, steps and concise error.
Do not request webcam footage by default.

Retain the previous verified model bundle, config, source and lock. Use `--bundle` / `--config`
to roll back; never mix a class map with another model. Use new output directories for training,
evaluation, and export. Check checksums, geometry, failure paths, model integration and benchmarks
after dependency/runtime changes. Dataset or class changes also require new independent splits,
training/validation, export parity, and held-out evaluation.
