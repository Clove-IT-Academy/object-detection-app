# Object Detection App: Development Plan

Date: 2026-10-07  
Status: Development implementation available; see `docs/IMPLEMENTATION_STATUS.md` for verified work and unresolved gates.  
Source: `Object_Detection_System_Plan_and_Architecture.pdf`, all eight pages.  
Purpose: Provide durable context and an ordered execution plan for building, training, testing, releasing, and maintaining the application.

## 1. How to use this document

Read the architecture PDF and this plan before implementation. The PDF defines the intended system; this document turns it into tasks, deliverables, dependencies, and completion gates. Proposed implementation choices below are planning recommendations, not decisions already made by the user.

Work through the numbered steps in order, completing each gate before depending on its output. Camera and rendering work may proceed with a fake detector while data preparation or training is underway. Do not treat that scaffold as evidence of model accuracy. Record actual decisions, commands, results, and deviations in the designated project documents. Update the progress checklist at the end as work is completed.

Paths in this document are relative to `object-detection-app/`. They describe planned files unless explicitly identified as existing. At the initial planning review, the project contained only the architecture PDF and an existing `venv/`. The subsequent implementation adds source, tests, configs, documentation, a pretrained checkpoint and a smoke-verified ONNX export. No custom project dataset or quality-validated custom model exists. See the implementation status before continuing; an environment's presence alone does not establish compatibility.

## 2. Product goal and boundaries

Build a local webcam application that detects instances of a defined set of object classes and displays each retained object's bounding box, class name, and confidence score. Train or fine-tune a YOLO detector on an annotated, versioned dataset; evaluate it; export it to a verified deployment format; and integrate the selected artifact into the application.

Required behavior:

- Open one webcam, validate incoming frames, and show an annotated live preview.
- Load model weights and their matching class metadata once at startup.
- Apply inspectable confidence and postprocessing settings.
- Handle unavailable cameras, failed reads, model load failures, malformed detections, and inference failures with useful errors.
- Support quitting and window closure; release the camera and display resources on every exit path.
- Process frames locally. Do not save images, record video, or upload frames by default.
- Document class limits, measured accuracy, hardware-specific performance, provenance, and setup.

Initial release exclusions: identity recognition, arbitrary-object recognition, tracking across frames, multiple simultaneous cameras, cloud inference, a web dashboard, and automatic recording. Any later extension requires an explicit scope update. Confidence is a model score, not a guaranteed probability of correctness.

## 3. Decisions to resolve before dependent work

| ID | Decision | Proposed starting point | Resolve by | Record |
| --- | --- | --- | --- | --- |
| D01 | Exact object classes and display names | A manageable set of visually distinct classes relevant to the intended use | Step 1 | `docs/project_brief.md` |
| D02 | Intended setting and camera | Document actual backgrounds, distances, lighting, and webcam | Step 1 | Project brief |
| D03 | Supported OS, device, compute, and memory | Measure on the user's intended computer; establish CPU fallback only if supported and tested | Step 2 | `docs/environment.md` |
| D04 | Accuracy and responsiveness thresholds | Agree per-class quality floors and application latency targets after a baseline; no invented FPS promise | Initial criteria in Step 1; finalize in Step 3 | `docs/acceptance_criteria.md` |
| D05 | Python/framework/model versions | Python, OpenCV, and a YOLO framework are candidates; verify compatibility, official documentation, and terms before pinning | Step 2–3 | Dependency lock and decision log |
| D06 | Dataset source, rights, and annotation tool | Representative licensed or permitted images plus consented captures as needed | Step 4 | Dataset manifest |
| D07 | Deployment runtime and format | Evaluate a suitable export such as ONNX against target hardware; retain the training checkpoint | Step 10 | Model manifest and decision log |
| D08 | Controls and failure policy | Quit key and window close; startup camera/threshold settings; bounded read retries followed by exit | Step 1 and Step 7 | Project brief and config |
| D09 | Artifact storage and release ownership | Local versioned bundles initially; choose storage, redistribution scope, and maintainer before release | Step 2 and Step 12 | Release notes and ownership guide |

If a decision is still open, label it unresolved. Do not silently assume class names, a model generation, GPU support, export compatibility, numerical targets, or permission to redistribute data/weights.

## 4. Proposed technical layout and contracts

Start with a small Python application using OpenCV for capture, drawing, display, and key input; a detector adapter isolates the chosen YOLO framework and later export runtime. Use a synchronous loop first. Add a capture worker with a latest-frame slot only if measurements show it is necessary. Avoid a server or frontend unless scope changes.

```text
object-detection-app/
  DEVELOPMENT_PLAN.md
  Object_Detection_System_Plan_and_Architecture.pdf
  README.md
  pyproject.toml                 # project metadata and tested dependency specification
  <dependency lock file>        # choose one reproducible dependency workflow
  .gitignore
  configs/
    app.yaml                    # camera, runtime, thresholds, input size, display
    dataset.yaml                # framework-compatible split paths and class order
    train.yaml                  # reproducible training settings
  src/object_detection_app/
    __init__.py
    __main__.py                 # startup/CLI entry point
    config.py
    contracts.py                # frame transforms, detections, bundle metadata
    camera.py
    preprocessing.py
    detector.py                 # interface and adapter selection
    backends/                   # framework adapter and selected export adapter
    postprocessing.py
    renderer.py
    controller.py               # lifecycle, controls, loop, error handling
    timing.py
  scripts/                      # data validation, training, evaluation, export, benchmarks
  tests/                        # unit, data, integration, lifecycle tests
    fixtures/                   # small permitted, non-sensitive samples
  docs/
    project_brief.md
    acceptance_criteria.md
    environment.md
    annotation_guidelines.md
    decision_log.md
    dataset_card.md
    model_card.md
    release_checklist.md
    maintenance.md
  data/                         # large/private data excluded from source control
  models/                       # checkpoints and deployable bundles; storage policy applies
  runs/                         # versioned run configs, metrics, and logs
  reports/                      # evaluation, export parity, performance, release notes
```

Do not commit `venv/`, private images, large generated runs, or weights indiscriminately. Keep small authorized test fixtures and artifact manifests in source control; document how to obtain external artifacts and verify their checksums.

### Shared contracts

Define these before adapters diverge:

- **Camera frame:** nonempty image, known width/height, documented color order and dtype; explicit failure rather than an ambiguous empty frame.
- **Preprocessing result:** model input plus original dimensions, model dimensions, scale and padding needed to invert resize/letterbox. Record tensor layout, dtype, color order, normalization, and batch semantics in the backend contract.
- **Detection:** `class_id`, display name resolved from the bundle, finite score in `[0, 1]`, and finite `xyxy` coordinates in original-frame pixels. Require positive area after clipping.
- **Detector interface:** load/validate bundle once, predict against its declared input contract, return declared raw or decoded output, and close if backend resources require it.
- **Model bundle:** artifact, ordered class map, model/dataset version, input/output specification, preprocessing owner, NMS owner, runtime requirements, thresholds, provenance, and checksum.
- **Timing sample:** capture, preprocessing, inference, postprocessing, rendering/display, and total loop duration, with clear boundaries and monotonic timing.

For a framework adapter that already preprocesses or performs NMS, document that ownership and consume its decoded output. Never preprocess twice or apply NMS twice. Framework-returned coordinates may already refer to the original image; verify before applying inverse transforms.

## 5. Ordered development steps

### Step 1 — Freeze the initial product brief

**Depends on:** Architecture review.

1. Choose classes and assign stable, contiguous IDs and display names.
2. Describe representative usage, expected object sizes/distances, lighting, occlusion, backgrounds, and empty scenes.
3. Specify preview behavior, startup settings, controls, failure behavior, and local-only privacy defaults.
4. Define acceptance measurements: per-class precision/recall and AP, aggregate mAP, loop latency, stability duration, and cleanup behavior. Record numerical targets as pending until baseline measurement where appropriate.
5. Define supported release scope and identify the person who owns release decisions.

**Outputs:** Project brief, initial acceptance criteria, initial decision log.

**Gate:** Class list and intended scenario are explicit; exclusions and privacy behavior are documented; every unresolved target has a resolution step.

### Step 2 — Establish a reproducible project environment

**Depends on:** Step 1; hardware and platform information.

1. Inspect the existing environment without assuming it is reusable. Check selected Python, framework, camera/display library, and export runtime support against current official documentation.
2. Select dependency management, create the source layout, configure package entry points, and pin a tested dependency set. Use a fresh environment if the existing one is unsuitable.
3. Add configuration validation, logging conventions, ignore rules, and basic automated checks.
4. Record OS, CPU/GPU, RAM, camera, device/runtime selection, installation commands, and tested versions.
5. Establish artifact naming/storage and provenance records. Keep model downloads explicit and reproducible; avoid downloading weights implicitly during every app startup.

**Outputs:** Installable scaffold, environment notes, dependency specification/lock, initial configuration and artifact policy.

**Gate:** A clean environment can install the project and run a smoke entry point; invalid configuration produces a clear error. No camera footage is persisted by startup.

### Step 3 — Run a feasibility baseline and settle targets

**Depends on:** Step 2.

1. Load a suitable small pretrained detector and run a permitted still image.
2. Confirm camera access and preview support on target hardware; measure short warm-up and steady-state runs.
3. Record model size, input resolution, device, startup cost, inference latency, and total loop timing. Synchronize asynchronous device work when timing it.
4. Determine whether existing pretrained classes match the project. Use this baseline to validate tooling and estimate compute needs; custom-class quality remains unproven until fine-tuning.
5. Select a starting checkpoint/input size and finalize measurable accuracy and latency targets before model selection. Record the false-alert versus missed-object tradeoff.

**Outputs:** Baseline report, chosen training candidate, finalized acceptance thresholds and benchmark protocol.

**Gate:** Target hardware can run the pipeline; targets are measurable and realistic. If compute is insufficient, revise model size, resolution, training resources, or agreed scope before proceeding.

### Step 4 — Collect and annotate a representative dataset

**Depends on:** Class definitions and data-rights decisions; Step 3 informs feasibility.

1. Write annotation rules for partial, occluded, truncated, overlapping, ambiguous, and absent objects. Decide when to exclude an image rather than assign an uncertain label.
2. Collect varied examples and empty scenes; record source, capture/session identity, rights or consent, and counts by class and scenario.
3. Annotate every in-scope object consistently. Use YOLO detection labels (`class_id center_x center_y width height`, normalized) if required by the selected framework.
4. Validate image decoding, label parsing, class IDs, finite values, normalized geometry, and image/label pairing. Define the framework-compatible representation of negative images explicitly.
5. Visually review a stratified sample and all flagged labels. Resolve duplicate, missing, wrong-class, loose, and invalid boxes.

**Outputs:** Annotated dataset candidate, annotation guide, dataset card/manifest, class and scenario coverage summary, review report.

**Gate:** Rights are documented; labels pass validation; review findings are resolved; meaningful gaps in class/scenario coverage are addressed. Total image count alone is not a quality gate.

### Step 5 — Split and version the dataset

**Depends on:** Step 4.

1. Group images by recording session, source, related burst, or other shared origin before splitting. Detect exact duplicates and review near-duplicates.
2. Choose and document train/validation/test proportions based on available groups and class coverage; do not force an arbitrary ratio at the cost of leakage.
3. Ensure each evaluation class has suitable independent coverage. Recollect data when there are too few independent groups.
4. Generate split manifests and framework dataset configuration; record seed, class order, dataset revision, counts, and hashes or stable image identifiers.
5. Freeze the final test split. Use train data for fitting and validation data for model, threshold, augmentation, and export choices.

**Outputs:** Versioned train/validation/test data manifests and validated `configs/dataset.yaml`.

**Gate:** No known source/session overlap across splits; duplicates are accounted for; class order matches all labels; the frozen test set is reserved for final evaluation.

### Step 6 — Fine-tune and select a candidate model

**Depends on:** Steps 3 and 5.

1. Run a tiny training smoke test to verify labels, class order, device, and output generation before a full run.
2. Record source checkpoint and license, dataset revision, seed, framework versions, input size, batch size, epochs, optimizer settings, augmentations, and stopping criteria.
3. Fine-tune the initial model; retain checkpoints, logs, run config, and validation results.
4. Inspect train/validation curves, class-specific performance, and validation predictions. Check whether augmentations preserve realistic semantics.
5. Compare a limited, documented set of improvements using validation data: better labels/coverage, class balance, capacity, input size, or training settings.
6. Select the checkpoint and operating threshold against the acceptance criteria. Package the matching class map and provenance.

**Outputs:** Reproducible training runs, selected checkpoint, validation report, draft model card and bundle manifest.

**Gate:** Candidate meets validation quality requirements or has an explicit unresolved gap and remediation plan. A completed training job alone does not satisfy this gate.

### Step 7 — Implement runtime components with synthetic tests

**Depends on:** Step 2 and finalized contracts; can begin before Step 6 finishes using fake outputs.

1. Implement validated config and command-line overrides for model bundle, camera, device/runtime, input size, confidence, and applicable NMS/IoU settings.
2. Implement camera open/read/release, frame validation, bounded retries, and explicit errors. Release any partially acquired resources when startup fails.
3. Implement backend-specific preprocessing and inverse geometry, or declare preprocessing delegated to the framework.
4. Implement detector load/predict contracts; check bundle/class-map compatibility and reject unsupported shapes or runtimes clearly.
5. Implement score filtering, class lookup, single-owner NMS, coordinate restoration, clipping, and invalid-output rejection. Document class-aware versus class-agnostic NMS behavior.
6. Implement readable boxes/text, safe label placement near image edges, no-detection display, quit/window-close controls, and `finally`-style cleanup.
7. Test rectangular images, padding, edge boxes, empty outputs, unknown class IDs, non-finite scores/coordinates, and overlapping boxes. Use a fake camera/detector to test error paths without physical hardware.

**Outputs:** Runtime modules and focused automated tests.

**Gate:** Synthetic geometry maps correctly; invalid detections cannot produce unsafe drawing coordinates; threshold semantics and NMS ownership are explicit; every lifecycle path cleans up.

### Step 8 — Integrate the trained model into the live app

**Depends on:** Steps 6 and 7.

1. Validate and load the selected training bundle once, then run capture → validate → preprocess → infer → postprocess → render → display → controls.
2. Verify class names and coordinates on permitted fixture images before live testing.
3. Test single/multiple objects, overlapping instances, empty scenes, varied lighting, partial occlusion, and objects outside the class list.
4. Test missing/corrupt model, invalid metadata, camera busy/unavailable, repeated failed reads, inference exception, quit, keyboard interruption, and window closure.
5. Run a sustained live session for the agreed duration; inspect memory/resource behavior, responsiveness, and errors. Save aggregate timing/error metadata, not frames, by default.

**Outputs:** Working live application using the trained checkpoint, integration tests, hardware smoke-test record.

**Gate:** Required behavior and failure handling work on the primary platform; the selected model is loaded once; resources are released on all exits; default operation writes no webcam media.

### Step 9 — Benchmark and improve responsiveness

**Depends on:** Step 8.

1. Measure capture, preprocessing, inference, postprocessing, rendering/display, and total loop time separately. Report hardware, input/capture resolution, device, batch size, warm-up, sample count, and method.
2. Report median and tail latency (for example p95), processed FPS, startup time, and relevant memory use. Distinguish model throughput from application responsiveness.
3. Identify the measured bottleneck. Try justified changes one at a time: model size, input size, runtime/device, drawing cost, or capture policy.
4. Retain a synchronous loop when it meets targets. If necessary, add a bounded latest-frame buffer, explicit drop policy, timestamped frame age, thread-safe shutdown, and camera ownership.
5. Recheck validation quality and lifecycle behavior after performance-affecting changes. Avoid unbounded queues and unexplained frame skipping.

**Outputs:** Performance report, documented capture policy, regression evidence for any optimization.

**Gate:** Application meets agreed latency/stability targets on target hardware, or release scope/targets are explicitly revised. Accuracy regressions are measured rather than assumed absent.

### Step 10 — Export and verify the deployment bundle

**Depends on:** Selected checkpoint, runtime decision, and baseline integration/benchmark evidence.

1. Select a target-supported format using current official documentation; export from the retained training checkpoint with recorded precision, input shape, opset/options where applicable, and package versions.
2. Package export, exact class map, checksums, preprocessing specification, output schema, NMS ownership, thresholds, and runtime/device requirements.
3. Implement the export backend behind the existing detector interface. Inspect the actual output contract rather than guessing tensor layout or postprocessing.
4. Compare checkpoint/export results on a fixed validation verification set using identical settings. Define score/box tolerances and matching rules before comparing; include empty scenes, overlapping objects, and image aspect-ratio extremes.
5. Recompute exported-runtime validation metrics where possible; investigate missing/extra detections and changes near thresholds. Document any evaluation limitation and compensating evidence.
6. Benchmark export startup and steady-state application performance, then repeat webcam and failure-path tests with the deployment bundle.

**Outputs:** Versioned deployment artifact, export backend, parity report, runtime benchmark, updated model card.

**Gate:** Export meets predefined parity and quality tolerances, integration tests, and performance requirements. If export fails, fix/re-export or choose a compatible format/runtime; do not call it release-ready merely because conversion completed.

### Step 11 — Perform final evaluation and release verification

**Depends on:** Steps 9 and 10; candidate model/settings frozen.

1. Evaluate the original checkpoint and deployment runtime on the reserved test split with the frozen settings. Report precision, recall, per-class AP, mAP@0.5, mAP@0.5:0.95, and evaluation protocol.
2. Review test prediction examples and false positives, misses, class confusion, localization errors, small objects, occlusion, and empty scenes. Report class/scenario weaknesses honestly.
3. Run the full automated suite and supported-platform camera smoke checks. Verify installation from a clean environment using release artifacts.
4. Check local-only behavior, artifact integrity, dependency compatibility, camera/window cleanup, and sustained operation against acceptance criteria.
5. Make a release decision using class-wise quality, runtime performance, and system reliability together.

**Outputs:** Final test evaluation, verification matrix, completed release checklist, documented known limitations.

**Gate:** All mandatory criteria pass and every claimed supported platform has evidence. If test failure drives further development, do not repeatedly tune on that test set: log the exposure and reserve new independent test data for the revised release.

### Step 12 — Document, package, and hand off the release

**Depends on:** Step 11 and license/provenance review.

1. Finish README instructions for clean install, model acquisition and checksum verification, startup commands, controls, supported hardware/platforms, training/export reproduction, troubleshooting, and privacy behavior.
2. Complete dataset/model cards, run manifest, license/terms review for code, weights, data, and runtime, plus measured quality/performance summaries.
3. Version code and model bundles; record their compatibility. Package only artifacts that may be redistributed and explain acquisition for excluded artifacts.
4. Create release notes with supported classes, known failure conditions, acceptance evidence, and rollback instructions to the previous verified bundle/config.
5. Assign maintainer and issue-report fields: app/model version, OS/device, reproduction steps, and errors. Do not request webcam footage by default.

**Outputs:** Versioned release package, complete documentation, maintenance handoff.

**Gate:** Another developer can install, run, and trace the release to its data/model/configuration without relying on chat history; maintainer and artifact sources are identified.

### Step 13 — Maintain and iterate

**Depends on:** Released baseline.

1. Triage issues using non-sensitive diagnostics; reproduce failures and add a focused regression fixture/test where appropriate.
2. Review dependencies, compatibility, licenses, provenance, and model limitations periodically.
3. For changed classes or data, create a new dataset/model version and repeat split integrity, training, evaluation, export parity, and integration gates.
4. For runtime/dependency changes, repeat affected functional and benchmark checks on supported targets.
5. Preserve the previous verified release and document migration/rollback. Keep plan and decision log aligned with actual scope.

**Outputs:** Maintenance records and subsequent verified releases.

**Gate:** Updates retain reproducibility, class-map compatibility, quality evidence, and a working rollback path.

## 6. Test and evidence matrix

| Area | Essential evidence | Introduced | Required before release |
| --- | --- | --- | --- |
| Data integrity | Parse/geometry/class checks, negative-image handling, split group and duplicate checks | Steps 4–5 | Versioned validation report |
| Geometry | Known synthetic boxes through resize/letterbox inverse; clipping, padding, non-square inputs | Step 7 | Passing focused tests |
| Postprocessing | Confidence boundaries, NMS ownership, class-map mismatch, invalid outputs, empty results | Step 7 | Passing backend contract tests |
| Lifecycle | Camera/model startup failures, failed reads, inference errors, quit/window close, partial-startup cleanup | Steps 7–8 | Automated fakes plus actual camera smoke record |
| Model quality | Per-class and aggregate metrics, validation selection, untouched-test final evaluation | Steps 6 and 11 | Quality floors met; limitations documented |
| Export fidelity | Same-image checkpoint/export comparisons, predefined tolerances, runtime evaluation | Step 10 | Parity report and passing integration |
| Performance | Warm-up, median/tail stage and loop timing, hardware/device/settings, sustained run | Steps 3 and 9–10 | Target-specific acceptance report |
| Privacy | No default recordings/uploads; inspect logging/output and artifact contents | Steps 8 and 11 | Verified default behavior |
| Reproduction | Clean install, artifact checksum, documented launch and run provenance | Steps 2 and 12 | Independent setup smoke test |

Use small deterministic fixtures for logic tests and real model fixtures for backend integration. Keep hardware-dependent tests clearly marked. Automate tests where practical, but do not replace live camera checks with mocks.

## 7. Milestones and execution dependencies

| Milestone | Completed steps | Reviewable result |
| --- | --- | --- |
| M1: Scope and feasibility | 1–3 | Explicit classes, environment, baseline, measurable criteria |
| M2: Data ready | 4–5 | Reviewed dataset with independent, versioned splits |
| M3: Model candidate | 6 | Reproducible checkpoint and validation evidence |
| M4: Live app | 7–8 | Modular webcam application with trained model and failure handling |
| M5: Deployable candidate | 9–10 | Measured responsiveness and verified exported bundle |
| M6: Release | 11–12 | Final evaluation, clean-install verification, documentation, ownership |
| M7: Ongoing maintenance | 13 | Tracked issues and verified updates |

Critical dependency path: scope → environment/baseline → data quality/splits → training/validation → trained-model integration → performance/export verification → final test evaluation → release.

Runtime scaffolding and synthetic tests can run alongside data work after contracts are settled. Documentation and provenance records should be updated during each step. Calendar estimates should be added after Step 3 and the initial data audit; dataset collection, annotation, and training hardware availability are the main unknowns.

## 8. Risks and planned responses

| Risk | Signal | Response |
| --- | --- | --- |
| Classes or setting remain vague | Inconsistent labels or changing class map | Resolve Step 1 and version any later scope change |
| Data leakage | Related sessions or duplicate imagery in different splits | Regroup/re-split before training; retain audit manifests |
| Poor generalization | Strong aggregate score but weak classes/live scenes | Review labels and scenario coverage; improve data using validation evidence |
| Unsupported environment | Install, camera display, device, or export failures | Verify compatibility early; rebuild reproducibly with supported versions |
| Low responsiveness | High tail loop time or stale frames | Measure stages; optimize bottleneck and bound buffering |
| Incorrect boxes | Offset overlays, wrong aspect ratio, border errors | Test inverse transforms and adapter coordinate ownership |
| Export changes behavior | Score/class/box mismatch or metric regression | Inspect export contract and settings; re-export or change supported runtime |
| Wrong model/class map | IDs display incorrect object names | Bundle and checksum together; reject incompatible metadata |
| Privacy or redistribution issue | Unrequested recordings or unclear artifact rights | Keep capture transient; document rights and exclude unauthorized artifacts |
| Test-set overuse | Repeated selection using test results | Freeze settings before final test; acquire independent replacement test data after exposure |

## 9. Instructions for future implementation context

1. Read this plan, the architecture PDF, and applicable repository instructions before editing.
2. Identify the earliest incomplete step and inspect its actual artifacts before continuing. Do not assume planned paths already exist.
3. Resolve decisions that block the current step; continue independent scaffold or documentation work when possible.
4. Consult current official framework/runtime documentation when choosing versions, weights, annotation format, export options, or licenses. The source PDF does not freeze a specific YOLO version.
5. Preserve separate capture, preprocessing, inference, postprocessing, renderer/controller, configuration, and timing responsibilities.
6. Keep class maps attached to model versions, make preprocessing/NMS ownership explicit, and validate coordinates before rendering.
7. Keep webcam media transient by default. Model training and export are explicit workflows rather than normal app startup work.
8. Update the decision log and relevant report with actual commands/configuration, artifact locations, versions, measured results, limitations, and gate status.
9. Mark a step complete only when its outputs and gate are satisfied. Record blockers precisely; do not report unmeasured accuracy/FPS or untested platform support.
10. At each handoff, state completed steps, current artifact/model/dataset versions, failed or pending checks, and the next concrete task.

## 10. Progress checklist

- [x] Review the architecture PDF and create this execution plan.
- [ ] Step 1: Product brief, class list, initial acceptance criteria.
- [x] Step 2: Reproducible environment and source scaffold (tested project venv; dependency lock and CLI available).
- [ ] Step 3: Feasibility baseline and finalized targets.
- [ ] Step 4: Representative data and reviewed annotations.
- [ ] Step 5: Leakage-aware versioned splits.
- [ ] Step 6: Fine-tuned candidate and validation selection.
- [x] Step 7: Runtime modules and synthetic tests (geometry, filtering, bundle, config and lifecycle checks pass).
- [ ] Step 8: Trained-model webcam integration.
- [ ] Step 9: Measured and accepted responsiveness.
- [ ] Step 10: Verified export and deployment bundle.
- [ ] Step 11: Final held-out evaluation and system verification.
- [ ] Step 12: Versioned release, documentation, ownership.
- [ ] Step 13: Maintenance process established.

## 11. Architecture traceability

| Source PDF section | Covered here |
| --- | --- |
| 1. Learning and build plan | Steps 1–13 and milestones |
| 2. Problem definition and requirements | Product boundaries, decisions, Steps 1 and 7–12 |
| 3. Training data and annotation plan; Figure 1 | Steps 4–6; dataset integrity and test protection |
| 4. YOLO model and training concepts | Baseline selection and Step 6 |
| 5. Evaluation and mAP | Acceptance criteria, Steps 6 and 11, evidence matrix |
| 6. Runtime architecture; Figure 2 | Technical layout/contracts and Steps 7–8 |
| 7. Frame inference sequence and safeguards | Contracts, Steps 7–9, geometry/lifecycle tests |
| 8. Model export and deployment | Step 10 and release artifact records |
| 9. Testing, release, and ownership | Evidence matrix and Steps 11–13 |

The PDF's official documentation references should be consulted during dependency/model selection: Ultralytics model export, detection datasets, performance metrics, and detection task documentation. This plan adds implementation sequencing and gates; it does not certify current package support or license terms.
