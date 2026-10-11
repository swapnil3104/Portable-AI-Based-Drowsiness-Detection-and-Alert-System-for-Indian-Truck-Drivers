# DrishtiDrive AI — Development Rules

**Project:** DrishtiDrive AI  
**Tagline:** *A Watchful Eye on Every Journey*  
**Document:** `rules.md`  
**Version:** 1.0  
**Last updated:** 11 October 2026  
**Applies to:** All contributors and AI coding assistants working on this repository

---

## 1. Purpose

These rules define how DrishtiDrive AI must be designed, implemented, tested, documented, and maintained. Follow them when creating or modifying code, training models, changing the UI, or updating project documents.

The initial product is the **Software Edition**, built with **Python, OpenCV, and a CNN trained on labelled open-eye/closed-eye images**. The future Raspberry Pi Edge Edition is a separate deployment target and must be validated on its actual hardware.

## 2. Non-Negotiable Safety Rules

1. Treat the project as an experimental prototype, not a certified vehicle-safety system.
2. Never claim the application guarantees drowsiness detection, prevents accidents, or diagnoses fatigue.
3. Never connect prototype software to steering, braking, the vehicle horn, or exterior-light controls.
4. Do not trigger a critical warning based on a single closed-eye frame. Use tested, time-based logic.
5. When observations are unreliable, show an explicit unavailable/uncertain state. Never silently interpret missing detections as `OPEN`.
6. Alerts must advise the driver to stop at a safe location and rest. Never instruct sudden braking or other unsafe maneuvers.
7. Do not conduct live tests while actively driving on public roads. Use recorded videos, a stationary vehicle, or a controlled test setup.
8. Treat thresholds as experimental configuration until validated; do not call them medically validated.
9. If model, camera, or audio fails, communicate the failure clearly rather than implying monitoring is healthy.

## 3. Core Technology Rules

- Use **Python** for application and training code.
- Use **OpenCV** for camera capture, frame handling, image operations, and rendering overlays.
- Use **TensorFlow/Keras** for training and evaluating CNN candidates unless a documented compatibility or performance reason justifies another framework.
- Keep face/eye localization separate from eye-state classification. A CNN trained on eye crops does not locate eyes in a full frame by itself.
- Load the selected model once at startup; do not reload it for every frame.
- Keep core inference and alerts functional offline.
- Avoid adding YOLO or other object-detection frameworks unless a demonstrated requirement justifies the additional complexity.
- Use a separate runtime adapter if the selected model needs conversion for Raspberry Pi.

## 4. Dataset and Data Governance Rules

1. Review the source, licence, access requirements, label definitions, and permitted usage before downloading or using a dataset.
2. Recommended starting point: **MRL Eye Dataset** — https://mrl.cs.vsb.cz/eyedataset
3. Possible alternative/supplement: **Kaggle open/closed-eye datasets** — https://www.kaggle.com/search?q=closed+eyes+open+eyes+dataset
4. Later sequence-level evaluation option: **NTHU Drowsy Driver Detection Dataset** — https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
5. Never commit downloaded datasets, private driver footage, personal information, or restricted data to the repository unless explicitly permitted and necessary.
6. Keep original dataset labels and metadata where possible; document any mapping to `open` and `closed`.
7. Split data into training, validation, and test sets before model selection. Prefer person/session-aware splits where metadata permits.
8. Prevent leakage: near-duplicate images or frames from one subject/session should not cross splits when avoidable.
9. Fit data-dependent preprocessing only on training data. Apply augmentation to training data only, not validation/test data.
10. Record dataset version/source, class counts, split manifests, random seeds, preprocessing, and augmentation choices.
11. Inspect class imbalance and apply class weighting or sampling only with a documented rationale.
12. Do not use the test set repeatedly to tune hyperparameters or choose the best checkpoint.

## 5. Model Training and Comparison Rules

The required initial benchmark contains five candidates:

1. Custom CNN
2. MobileNetV2
3. EfficientNetB0
4. ResNet50
5. Depthwise-separable CNN

Rules:
- Use comparable data splits and document any model-specific input differences.
- Save training history, best validation checkpoint, configuration, seed, runtime, and package versions.
- Select checkpoints using validation data, not the test set.
- Evaluate all final candidates on the same held-out test set.
- Report accuracy, macro precision, macro recall, macro F1, per-class metrics, closed-eye recall, confusion matrix, inference latency, and model size.
- Include false-positive and false-negative analysis, especially cases where closed eyes are missed.
- Select the final model based on safety-relevant metrics, robustness, latency, and size—not accuracy alone.
- Never fabricate results, write assumed scores into reports, or call a model “best” before actual evaluation.
- If a run fails, document the error and whether the comparison is still fair.
- Record a model card with intended use, dataset details, limitations, and measured performance.

## 6. Preprocessing and Inference Consistency

1. Training and live inference must use the same input dimensions, color-channel convention, normalization, and label order.
2. Store or document preprocessing settings alongside the model.
3. Validate input shape, dtype, and pixel range before inference.
4. Map model output indices to class names explicitly; never depend on implicit folder ordering.
5. Return a safe `UNCERTAIN`/unavailable state when the eye crop is invalid or confidence is insufficient.
6. Add tests that compare training preprocessing and inference preprocessing.
7. Avoid heavy image transformations on every frame unless they are necessary and benchmarked.

## 7. Real-Time Detection Rules

- Use a documented face/eye localization method, such as an appropriate OpenCV detector or facial landmarks.
- Handle no face, partial face, eye occlusion, glasses, multiple faces, low light, and camera disconnection.
- Use timestamps and elapsed time for closure duration; do not rely only on a fixed frame count because FPS can vary.
- Smooth noisy predictions without delaying the detection of sustained closure excessively.
- Use a state machine with documented transitions and configurable thresholds.
- Reset or pause timing when observations become unreliable.
- Keep camera capture, inference, UI, and speech playback from blocking each other where practical.
- Measure actual FPS and end-to-end latency on the target computer.

## 8. Alert and Voice Rules

**Required voice principle:** Use short, clear, firm voice alerts that get the driver's attention without creating unnecessary panic.

Default messages:

- **Stage 1 — Early warning:** “Please stay alert. Keep your eyes on the road.”
- **Stage 2 — Warning:** “Warning! You may be getting drowsy. Please find a safe place to stop.”
- **Stage 3 — Critical:** “Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.”

Implementation rules:
1. Prefer prerecorded local audio for predictable, low-latency playback; offline TTS may be used as a fallback.
2. Prevent overlapping messages and unbounded queues.
3. Add configurable cooldowns so repeated predictions do not produce constant speech.
4. Provide manual audio testing and volume control.
5. Keep sounds audible and distinct but not unnecessarily startling.
6. Show visual alerts alongside audio where possible.
7. Optional Hindi/Marathi support must preserve the same safety meaning and be tested for clarity.
8. Never make a warning depend on internet access.

## 9. UI and UX Rules

- Brand the UI as **DrishtiDrive AI** with the tagline **“A Watchful Eye on Every Journey.”**
- Keep the live camera feed dominant.
- Do not add a left navigation sidebar.
- Place Driver Metrics and Event Log to the right, with Camera Controls and Alert System in lower panels where practical.
- Use a restrained palette: light/neutral surfaces, navy/slate text, muted blue accent, and restrained green/amber/red status colors.
- Avoid neon colors, excessive gradients, decorative AI effects, and unnecessary animations.
- Clearly display camera, model, and audio status.
- Show eye-state label, confidence, closure duration, FPS/latency, current alert stage, and recent events.
- Label any overall risk value as an experimental estimate; do not present it as a medically validated score.
- Use meaningful error messages and accessible text sizes.
- Keep the UI responsive during inference and voice playback.

## 10. Privacy and Logging Rules

1. Process frames locally by default.
2. Do not record or save camera frames, eye crops, or video unless a clearly documented feature requires it and the user explicitly enables it.
3. Store only minimal event metadata by default: timestamp, event type, alert stage, and relevant non-identifying measurements.
4. Make logging and retention behavior clear; bound log size and retention.
5. Do not send images or personal data to external services.
6. Exclude datasets, generated models, private footage, logs, caches, secrets, and virtual environments from version control by default.
7. Do not log credentials or sensitive environment values.

## 11. Code Quality and Architecture Rules

- Keep modules small, focused, and testable.
- Separate camera capture, localization, preprocessing, model inference, temporal logic, alerts, UI, and event logging.
- Centralize configurable thresholds and settings in a configuration file.
- Avoid duplicated logic and unexplained magic numbers.
- Use descriptive names, type hints for important interfaces, and docstrings for non-obvious behavior.
- Handle exceptions at component boundaries; do not hide errors with broad empty `except` blocks.
- Use structured logging rather than scattered debug prints in production paths.
- Do not introduce a dependency without a clear need; document major dependencies.
- Keep UI operations on the appropriate UI thread and use worker/background execution for long-running inference where needed.
- Release camera, audio, and file resources reliably.

## 12. Testing Rules

Every meaningful change should include tests appropriate to the change.

- Unit-test image preprocessing, class mapping, model loading, state transitions, cooldowns, and alert escalation.
- Integration-test the camera-to-model-to-state-to-alert pipeline using recorded clips or simulated frames.
- Test normal blinking and sustained closure separately.
- Test glasses, lighting variation, head movement, partial occlusion, and different camera positions.
- Test camera unavailable/disconnected, no face, invalid crop, corrupt model, and audio output failure.
- Measure false alarms and missed events on independent video sequences.
- Verify offline operation.
- Do not claim “tested” unless the test actually ran; include commands and results where appropriate.
- Use deterministic seeds where feasible and document unavoidable nondeterminism.

## 13. Git and Documentation Rules

- Use clear commit messages describing one logical change.
- Never commit secrets, personal footage, restricted datasets, or large binary artifacts without approval.
- Update `README.md`, `task.md`, `architecture.md`, and relevant requirements whenever behavior or structure changes.
- Keep model comparison reports aligned with the latest actual experiment outputs.
- Mark tasks complete only when acceptance criteria are met.
- Use `TBD` for unknown values instead of inventing them.
- Document setup commands, dataset preparation, training, evaluation, runtime launch, and troubleshooting.
- Check dataset and model licences before redistributing artifacts.

## 14. Raspberry Pi / Edge Edition Rules

- Treat Edge Edition as a separate deployment target; do not assume desktop performance transfers to Raspberry Pi.
- Benchmark the chosen model on actual hardware for FPS, latency, RAM, CPU, and temperature.
- Use appropriate protected DC-DC conversion for the target vehicle power supply; do not connect vehicle power directly to low-voltage electronics.
- Bench-test the camera, speaker, buzzer, and indicator separately.
- Keep initial hardware isolated from vehicle-control and exterior-light circuits.
- Require additional engineering and safety review before any vehicle integration.

## 15. Definition of Done

A change or feature is done only when:
- [ ] It meets the relevant requirement and has a clear acceptance criterion.
- [ ] Tests have run and their outcomes are known.
- [ ] Error paths and uncertainty states are handled.
- [ ] Documentation and configuration examples are updated.
- [ ] No dataset, privacy, or safety rule is violated.
- [ ] Reported performance claims are supported by measured results.

The overall Software Edition prototype is ready for a controlled demonstration only after dataset governance, five-model comparison, OpenCV integration, temporal alert logic, UI integration, offline checks, and safety/limitations documentation are complete.

---

## 16. Safety Notice

DrishtiDrive AI is an experimental aid. It can miss drowsiness or raise false alarms, especially with poor lighting, eyewear, head movement, or camera misalignment. It does not replace sleep, breaks, attentive driving, or certified vehicle safety systems. If a driver feels sleepy, they should stop in a safe place and rest.
