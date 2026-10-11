# DrishtiDrive AI — Software Edition
## Project Requirements Document (PRD)

**Tagline:** *A Watchful Eye on Every Journey.*  
**Variant:** Software-only  
**Document type:** Detailed project requirements  
**Version:** 1.2  
**Status:** Updated — OpenCV + dataset-trained CNN workflow

---

## 1. Project Overview

DrishtiDrive AI — Software Edition is a computer-vision application that uses an existing webcam or connected camera to monitor visible signs associated with driver drowsiness. It processes camera frames, estimates eye openness and mouth movement, observes head pose where supported, and presents understandable status information and escalating software alerts.

This edition is intended for software development, academic demonstration, algorithm evaluation, and testing on a laptop or desktop. It does not require a dedicated Raspberry Pi device, vehicle wiring, or direct control of a car's horn, lights, steering, or braking.

**Safety limitation:** This is a prototype driver-assistance tool, not a certified safety system. It cannot guarantee that a driver is awake or safe to drive. It must not encourage a drowsy driver to continue driving; alerts should advise the driver to stop at a safe location and rest.

## 2. Problem Statement

Long-route driving, night driving, irregular sleep, and extended working hours can contribute to driver fatigue. Many vehicles do not have driver-monitoring systems. Commercial systems may be inaccessible for low-budget users or may require dedicated hardware. A software-only solution can make early experimentation and evaluation more accessible by using a computer and webcam already available to the user.

The project will provide a local, understandable monitoring interface that detects selected visual indicators, warns when configured thresholds are exceeded, and records system events for review. It must communicate uncertainty and camera failures clearly rather than treating missing data as a normal state.

## 3. Goals and Objectives

### 3.1 Goals
- Capture video from a user-selected webcam.
- Use OpenCV for camera capture, frame processing, and real-time display.
- Train and compare 4–5 candidate CNN/image-classification models on labelled open-eye/closed-eye images.
- Select the best model using held-out test metrics and practical inference latency, then integrate it into the live software pipeline.
- Detect eyes and derive optional mouth/head-pose indicators where supported.
- Estimate drowsiness risk from multiple frames rather than one isolated frame.
- Provide configurable three-stage software alerts.
- Display current metrics, camera status, alert status, and event history.
- Operate locally without requiring cloud access for core detection.
- Provide settings and logs useful for debugging and controlled evaluation.

### 3.2 Out of scope
- Direct connection to vehicle horn, parking lights, hazard lights, CAN bus, or ECU.
- Automatic vehicle control.
- Medical diagnosis or a guarantee of fatigue detection.
- Cloud-based driver surveillance by default.
- Certified production deployment without formal validation.

## 4. Target Users

- Students and researchers developing computer-vision projects.
- Developers evaluating drowsiness-detection approaches.
- Demonstrators who want to show live camera processing and alert logic.
- Test users using a laptop/desktop webcam in a stationary or controlled environment.

## 4A. Dataset and Model Strategy (Required)

### Selected approach
**OpenCV + dataset-trained CNN.** OpenCV is responsible for camera capture, preprocessing, eye-region cropping, overlays, and real-time application flow. A CNN trained on labelled eye images predicts `open` or `closed`. Drowsiness alerts must be based on persistent predictions over time, not a single closed-eye frame.

### Recommended datasets

1. **MRL Eye Dataset** — recommended primary candidate for open/closed eye-state classification. It contains eye images captured under varied conditions. Review its official terms, label format, and permitted use before downloading: https://mrl.cs.vsb.cz/eyedataset
2. **Kaggle open/closed eye image datasets** — useful alternatives or supplements, but select only a dataset with clear labels, documented provenance, an acceptable license, and enough variation in people, glasses, lighting, and camera quality: https://www.kaggle.com/search?q=closed+eyes+open+eyes+dataset
3. **NTHU Drowsy Driver Detection Dataset** — useful for later sequence-level validation of drowsiness-related behaviours (eye closure, yawning, head movement). Access may require an academic request/registration. It is not a drop-in replacement for a simple eye-crop dataset: https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

The team must record the dataset version/source, license, class counts, image dimensions, and preprocessing steps. Do not merge datasets blindly when label conventions or capture conditions differ.

### Data split and leakage prevention
- Prefer subject/person-independent train, validation, and test splits. Images from the same person or video sequence must not leak across splits.
- Target split: 70% train, 15% validation, 15% test, adjusted to keep subjects separate and class representation reasonable.
- Report class counts and imbalance; use class weights or training-only augmentation if needed.
- Apply augmentation only to training data (small rotations, brightness/contrast variation, modest shifts/zoom). Do not use transformations that change the eye-state label.
- Keep the test set untouched until model selection is complete.

### Required model comparison (4–5 candidates)
Train and evaluate at least these five candidates when compute/resources permit:
1. **Custom compact CNN** — baseline, trained from scratch.
2. **MobileNetV2 transfer learning** — lightweight candidate for real-time CPU and Raspberry Pi use.
3. **EfficientNetB0 transfer learning** — accuracy/compute trade-off candidate.
4. **ResNet50 transfer learning** — heavier comparison baseline; resize input and batch size to fit available memory.
5. **Small CNN with depthwise-separable convolutions** — edge-oriented lightweight candidate.

All models must use the same data splits and equivalent evaluation preprocessing. If a candidate cannot be trained due to hardware/runtime constraints, record it as not run with the reason; never invent results.

### Metrics and model-selection rule
Record per-model: test accuracy, macro precision, macro recall, macro F1, confusion matrix, model file size, average inference latency (ms/image), and end-to-end webcam FPS where possible. Prioritize recall for the `closed` class and acceptable false-alarm rates, but choose based on the complete validation profile rather than accuracy alone. Report confidence intervals or repeated-seed variability if practical.

The model comparison table must be populated with measured values only. No model is considered selected until it has been evaluated on the held-out test set and tested with real webcam images under varied lighting, glasses, face positions, and camera qualities.

### Software integration requirements
- Store the selected model and its preprocessing configuration under `models/`.
- Keep training and inference preprocessing identical (colour order, crop rules, resize, normalization).
- OpenCV must capture frames and produce eye crops; the CNN returns eye-state probabilities.
- Use a temporal state machine with configurable closure duration, smoothing, cooldown, and a `monitoring unavailable` state for poor/absent detections.
- Keep camera capture/UI responsive by running inference outside the UI thread or with a bounded frame queue.
- Log model version, threshold configuration, timestamps, and alert events; do not save raw video by default.
- Support CPU-only inference first. Benchmark on the target desktop and, separately, on Raspberry Pi before claiming edge performance.

### Implementation status rule
This document defines the required experiment, not completed experimental results. Training and comparison can only be honestly reported after the selected dataset is available in the runtime and the experiments have actually run. Do not write placeholder metrics as if measured.

## 5. Functional Requirements

### FR-01 — Camera discovery and selection
- Detect available camera devices where the operating system permits.
- Let the user select a camera source.
- Show a clear error if no camera is available or access is denied.
- Support start, stop, and retry operations.
- Release camera resources when monitoring stops or the application closes.

### FR-02 — Live camera monitoring
- Display a live video feed while monitoring is active.
- Optionally overlay face bounding boxes and facial landmarks.
- Show a visible camera-active state and elapsed monitoring time.
- Avoid implying recording is enabled unless video recording is actually active.
- Allow landmarks to be hidden to reduce visual clutter.

### FR-03 — Face and feature analysis
- Use OpenCV for frame capture, image preprocessing, and eye-region extraction.
- Run the selected dataset-trained CNN on eye crops to estimate `open` versus `closed` probability.
- Detect whether a face/eye region is visible; if eye extraction is unreliable, mark monitoring as unavailable rather than assuming eyes are open.
- Estimate eye openness using an appropriate landmark-based metric such as Eye Aspect Ratio (EAR), if supported by the chosen detector.
- Estimate mouth openness using a metric such as Mouth Aspect Ratio (MAR), where supported.
- Optionally estimate head pose or head nodding.
- Track measurements over time and reject low-confidence or invalid frames.
- Treat sunglasses, occlusion, extreme head angles, low light, and no-face frames as limitations or monitoring faults.

### FR-04 — Drowsiness estimation
- Combine temporal indicators rather than triggering on every ordinary blink.
- Use configurable thresholds and time windows.
- Include hysteresis or cooldown logic to prevent rapid alert switching.
- Display a monitoring-unavailable state when the face cannot be reliably assessed.
- Never present a single attention score as a medically validated measurement.

### FR-05 — Three-stage alert logic
- **Voice-alert design principle:** Use short, clear, firm voice alerts that get the driver's attention without creating unnecessary panic. Messages must be actionable, easy to understand, and brief enough not to distract from the road.
- Prefer offline prerecorded audio clips for consistent, low-latency safety messages; offline text-to-speech may be offered as a configurable fallback. Do not depend on internet connectivity for alerts.
- Avoid long explanations, blame, jokes, startling sound effects, or unnecessarily frightening language. Prevent overlapping voice messages and apply cooldown/repetition limits.
- **Stage 1 — Initial alert:** short on-screen warning and computer-generated beep when early credible indicators persist.
- **Stage 2 — High alert:** stronger repeated sound and a spoken warning if indicators continue or worsen.
- **Stage 3 — Critical alert:** prominent visual warning and a distinct local alarm pattern for severe or persistent indicators.
- Provide a preview/test-alert control that does not require a drowsiness event.
- Provide acknowledgement/snooze controls with safe limits. Acknowledgement must not automatically mark the driver as alert.
- Use these default voice prompts:
  - **Stage 1 — Early warning:** “Please stay alert. Keep your eyes on the road.”
  - **Stage 2 — Drowsiness warning:** “Warning! You may be getting drowsy. Please find a safe place to stop.”
  - **Stage 3 — Critical warning:** “Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.”
- Voice delivery should be calm at Stage 1, firm at Stage 2, and urgent but understandable at Stage 3. The volume and alarm pattern may escalate by stage, but should not be needlessly startling.
- Support optional voice-language selection (for example, English, Hindi, or Marathi) only when suitable voice clips are available and tested.
- Use neutral, actionable wording; never tell a drowsy driver to simply continue driving.
- Keep thresholds configurable for controlled testing; final values require validation.

### FR-06 — Driver metrics
Show the following where reliable:
- Attention/risk indicator with an explanation that it is an estimate.
- Eye openness (EAR or equivalent).
- Mouth openness (MAR or equivalent).
- Blink rate over a defined rolling time window.
- Optional head yaw/pitch estimates.
- Face visibility / detection confidence.
- Current alert stage.

If a metric is unavailable, show “—” or “Unavailable”; do not fabricate values.

### FR-07 — Event log
- Record application startup and shutdown.
- Record camera connected/disconnected and monitoring start/stop.
- Record alert-stage changes, camera faults, and recovery.
- Add timestamps and concise event descriptions.
- Keep logs local by default.
- Provide clear-log and export options, with confirmation before deleting records.
- Avoid storing raw video or face images by default.

### FR-08 — Configuration
- Camera source selector.
- Detection sensitivity control with explanatory labels.
- Toggle for facial landmarks.
- Video resolution / processing quality selector where supported.
- Alert sound and volume settings where supported by the OS.
- Optional language selection for voice warnings if implemented.
- Settings validation and safe defaults.
- Persist non-sensitive settings locally.

### FR-09 — Application status and recovery
- Show system status: Initializing, Ready, Monitoring, Alert, Camera Fault, or Error.
- Recover gracefully from camera disconnection.
- Prevent multiple concurrent camera loops.
- Log unexpected exceptions without exposing sensitive data.
- Provide a clear restart/retry action.

### FR-10 — Privacy
- Process frames locally by default.
- Do not upload video or biometric-like data without explicit user consent and a clear purpose.
- Explain what is logged and where logs are stored.
- Provide a way to clear locally stored events and settings.
- Avoid face identification; the purpose is state estimation, not identity recognition.

## 6. UI/UX Requirements

The supplied reference wireframe is the visual baseline. The interface should be a calm, professional desktop dashboard with restrained colors rather than a highly saturated “AI neon” aesthetic.

### 6.1 Visual direction
- Light theme with white cards, pale gray page background, and subtle borders.
- Navy or dark slate typography for readability.
- Muted blue as the primary accent.
- Green for normal/ready states, amber for warnings, and red for critical alerts only.
- Avoid excessive gradients, glowing effects, glassmorphism, and decorative animations.
- Use a consistent sans-serif typeface, consistent spacing, rounded corners, and restrained shadows.
- Maintain strong contrast and readable font sizes.

### 6.2 Header
- Product name: **DrishtiDrive AI**.
- Tagline: **A Watchful Eye on Every Journey.**
- Simple eye/road-inspired logo mark or placeholder.
- Navigation links may include Dashboard, Analytics, Settings, and Help.
- System state indicator: Online / Initializing / Error.
- The dashboard must not require a left navigation sidebar; use a compact top navigation/header as shown in the revised reference direction.

### 6.3 Main dashboard layout
Use a responsive card grid:
- Main/left area: Live Driver Monitoring and Camera Controls.
- Right area: Driver Metrics, Event Log, and status summary.
- Alert System card: three stages, visible near the camera controls.
- Keep the live camera feed the most prominent element.
- On narrower screens, stack cards vertically without clipping controls.

### 6.4 Live Driver Monitoring card
- Heading: “Live Driver Monitoring”.
- Camera preview with Start Camera / Stop Camera control.
- Show Ready, Camera Active, or Camera Error status.
- Overlay face/eye landmarks only when enabled.
- Optional small telemetry strip: EAR, MAR, head pose, attention/risk estimate, FPS and resolution.
- Use a clear camera permission/error state rather than an empty gray box.

### 6.5 Driver Status and Metrics
- Status label: Normal, Stage 1, Stage 2, Stage 3, or Monitoring Unavailable.
- Attention/risk indicator should be labelled as an estimate and must not imply validated medical accuracy.
- Metrics: Eye Openness, Mouth Openness, Blink Rate, optional Head Pose.
- Use simple progress bars only when the metric has a meaningful range.
- Use em dashes for missing values.

### 6.6 Camera Controls card
- Camera source dropdown.
- Detection sensitivity slider with “More Sensitive” and “Less Sensitive” labels.
- “Show Facial Landmarks” toggle.
- Resolution selector if supported.
- Controls must have visible focus states and clear labels.

### 6.7 Alert System card
- Voice-alert principle: **short, clear, firm, and attention-getting without creating unnecessary panic**.
- Stage 1 sample voice: “Please stay alert. Keep your eyes on the road.”
- Stage 2 sample voice: “Warning! You may be getting drowsy. Please find a safe place to stop.”
- Stage 3 sample voice: “Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.”
- Prefer local prerecorded clips for consistent playback; optional offline text-to-speech may be used as a fallback.
- Avoid overlapping messages, excessive repetition, and startling audio effects.
- Three visually distinct but restrained cards:
  - Stage 1 — Initial Alert / Buzzer or short system beep.
  - Stage 2 — High Alert / Repeating sound and voice warning.
  - Stage 3 — Critical Alert / Distinct local alarm and prominent warning.
- Include a Preview Alert button.
- Clearly distinguish current Active stage from Ready/inactive stages.
- Do not label the software-only edition as controlling a vehicle horn or parking lights.

### 6.8 Event Log card
- Chronological timestamped entries.
- Use small status icons plus readable text.
- Include clear-log confirmation and export option.
- Scroll independently if the list grows.

### 6.9 Accessibility and usability
- Do not rely on color alone to convey severity; include text and icons.
- Provide keyboard-accessible controls and visible focus indicators.
- Avoid flashing or rapidly moving UI elements.
- Use clear error messages and simple recovery instructions.
- Do not show fake camera data or simulated metrics as live values without a clear Demo Mode label.

## 7. Non-Functional Requirements

- **Performance:** Target smooth monitoring on the supported development machine; measure actual FPS and end-to-end alert latency.
- **Responsiveness:** The UI must remain usable while inference runs; use a worker/thread or process for camera/inference tasks.
- **Reliability:** Camera and audio resources must be released on stop, error, and application exit.
- **Maintainability:** Separate capture, detection, decision logic, alerts, UI, and logging modules.
- **Portability:** Document supported OS and Python versions; prioritize Windows and Linux if feasible.
- **Privacy:** Local processing and minimal retention by default.
- **Safety:** Clear monitoring faults; no claims of guaranteed prevention or certified safety performance.

## 8. Suggested Technical Architecture

### 8.1 Stack
- Python 3.x.
- OpenCV for image capture and image processing.
- MediaPipe Face Landmarker or a compatible landmark detector.
- Optional ONNX Runtime / TensorFlow Lite if a trained model is added.
- PySide6, Tkinter, or a local web UI (choose one UI framework and keep it consistent).
- SQLite or rotating JSON logs.
- Local audio playback library compatible with the target OS.

### 8.2 Module boundaries
- `camera_service`: discovery, frame capture, lifecycle and camera faults.
- `face_analyzer`: face landmarks, eye/mouth metrics and optional head pose.
- `drowsiness_engine`: temporal smoothing, confidence and risk state.
- `alert_manager`: stage transitions, sound playback and acknowledgement.
- `event_logger`: timestamps and local events.
- `settings_service`: validated local settings.
- `dashboard_ui`: live preview, metrics, controls, status and event log.

### 8.3 Data flow
1. Camera service produces frames.
2. Face analyzer extracts landmarks and measurements.
3. Drowsiness engine evaluates a rolling window of measurements.
4. Alert manager decides the active stage.
5. UI displays current status and metrics.
6. Event logger records state changes and faults.
7. No frame leaves the machine unless a future opt-in feature is explicitly added.

## 9. Suggested Repository Structure

```text
drishtidrive-software/
├── README.md
├── requirements.txt
├── app.py
├── config/
│   └── default_settings.yaml
├── camera/
│   ├── camera_service.py
│   └── camera_health.py
├── detection/
│   ├── face_analyzer.py
│   ├── eye_metrics.py
│   ├── mouth_metrics.py
│   └── head_pose.py
├── engine/
│   ├── drowsiness_engine.py
│   └── state_machine.py
├── alerts/
│   ├── alert_manager.py
│   └── audio_service.py
├── logging_service/
│   └── event_logger.py
├── settings/
│   └── settings_service.py
├── ui/
│   ├── dashboard.py
│   ├── camera_panel.py
│   ├── metrics_panel.py
│   ├── alert_panel.py
│   └── event_log_panel.py
├── tests/
│   ├── test_state_machine.py
│   ├── test_metrics.py
│   └── test_camera_faults.py
└── docs/
    └── testing_plan.md
```

## 10. Acceptance Criteria

- User can select and start an available camera.
- Camera frames appear in the monitoring panel and stop cleanly.
- Face/eye analysis can report unavailable states when confidence is insufficient.
- A test sequence can trigger each of the three alert stages deterministically.
- Ordinary short blinks do not automatically trigger critical alerts.
- Camera loss is shown clearly and logged.
- Metrics and event log update without freezing the UI.
- Settings persist across restarts where applicable.
- Local processing works without an internet connection after dependencies/models are installed.
- No vehicle electrical output is controlled by this edition.
- UI follows the defined restrained palette and layout requirements.

## 11. Testing Plan

- Unit tests for eye/mouth metric calculations and alert state transitions.
- Recorded-video tests covering normal blinking, prolonged eye closure, yawning, head nodding, occlusion and no-face conditions.
- Camera permission denied, camera disconnected and camera busy tests.
- Audio unavailable and audio-device change tests.
- Verify all three voice prompts are short, clear, audible, actionable, and do not overlap; test the available languages and alert cooldown behavior.
- UI resize and low-resolution tests.
- Performance measurement across supported computers.
- Privacy test confirming there is no unapproved network upload.
- Controlled user evaluation; never ask participants to deliberately fall asleep while driving.

### Model experiment acceptance criteria
- At least four candidate models are trained where resources permit; target five as listed in Section 4A.
- All candidates use identical subject-independent splits and the same held-out test set.
- A comparison report includes metrics, confusion matrices, latency, and model size.
- The chosen model is integrated into the OpenCV webcam pipeline and is switchable/configurable without changing UI code.
- Test cases cover glasses, different face/eye appearances, bright/dim lighting, camera motion, no face, multiple faces, and camera disconnection.
- Acceptance thresholds must be set after baseline results; do not claim production safety from dataset accuracy alone.

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| False positives | Temporal filtering, sensitivity configuration, representative testing |
| Missed drowsiness | Multiple indicators, explicit limitations, no safety guarantees |
| Poor lighting | Camera guidance, low-light tests, monitoring-unavailable state |
| Face occlusion / sunglasses | Confidence checks and clear unavailable status |
| UI freezes | Separate camera/inference worker from UI thread |
| Privacy concerns | Local processing and minimal event logging |
| Over-reliance | Warnings advise safe stopping and rest; software is not a substitute for sleep |

## 13. Future Enhancements

- Optional multilingual voice alerts.
- Analytics dashboard for event trends.
- Exportable CSV/PDF session report.
- Model benchmarking and calibration tools.
- Optional, consent-based fleet reporting with strict privacy controls.
- Packaging as a desktop installer.

## 14. Final Product Definition

DrishtiDrive AI — Software Edition is a local computer-vision application for experimentation and driver-alertness monitoring using an existing webcam. It includes a live dashboard, configurable three-stage software alerts, metrics, event logging, and clear fault handling. It does not control vehicle hardware and must not be represented as a certified safety device.
