# DrishtiDrive AI — Software Edition
## Executive Summary

### 1. Overall Description
**DrishtiDrive AI — Software Edition** is a local, privacy-preserving computer-vision application designed to monitor visible indicators of driver fatigue and drowsiness using an existing desktop/laptop webcam. It integrates an OpenCV video capture and processing pipeline, facial landmark detection (EAR, MAR, head pose), and a deep learning CNN classifier trained on eye-state datasets (open vs. closed). The system implements a robust temporal state machine with configurable thresholds and cooldowns, driving an escalating 3-stage visual and voice alert system.

The software is intended for research, algorithm evaluation, and controlled demonstration without requiring vehicle hardware interfaces, CAN bus connections, or external cloud dependencies.

---

### 2. Main Features
1. **Camera Discovery & Live Stream Management**:
   - Automatic detection of local webcam devices with start/stop controls.
   - Resilient fault handling: detects disconnections, camera busy states, and permission denials with automatic recovery and graceful status reporting.
2. **Dual-Layer Vision & Drowsiness Analysis**:
   - **Facial Landmark Analysis**: High-precision Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR for yawning), and head pose estimation (yaw/pitch).
   - **CNN Eye-State Inference**: Classification of cropped eye regions using lightweight CNNs (MobileNetV2 / EfficientNetB0 / custom compact CNN).
   - **Temporal State Machine**: Rolling-window evaluation to prevent false triggers on regular blinks, supporting hysteresis and cooldowns.
3. **Escalating 3-Stage Software Alerts**:
   - **Stage 1 (Initial Alert)**: Visual prompt + subtle audible beep when initial closure indicators persist.
   - **Stage 2 (High Alert)**: Stronger repeated alert chime + clear voice warning (*"Warning! You may be getting drowsy. Please find a safe place to stop."*).
   - **Stage 3 (Critical Alert)**: Urgent visual alert + distinctive local alarm pattern (*"Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."*).
   - Preview alert feature & safe acknowledgment/snooze controls.
4. **Driver Telemetry & Metrics Dashboard**:
   - Real-time display of EAR, MAR, blink rate per minute, head pose yaw/pitch, FPS, camera resolution, and estimated attention/risk level.
5. **Local Event Logging & Settings**:
   - Structured local logging of camera lifecycle, stage transitions, and camera faults with clear/export capabilities.
   - Configurable sensitivity, threshold timings, voice alert settings, and persistent local configuration.

---

### 3. Key User Flows
1. **Monitoring Setup & Initiation**:
   - User opens the DrishtiDrive dashboard.
   - Selects camera source from dropdown, adjusts sensitivity slider, toggles landmark overlays, and clicks **Start Camera**.
   - Camera stream activates with live overlays and real-time telemetry strip.
2. **Real-Time Drowsiness Detection & Alert Escalation**:
   - Normal driving: Eye openness and blink rates tracked within normal bounds (`Normal` state).
   - Prolonged eye closure or frequent yawning triggers Stage 1 alert.
   - Continued closure elevates state to Stage 2 (voice prompt plays).
   - Persistent severe closure elevates to Stage 3 (urgent alarm). Driver acknowledgment silences the alert without resetting underlying risk state until indicators return to normal.
3. **Fault & Recovery Handling**:
   - If user covers the camera, wears opaque sunglasses, or faces away: state transitions cleanly to `Monitoring Unavailable` without false-positive alarms.
   - Camera unplugged or preempted: system transitions to `Camera Fault` and provides a one-click Retry action.

---

### 4. Key Requirements & Boundaries
- **Local & Offline Operation**: Zero cloud dependencies or external data uploads for core detection and alerts.
- **Responsive UI Threading**: Inference and video capture run asynchronously off the main UI loop to maintain smooth responsiveness.
- **No Vehicle Electrical Control**: Strictly software alerts on the host computer; no physical horn, lights, or vehicle actuation.
- **Safety Prototype Notice**: Displayed clearly as an experimental driver-assistance prototype, not a medical or certified safety device.
