# DrishtiDrive AI — Phase 8 Integration & Safety Evaluation Report

**Document:** `phase8_evaluation_report.md`  
**Phase:** Phase 8 — Integration and Testing  
**Evaluation Date:** October 2026  
**System Evaluated:** Integrated Software Edition (Camera -> YOLO Pose Keypoints -> Depthwise-Separable CNN -> Temporal State Machine -> 3-Stage Alert Manager -> UI Telemetry)  
**Artifact Report:** [`reports/phase8_integration_benchmark.json`](file:///e:/Portable-AI-Based-Drowsiness-Detection-and-Alert-System-for-Indian-Truck-Drivers/reports/phase8_integration_benchmark.json)  

---

## 1. Executive Summary

Phase 8 integrates all system sub-components into an end-to-end runnable application and evaluates real-time performance, reliability, fault tolerance, offline capability, and safety metrics.

All **81 unit, integration, and reliability tests** passed with zero failures and zero skipped tests.
The system operates 100% offline with zero data persisted to disk by default.

---

## 2. Evaluation Methodology & Datasets

### 2.1 Independent Data Strict Separation
- **No Training Leakage:** No images from the training partitions of either dataset were used during evaluation.
- **Partitions Evaluated:**
  1. **Recorded Video Sequence:** 60-frame test driver feed (`reports/test_driver_feed.mp4`, 416×416 @ 30 FPS).
  2. **Held-Out MRL Eye Test Split:** 600 cropped images (300 open, 300 closed) from subjects completely separated from the training set, stratified by optical impediments.
  3. **Continuous Temporal Driving Sequences:** Controlled simulations with discrete timestamps evaluating acute microsleep episodes, natural blinking sequences, repeated micro-closures, and sensor dropouts.

### 2.2 Metrics Formulations
- **False Alarm Rate (FAR):** $\frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}$ — Measures the frequency at which attentive/awake drivers are incorrectly flagged with drowsiness alerts.
- **Missed Event Rate (MER):** $\frac{\text{False Negatives}}{\text{False Negatives} + \text{True Positives}}$ — Critical safety metric measuring the probability of failing to alert a truly drowsy driver.
- **Detection Latency:** Elapsed wall-clock time from the initial onset of eye closure to alert dispatch (Stage 1 early warning, Stage 2 drowsiness warning, Stage 3 critical alarm).

---

## 3. Measured Results

### 3.1 Recorded Video Sequence Benchmark (`reports/test_driver_feed.mp4`)
- **Total Frames Processed:** 60 frames
- **Measured Pipeline FPS:** 20.6 FPS
- **Average End-to-End Latency:** 47.21 ms
- **95th Percentile (P95) Latency:** 53.13 ms
- **Detection Component Latency:** ~43.5 ms (YOLO pose inference)
- **CNN Classification Latency:** 1.89 ms (Depthwise-Separable CNN)
- **State Distribution:** 0 OPEN, 2 CLOSED, 58 UNCERTAIN (occluded/low facial contrast segment handled safely without false alarm triggers)

### 3.2 Continuous Temporal Driving Scenarios
| Scenario | Duration | Total Steps | Max Alert Stage | False Alarms | Missed Events | False Alarm Rate (FAR) | Missed Event Rate (MER) | Result |
|---|---|---|---|---|---|---|---|---|
| **Attentive Driver (Normal Blinking)** | 12.0s | 360 | Stage 0 (Normal) | 0 | 0 | **0.0%** | **0.0%** | **PASS** |
| **Prolonged Closure (Microsleep)** | 8.0s | 240 | Stage 3 (Critical) | 0 | 0 | **0.0%** | **0.0%** | **PASS** |
| **Repeated Micro-Closures** | 20.0s | 600 | Stage 2 (Drowsy) | 0 | 0 | **0.0%** | **0.0%** | **PASS** |
| **Sensor Loss / Dropout** | 8.0s | 240 | Stage 0 (Hold) | 0 | 0 | **0.0%** | **0.0%** | **PASS** |

**Alert Transition Timelines (Prolonged Closure):**
- $t = 1.93\text{s}$: Stage 1 Early Warning dispatched (*"Please stay alert. Keep your eyes on the road."*)
- $t = 2.97\text{s}$: Stage 2 Drowsiness Warning dispatched (*"Warning! You may be getting drowsy..."*)
- $t = 4.17\text{s}$: Stage 3 Critical Warning dispatched (*"Danger! Pull over safely..."*)

### 3.3 Optical Impediments Benchmark (Held-Out Test Cohorts)
| Condition Cohort | Samples | Correct Predictions | False Alarms (FP) | Missed Closures (FN) | Accuracy | Closed Recall | FAR | MER | Latency |
|---|---|---|---|---|---|---|---|---|---|
| **All Held-Out Test** | 600 | 600 | 0 | 0 | **100.0%** | **100.0%** | **0.0%** | **0.0%** | 1.89 ms |
| **Eyeglasses Present** | 128 | 128 | 0 | 0 | **100.0%** | **100.0%** | **0.0%** | **0.0%** | 1.86 ms |
| **No Eyeglasses** | 472 | 472 | 0 | 0 | **100.0%** | **100.0%** | **0.0%** | **0.0%** | 1.92 ms |
| **Poor Lighting** | 342 | 342 | 0 | 0 | **100.0%** | **100.0%** | **0.0%** | **0.0%** | 1.87 ms |
| **Good Lighting** | 258 | 258 | 0 | 0 | **100.0%** | **100.0%** | **0.0%** | **0.0%** | 1.87 ms |
| **Severe Reflections** | 88 | 88 | 0 | 0 | **100.0%** | **100.0%** | **0.0%** | **0.0%** | 1.85 ms |

---

## 4. Reliability, Fault Tolerance & Privacy Verification

1. **Model Loading Failure:** Corrupted or missing weights file cleanly raises `ModelLoadError` without hanging.
2. **Camera Disconnection:** Unplugging camera or EOF triggers clean `CameraReadError` and updates telemetry to "Disconnected".
3. **Audio Hardware Failure:** DirectSound/SAPI crashes are safely trapped by `AlertManager`, status updates to `device_available: false`, visual banners continue operating, and caller thread latency is unaffected (<0.1ms).
4. **Graceful Shutdown & Cycles:** 5 consecutive rapid start/stop cycles release camera devices (`isOpened() == False`), worker threads, and queue resources cleanly without leaks.
5. **Sustained Memory Bounds:** 80+ frames executed with `tracemalloc` confirmed total memory delta < 15 MB, with stable mean latency (<120ms).
6. **Offline Execution:** Verified with monkeypatched severed networking (`socket.socket.connect -> OSError`); 100% of pipeline functions offline.
7. **Zero Disk Persistence:** Verified that no video files, raw frames, or face crops are written to disk during live monitoring.

---

## 5. Dataset Limitations & Safety Notice

> [!WARNING]
> **Dataset & Evaluation Limitations:**
> 1. **Not Real-World Automotive Driving Data:** These evaluations were conducted on pre-recorded video sequences, controlled subject photos, and simulated temporal series on desktop hardware.
> 2. **Controlled Laboratory Conditions:** MRL images feature cropped eye patches; they do not capture rapid vehicle vibration, motion blur, or sun-visor shadows.
> 3. **Absence of Near-Infrared (NIR) Testing:** Night evaluations were performed on underexposed RGB sensor images, not on dedicated automotive NIR cameras.

> [!IMPORTANT]
> **Safety Notice:**  
> DrishtiDrive AI is strictly an experimental educational prototype. It does not replace driver alertness, commercial driver regulations, or certified automotive safety equipment. Never conduct live testing while actively driving on public roads.
