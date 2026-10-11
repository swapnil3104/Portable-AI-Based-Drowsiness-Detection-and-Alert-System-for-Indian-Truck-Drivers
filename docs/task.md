# DrishtiDrive AI — Project Tasks

**Project:** DrishtiDrive AI  
**Tagline:** *A Watchful Eye on Every Journey*  
**Document:** `task.md`  
**Version:** 1.0  
**Last updated:** 11 October 2026  
**Primary scope:** Software Edition — OpenCV + dataset-trained CNN  
**Future scope:** Raspberry Pi Edge Edition

---

## 1. Project Goal

Build a desktop application that uses a webcam and computer vision to detect eye states, estimate possible driver drowsiness over time, and provide staged audio and visual warnings.

**Voice-alert principle:** Use short, clear, firm voice alerts that get the driver's attention without creating unnecessary panic.

This is an experimental prototype, not a certified vehicle-safety system. It must not control steering, braking, the vehicle horn, or exterior lights. It must not claim to diagnose fatigue or guarantee safety.

## 2. Task Status Legend

- `[ ]` Not started
- `[/]` In progress
- `[x]` Completed and verified

Do not mark a task complete until implementation or evidence has been checked. Model metrics must come from actual runs; never insert assumed results.

---

## 3. Phase 1 — Requirements and Repository Setup

- [x] **T1.1** Review and freeze the Software Edition requirements (`docs/rules.md`, `docs/design.md`).
- [x] **T1.2** Create the repository and baseline project structure (`backend/`, `frontend/`, `tests/`, `docs/`).
- [x] **T1.3** Add `README.md` with project purpose, setup, usage, limitations, and safety notice.
- [x] **T1.4** Add `requirements.txt` or a managed environment file with compatible dependencies.
- [x] **T1.5** Add `.gitignore` for virtual environments, caches, logs, datasets, and generated models.
- [x] **T1.6** Add configuration for camera index, model path, thresholds, alert timings, and logging (`backend/config.py`).
- [x] **T1.7** Record environment details and supported operating systems (`README.md`).

**Acceptance criteria:** A fresh environment can be set up from documentation; no credentials or private driver data are committed; dataset and model licences are documented.

## 4. Phase 2 — Dataset Selection and Preparation

- [x] **T2.1** Review the MRL Eye Dataset as the primary candidate (`docs/dataset_guide.md`).
- [x] **T2.2** Review Kaggle eye-state datasets as alternatives or supplements (`docs/dataset_guide.md`).
- [x] **T2.3** Review the NTHU Drowsy Driver Detection Dataset for later sequence-level evaluation (`docs/dataset_guide.md`).
- [x] **T2.4** Verify access conditions, licence, labels, class counts, and permitted use (`LICENSE`, `docs/dataset_guide.md`).
- [x] **T2.5** Build a data-audit script for image counts, class distribution, unreadable files, dimensions, and duplicates (`backend/training/data_audit.py`, `reports/data_audit_report.json`).
- [x] **T2.6** Standardize labels as `open` and `closed`, preserving source metadata as needed (`backend/training/prepare_dataset.py`).
- [x] **T2.7** Create train, validation, and test splits with subject/session separation (`data/eye_state` train 2800, val 600, test 600).
- [x] **T2.8** Prevent leakage by keeping near-duplicates and frames from the same person/session in one split (`backend/training/prepare_dataset.py`).
- [x] **T2.9** Define preprocessing: color format, resizing, normalization, and training-only augmentation (`backend/preprocessing.py`, `backend/training/dataset.py`).
- [x] **T2.10** Address class imbalance with justified class weights or sampling (`reports/data_audit_report.json`, balanced 50/50).
- [x] **T2.11** Save split manifests and random seeds for reproducibility (`prepare_dataset.py`, `data_audit_report.json`).

**Acceptance criteria:** Dataset sources, licences, labels, class counts, and split strategy are documented. Test data is not used for tuning.

## 5. Phase 3 — Train and Compare Five CNN Candidates

Train all candidates on the same documented splits and comparable input settings.

- [x] **T3.1** Train a small **Custom CNN** baseline (86.83% test acc, 97.0% closed-eye recall, 1.66ms CPU latency, 1.19MB).
- [x] **T3.2** Train **MobileNetV2** as a lightweight real-time candidate (98.17% test acc, 97.33% closed-eye recall, 8.8ms CPU latency, 9.66MB).
- [x] **T3.3** Train **EfficientNetB0** for an accuracy/efficiency comparison (98.33% test acc, 98.33% closed-eye recall, 13.11ms CPU latency, 16.52MB).
- [x] **T3.4** Train **ResNet50** as a heavier comparison candidate (100.0% test acc, 100.0% closed-eye recall, 25.11ms CPU latency, 91.48MB).
- [x] **T3.5** Train a **Depthwise-separable CNN** as a lightweight candidate (100.0% test acc, 100.0% closed-eye recall, 1.93ms CPU latency, 0.38MB — Selected Winner).
- [x] **T3.6** Use consistent preprocessing and comparable training/validation protocols.
- [x] **T3.7** Save each best validation checkpoint and training history (`backend/models/*_best.pt`).
- [x] **T3.8** Track configuration, seed, runtime, and dependency versions.
- [x] **T3.9** Evaluate candidates on the held-out test set after model selection.
- [x] **T3.10** Generate a comparison table and confusion matrix for each candidate (`reports/model_comparison.csv`, `reports/*_confusion_matrix.png`).
- [x] **T3.11** Record test accuracy, macro precision, macro recall, macro F1, per-class recall, and closed-eye recall.
- [x] **T3.12** Measure model file size and inference latency on the target development machine.
- [x] **T3.13** Inspect false positives/negatives and document limitations.
- [x] **T3.14** Select a candidate using safety-relevant metrics, false-alarm behavior, latency, and size—not accuracy alone (Selected: Depthwise-Separable CNN).
- [x] **T3.15** Save a model card with dataset, intended use, limitations, and evaluation results (`docs/model_card.md`, `docs/MODEL_RESULT.md`).

**Acceptance criteria:** All five candidates have actual reproducible results or a documented reason a run could not complete. Never fabricate scores or claim a winning model without evidence.

## 6. Phase 4 — OpenCV Camera and Eye-Region Pipeline

- [x] **T4.1** Implement camera initialization and configurable camera source (`backend/camera.py`).
- [x] **T4.2** Add start/stop controls and graceful camera release (`backend/camera.py`).
- [x] **T4.3** Detect the face and locate eye regions using a documented detector or facial landmarks (`backend/eye_detection.py` using YOLO Pose keypoints).
- [x] **T4.4** Crop and preprocess eye regions consistently with CNN training (`backend/preprocessing.py`, verified with `torch.allclose` in `tests/test_preprocessing.py`).
- [x] **T4.5** Run inference and display `OPEN`, `CLOSED`, or `UNCERTAIN` with confidence (`backend/model_inference.py`, `backend/pipeline.py`).
- [x] **T4.6** Handle no-face, multiple-face, poor-frame, and low-confidence cases safely (`backend/eye_detection.py`, `tests/test_eye_detection.py`).
- [x] **T4.7** Do not treat a single closed-eye frame as a confirmed drowsiness event (`backend/pipeline.py`, verified in `tests/test_pipeline.py`).
- [x] **T4.8** Measure camera FPS and end-to-end inference latency (Actual: 18.6 FPS, 53.41ms total, 2.95ms inference; `reports/phase4_pipeline_benchmark.json`).
- [x] **T4.9** Test camera unavailable, permission denied, and repeated start/stop (`tests/test_camera.py`).

**Acceptance criteria:** Live webcam processing works reliably; inference preprocessing matches training; uncertain/missing detections are not silently treated as confident `OPEN` predictions.


## 7. Phase 5 — Temporal Drowsiness Logic

- [x] **T5.1** Define configurable thresholds for sustained closure and repeated events (`backend/config.py`, `TemporalConfig`).
- [x] **T5.2** Implement a time-based state machine rather than single-frame decisions (`backend/temporal_machine.py`, `TemporalDrowsinessDetector`).
- [x] **T5.3** Add smoothing/debouncing to reduce prediction flicker (`backend/temporal_debouncer.py`, `EyeStateDebouncer`).
- [x] **T5.4** Reset or pause timers when face/eye observations are unreliable (`backend/temporal_machine.py`, safe degradation hold without runaway escalation).
- [x] **T5.5** Prevent duplicate alerts while an alert is active (`backend/temporal_machine.py`, deduplication & `alert_cooldown_sec`).
- [x] **T5.6** Document that eye closure is an indicator, not definitive proof of drowsiness (`backend/temporal_machine.py`, `backend/config.py`).
- [x] **T5.7** Validate thresholds on held-out video sequences (`backend/temporal_validator.py`, `reports/phase5_temporal_benchmark.json`).

**Acceptance criteria:** Alerts follow documented temporal rules; poor detections are handled explicitly; thresholds are not described as medically validated.

## 8. Phase 6 — Three-Stage Audio and Visual Alerts

**Voice design principle:** Short, clear, firm, attention-getting, and not unnecessarily alarming.

- [x] **T6.1 — Stage 1, early warning:** gentle beep and: “Please stay alert. Keep your eyes on the road.” (`backend/alert_catalog.py`, `backend/alert_sounds.py`).
- [x] **T6.2 — Stage 2, drowsiness warning:** firmer repeating beep and: “Warning! You may be getting drowsy. Please find a safe place to stop.” (`backend/alert_catalog.py`, `backend/alert_sounds.py`).
- [x] **T6.3 — Stage 3, critical warning:** urgent but understandable alarm and: “Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.” (`backend/alert_catalog.py`, `backend/alert_sounds.py`).
- [x] **T6.4** Use prerecorded audio for predictable playback, or offline text-to-speech (e.g. `pyttsx3`) with a prerecorded fallback (`backend/audio_backend.py`).
- [x] **T6.5** Add optional English, Hindi, and Marathi support after default messages are validated (`backend/alert_catalog.py`).
- [x] **T6.6** Prevent speech overlap, excessive repetition, and alert queue buildup (`backend/alert_manager.py` bounded queue, cooldown & deduplication).
- [x] **T6.7** Add a manual audio test and configurable volume (`backend/alert_manager.py`, `AlertManager.set_volume`, `AlertManager.trigger_manual_test`).
- [x] **T6.8** Show a visual alert alongside audio where practical (`backend/alert_visual.py`, `render_alert_overlay`, `PipelineFrameResult.visual_alert`).
- [x] **T6.9** Test voice clarity and intelligibility at different speaker volumes (`tests/test_alert_manager.py`, `backend/validate_alerts.py`).
- [x] **T6.10** Keep core audio operation local/offline (`backend/alert_sounds.py`, `backend/audio_backend.py`, 100% offline WAV & pyttsx3).

**Acceptance criteria:** Correct message and sound play for each stage; messages do not overlap or queue indefinitely; alerts instruct the driver to stop safely and avoid startling sounds.

## 9. Phase 7 — Software UI and Event Log

- [x] **T7.1** Show `DrishtiDrive AI` and `A Watchful Eye on Every Journey` in the header (`frontend/src/components/Header.jsx`).
- [x] **T7.2** Keep the live camera view dominant; do not add a left navigation sidebar (`frontend/src/components/LiveMonitoring.jsx`).
- [x] **T7.3** Add camera source, start/stop, resolution, and optional landmarks/overlay controls (`frontend/src/components/CameraControls.jsx`).
- [x] **T7.4** Add metrics for eye state, confidence, closure duration, FPS, and latency (`frontend/src/components/DriverMetrics.jsx`).
- [x] **T7.5** Label any risk indicator as an experimental estimate, not a medical or validated fatigue score (`frontend/src/components/DriverStatus.jsx`).
- [x] **T7.6** Add an alert panel showing current stage and audio status (`frontend/src/components/AlertSystem.jsx`).
- [x] **T7.7** Add an event log with timestamp, event type, stage, and relevant measurements (`frontend/src/components/EventLog.jsx`).
- [x] **T7.8** Do not save video/facial images by default; document retention settings if capture is later added (`frontend/src/components/CameraControls.jsx`).
- [x] **T7.9** Use restrained colors: light/neutral surfaces, navy/slate text, muted blue accent, and restrained green/amber/red status colors (`frontend/src/index.css`).
- [x] **T7.10** Show camera/model/system status and useful error messages (`frontend/src/components/Header.jsx`, `frontend/src/components/AlertSystem.jsx`).
- [x] **T7.11** Keep the UI responsive while inference and audio run (`backend/app.py` async pipeline and non-blocking streaming).
- [x] **T7.12** Provide readable text and accessible controls (`frontend/src/components/`).

**Acceptance criteria:** Camera feed, metrics, event log, and alert state are clear; UI remains responsive; privacy behavior is explicit and safe by default.

## 10. Phase 8 — Integration and Testing

- [x] **T8.1** Integrate camera, eye detector, CNN inference, temporal logic, alert manager, and UI (`backend/pipeline.py`, `backend/app.py`, `run.py`).
- [x] **T8.2** Add unit tests for preprocessing, output labels, state transitions, and alert escalation (`tests/test_preprocessing.py`, `tests/test_model_inference.py`, `tests/test_temporal_logic.py`).
- [x] **T8.3** Add integration tests using recorded clips and simulated camera frames (`tests/test_integration_pipeline.py`, `reports/test_driver_feed.mp4`).
- [x] **T8.4** Test blinking, prolonged closure, glasses, varied lighting, camera angles, and head movement (`tests/test_integration_pipeline.py`).
- [x] **T8.5** Test no-face, occlusion, model-load failure, camera disconnection, and audio-device failure (`tests/test_fault_tolerance.py`).
- [x] **T8.6** Measure false-alarm and missed-event rates on independent video sequences (`backend/evaluate_integration.py`, `reports/phase8_integration_benchmark.json`).
- [x] **T8.7** Test sustained runtime for memory leaks, overheating, and UI responsiveness (`tests/test_reliability_privacy_offline.py`, tracemalloc bounded memory < 15MB).
- [x] **T8.8** Verify core inference and alerts work without internet (`tests/test_reliability_privacy_offline.py`, 100% offline verified with network blocking).
- [x] **T8.9** Document known limitations and unresolved failure modes (`docs/model_card.md`, `README.md`, `reports/phase8_integration_benchmark.json`).

## 11. Phase 9 — Packaging and Documentation

- [x] **T9.1** Provide reproducible environment setup and launch instructions (`README.md`, `run.py`).
- [x] **T9.2** Document dataset download/extraction without committing restricted data (`docs/dataset_guide.md`).
- [x] **T9.3** Document how to train all five candidates and reproduce comparisons (`README.md`, `backend/training/train_compare.py`).
- [x] **T9.4** Document how to select and load the winning model (`README.md`, `backend/config.py`).
- [x] **T9.5** Add troubleshooting for camera permissions, model files, and audio devices (`README.md`).
- [x] **T9.6** Include measured model comparison results and test configuration (`reports/model_comparison.csv`, `README.md`).
- [x] **T9.7** Include privacy, safety, and limitations notices (`README.md`, `docs/rules.md`).
- [x] **T9.8** Verify redistribution/licensing terms before publishing models or datasets (`LICENSE`, `docs/dataset_guide.md`).

## 12. Phase 10 — Future Raspberry Pi Edge Edition

- [ ] **T10.1** Benchmark the selected model on the intended Raspberry Pi hardware.
- [ ] **T10.2** Measure FPS, latency, CPU/RAM usage, and temperature.
- [ ] **T10.3** Evaluate conversion to a compatible lightweight runtime if needed.
- [ ] **T10.4** Bench-test camera, speaker, buzzer, and indicator interfaces.
- [ ] **T10.5** Design protected vehicle power input with suitable DC-DC conversion.
- [ ] **T10.6** Keep the prototype isolated from vehicle horn, braking, steering, and exterior-light circuits.
- [ ] **T10.7** Repeat validation on actual target hardware before considering road use.

## 13. Model Comparison Results Template

Measured on CPU development machine (Intel/AMD x64, PyTorch 2.14.1+cpu):

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Closed-eye recall | Inference latency | Model size |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Custom CNN** | 86.83% | 88.42% | 86.83% | 86.70% | 97.00% | 1.66 ms | 1.19 MB |
| **Depthwise-separable CNN** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.93 ms** | **0.38 MB** |
| **MobileNetV2** | 98.17% | 98.18% | 98.17% | 98.17% | 97.33% | 8.80 ms | 9.66 MB |
| **EfficientNetB0** | 98.33% | 98.33% | 98.33% | 98.33% | 98.33% | 13.11 ms | 16.52 MB |
| **ResNet50** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 25.11 ms | 91.48 MB |

**Selection decision:** **Depthwise-separable CNN** is selected as the winning architecture. It achieves 100% closed-eye recall (0 false negatives), 0 false alarms (0 false positives), 1.93ms CPU latency (>500 FPS throughput capability), and an ultra-compact size of only 0.38 MB (29,522 parameters). MobileNetV2 serves as an alternative pretrained transfer learning candidate.

## 14. Definition of Done

- [x] Dataset sources, licences, labels, and split strategy are documented.
- [x] Five candidate models have been trained and compared, or incomplete runs are explicitly explained.
- [x] The selected model has reproducible evaluation evidence.
- [x] OpenCV live-camera inference is integrated.
- [x] Temporal logic drives all three alert stages.
- [x] Short, clear, firm voice alerts are implemented without unnecessary panic.
- [x] UI displays camera status, model output, alert stage, and event history.
- [x] Tests cover blinking, sustained closure, poor visibility, and component failures.
- [x] Offline operation and privacy defaults are verified.
- [x] Documentation explains limitations and advises the driver to stop safely and rest when drowsy.

**Safety note:** This project is experimental. It cannot guarantee drowsiness detection and must not replace adequate rest, safe driving decisions, or certified safety systems. Never conduct development testing while actively driving on public roads.
