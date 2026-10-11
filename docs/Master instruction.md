# DrishtiDrive AI — Temporal Scoring & Alert Logic

## 1. Time-Smoothed Drowsiness & Attention Scoring Engine
- **Engine Module:** [`backend/scoring.py`](file:///e:/Portable-AI-Based-Drowsiness-Detection-and-Alert-System-for-Indian-Truck-Drivers/backend/scoring.py)
- **Measurement Inputs:**
  - PERCLOS (% of eye closure over 60-second rolling window)
  - Exponentially smoothed Eye Aspect Ratio (EAR)
  - Sustained closure duration penalty
  - Rolling Mouth Aspect Ratio (MAR) and yawn accumulation
  - Blink dynamics (blinks per minute)
- **Filtering:** Continuous time-decay exponential smoothing filter ($\alpha = 1 - e^{-dt / \tau}$, $\tau = 1.5\text{s}$), ensuring a single video frame never collapses the driver's attention score or triggers false alerts.
- **Calibration Notice:** Calibrated against labelled eye datasets and synthetic driving sequences. These weights are initial engineering values, not clinically validated measurements.

---

## 2. Three-Stage Alert Escalation Logic
Implemented in [`backend/temporal_machine.py`](file:///e:/Portable-AI-Based-Drowsiness-Detection-and-Alert-System-for-Indian-Truck-Drivers/backend/temporal_machine.py) and configured in [`backend/config.py`](file:///e:/Portable-AI-Based-Drowsiness-Detection-and-Alert-System-for-Indian-Truck-Drivers/backend/config.py):

| Alert Stage | Trigger Condition | Spoken Voice Message | Visual Directive |
| :--- | :--- | :--- | :--- |
| **Stage 1 — Attention Warning** | Score $\le 75$ for $60$ seconds | *"Please stay alert. Keep your eyes on the road."* | Gentle Attention Beep + Gold Banner |
| **Stage 2 — Drowsiness Warning** | Score $\le 60$ for $60$ seconds | *"Warning! You may be getting drowsy. Please find a safe place to stop."* | Firm Repeating Beep + Orange Banner |
| **Stage 3 — Critical Warning** | Score $\le 30$ for $30$ seconds (or microsleep $>3.0\text{s}$) | *"Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."* | High-Urgency Alarm + Red Perimeter |

---

## 3. Verification & Test Coverage
- Dedicated unit suite: [`tests/test_scoring.py`](file:///e:/Portable-AI-Based-Drowsiness-Detection-and-Alert-System-for-Indian-Truck-Drivers/tests/test_scoring.py)
- Full test suite: **94 passed** across all integration, pipeline, audio, and API tests.
