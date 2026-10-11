# DrishtiDrive AI — Dataset Selection and Preparation Guide

**Document:** `dataset_guide.md`  
**Version:** 1.0  
**Project:** DrishtiDrive AI — Software Edition  

---

## 1. Candidate Datasets Review

### 1.1 MRL Eye Dataset (Primary Candidate)
- **Source:** [MRL Eye Dataset — VSB Technical University of Ostrava](https://mrl.cs.vsb.cz/eyedataset)
- **Description:** Large-scale dataset of eye crops collected from 37 subjects under varied lighting conditions, with and without eyeglasses, and varying camera reflections.
- **Labels:** Binary `0` (closed) and `1` (open), with metadata encoding subject ID, glasses, light conditions, and reflection.
- **Filename convention:** `s0001_00001_0_0_0_0_0_01.png`
  - `s0001`: Subject identifier
  - 3rd field: Eye state (`0` = closed, `1` = open)
  - 4th field: Eyeglasses (`0` = none, `1` = glasses present)
- **License / Terms:** Free for research and non-commercial educational evaluation with attribution.
- **Suitability:** Primary choice for training CNN eye-state classifiers.

### 1.2 Kaggle Eye State Datasets (Alternative / Supplementary)
- **Source:** [Kaggle Closed vs Open Eyes Search](https://www.kaggle.com/search?q=closed+eyes+open+eyes+dataset)
- **Common subsets:** CEW (Closed Eyes in the Wild) dataset, yawning/eye datasets.
- **Permitted use:** Check individual dataset license (CC BY-NC, CC BY 4.0, or Public Domain).
- **Caution:** Verify image quality, avoid duplicates, and confirm that labels are verified.

### 1.3 NTHU Drowsy Driver Detection Dataset (Sequence Evaluation)
- **Source:** [NTHU DDD Dataset](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)
- **Description:** Video sequence dataset capturing realistic driver fatigue behaviors: blinking, prolonged closures, yawning, nodding, and looking away under varied driving simulation conditions.
- **Access condition:** Requires academic registration / agreement.
- **Use case:** Validation of temporal state machine and sequence-level threshold verification.

---

## 2. Dataset Splitting & Leakage Prevention Strategy

To prevent data leakage and guarantee that models generalize well to new drivers:
1. **Subject-Independent Splits:** All images originating from the same individual/session must strictly belong to either the training, validation, or test set.
2. **Target Ratio:**
   - **Training Set:** 70% of subjects
   - **Validation Set:** 15% of subjects
   - **Test Set (Held-Out):** 15% of subjects
3. **Reproducibility:** Seed fixed at `42`, with a `split_manifest.json` recording every image's assigned partition and subject ID.

---

## 3. Preprocessing Specifications

- **Input dimensions:** 96 × 96 pixels.
- **Color format:** RGB, normalized to range `[0.0, 1.0]`.
- **Training-Only Augmentations:**
  - Random horizontal flip (preserving natural bilateral eye shape).
  - Subtle brightness & contrast adjustment (±15%).
  - Rotation limited to ±8 degrees (avoiding unnatural head rolls).
- **Label Encoding:** `0` = open, `1` = closed.
