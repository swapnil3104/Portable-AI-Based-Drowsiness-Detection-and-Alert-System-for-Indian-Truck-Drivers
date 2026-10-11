"""Prepare and extract balanced eye-region dataset from Dataset 2 for DrishtiDrive AI.

Extracts real visible-light driver eye crops from Dataset 2 (awake -> open, drowsy -> closed)
across train, valid, and test splits with strict class balance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from backend.config import DetectionConfig
from backend.eye_detection import PoseEyeDetector


def extract_crops_for_split(
    detector: PoseEyeDetector,
    images_dir: Path,
    labels_dir: Path,
    output_split_dir: Path,
    max_per_class: int,
    scale: float = 0.65,
) -> Dict[str, int]:
    """Extract balanced open and closed eye crops for a single split."""
    open_dir = output_split_dir / "open"
    closed_dir = output_split_dir / "closed"
    open_dir.mkdir(parents=True, exist_ok=True)
    closed_dir.mkdir(parents=True, exist_ok=True)

    counts = {"open": 0, "closed": 0}
    label_files = sorted(list(labels_dir.glob("*.txt")))
    random.shuffle(label_files)

    for lp in label_files:
        if counts["open"] >= max_per_class and counts["closed"] >= max_per_class:
            break

        lines = lp.read_text().strip().splitlines()
        if not lines:
            continue

        # Check ground truth class from label file
        classes = [int(l.split()[0]) for l in lines if l.strip()]
        if not classes:
            continue
        # 0 = awake (open eyes), 1 = drowsy (closed eyes)
        is_awake = (0 in classes and 1 not in classes)
        is_drowsy = (1 in classes and 0 not in classes)
        if not is_awake and not is_drowsy:
            continue

        target_class = "open" if is_awake else "closed"
        if counts[target_class] >= max_per_class:
            continue

        img_path = images_dir / (lp.stem + ".jpg")
        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None or frame.size == 0:
            continue

        ih, iw = frame.shape[:2]
        extracted_crops = []

        # Try YOLO Pose keypoint extraction with tight crop scale
        results = detector.model.predict(frame, verbose=False)
        if results and len(results[0].boxes) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy[0].cpu().numpy()
            confs = (
                results[0].keypoints.conf[0].cpu().numpy()
                if results[0].keypoints.conf is not None
                else np.ones(len(kpts))
            )
            left_pt, left_conf = kpts[1], float(confs[1])
            right_pt, right_conf = kpts[2], float(confs[2])

            if left_conf >= 0.20 and right_conf >= 0.20 and left_pt[0] > 0 and right_pt[0] > 0:
                iod = np.hypot(right_pt[0] - left_pt[0], right_pt[1] - left_pt[1])
                box_size = max(24, int(iod * scale))
                half = box_size // 2

                # Left crop
                lx, ly = int(left_pt[0]), int(left_pt[1])
                lc = frame[max(0, ly - half) : min(ih, ly + half), max(0, lx - half) : min(iw, lx + half)]
                if lc.shape[0] >= 16 and lc.shape[1] >= 16:
                    extracted_crops.append(lc)

                # Right crop
                rx, ry = int(right_pt[0]), int(right_pt[1])
                rc = frame[max(0, ry - half) : min(ih, ry + half), max(0, rx - half) : min(iw, rx + half)]
                if rc.shape[0] >= 16 and rc.shape[1] >= 16:
                    extracted_crops.append(rc)

        # Fallback to eye region inside the annotated face bounding box if pose missed
        if not extracted_crops:
            parts = lines[0].split()
            if len(parts) >= 5:
                _, cx, cy, bw, bh = map(float, parts[:5])
                fx1 = max(0, int((cx - bw / 2.0) * iw))
                fy1 = max(0, int((cy - bh / 2.0) * ih))
                fx2 = min(iw, int((cx + bw / 2.0) * iw))
                fy2 = min(ih, int((cy + bh / 2.0) * ih))
                face_w, face_h = fx2 - fx1, fy2 - fy1
                if face_w >= 30 and face_h >= 30:
                    # Eye region is approximately top 20% to 55% of face height
                    ey1 = fy1 + int(face_h * 0.20)
                    ey2 = fy1 + int(face_h * 0.55)
                    # Left third and right third
                    ex1_l = fx1 + int(face_w * 0.10)
                    ex2_l = fx1 + int(face_w * 0.50)
                    ex1_r = fx1 + int(face_w * 0.50)
                    ex2_r = fx1 + int(face_w * 0.90)
                    c_left = frame[ey1:ey2, ex1_l:ex2_l]
                    c_right = frame[ey1:ey2, ex1_r:ex2_r]
                    if c_left.shape[0] >= 12 and c_left.shape[1] >= 12:
                        extracted_crops.append(c_left)
                    if c_right.shape[0] >= 12 and c_right.shape[1] >= 12:
                        extracted_crops.append(c_right)

        target_dir = open_dir if target_class == "open" else closed_dir
        for idx, crop in enumerate(extracted_crops):
            if counts[target_class] >= max_per_class:
                break
            out_name = f"{lp.stem}_crop{idx}.jpg"
            cv2.imwrite(str(target_dir / out_name), crop)
            counts[target_class] += 1

    return counts


def main() -> None:
    """Extract and split Dataset 2 eye images."""
    parser = argparse.ArgumentParser(description="Extract Dataset 2 eye crops")
    parser.add_argument("--dataset2-dir", default="dataset/Dataset 2", help="Root of Dataset 2")
    parser.add_argument("--output-dir", default="data/dataset2_eye_state", help="Output directory")
    parser.add_argument("--train-per-class", type=int, default=1500, help="Train samples per class")
    parser.add_argument("--val-per-class", type=int, default=300, help="Val samples per class")
    parser.add_argument("--test-per-class", type=int, default=300, help="Test samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    d2_root = Path(args.dataset2_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = DetectionConfig(allow_multiple_faces=True, min_keypoint_confidence=0.20)
    detector = PoseEyeDetector(config=cfg)

    print("=" * 60)
    print("Extracting Dataset 2 Eye Crops (Train, Val, Test)")
    print(f"Target: Train={args.train_per_class*2}, Val={args.val_per_class*2}, Test={args.test_per_class*2}")
    print("=" * 60)

    splits = [
        ("train", d2_root / "train" / "images", d2_root / "train" / "labels", args.train_per_class),
        ("val", d2_root / "valid" / "images", d2_root / "valid" / "labels", args.val_per_class),
        ("test", d2_root / "test" / "images", d2_root / "test" / "labels", args.test_per_class),
    ]

    manifest = {"splits": {}}
    for split_name, img_dir, lbl_dir, count in splits:
        t0 = time.perf_counter()
        print(f"\nProcessing split '{split_name}' from {img_dir}...")
        split_out = out_root / split_name
        res = extract_crops_for_split(detector, img_dir, lbl_dir, split_out, count)
        manifest["splits"][split_name] = res
        dt = time.perf_counter() - t0
        print(f"Split '{split_name}' complete in {dt:.1f}s: Open={res['open']}, Closed={res['closed']} (Total: {sum(res.values())})")

    manifest_path = out_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDataset 2 eye extraction complete! Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
