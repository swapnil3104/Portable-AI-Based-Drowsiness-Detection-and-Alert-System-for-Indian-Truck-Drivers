"""Dataset Audit Script for DrishtiDrive AI.

Scans dataset folders, validates image readability, analyzes class distribution,
identifies image dimension statistics, and detects duplicate or corrupted files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import cv2
import numpy as np


class DatasetAuditor:
    """Audits image datasets for eye-state classification (open vs closed)."""

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize DatasetAuditor with dataset root path."""
        self.data_dir = Path(data_dir)
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    def audit(self) -> Dict[str, Any]:
        """Audit all splits and classes in the dataset directory.

        Returns a detailed summary dictionary containing file counts,
        dimension statistics, corrupt files, duplicates, and class balance.
        """
        if not self.data_dir.exists():
            return {
                "status": "error",
                "message": f"Dataset directory {self.data_dir} does not exist",
            }

        report: Dict[str, Any] = {
            "dataset_path": str(self.data_dir.resolve()),
            "total_files": 0,
            "valid_images": 0,
            "corrupted_images": [],
            "duplicate_count": 0,
            "classes": {},
            "splits": {},
            "dimensions": {
                "min_width": None,
                "max_width": None,
                "min_height": None,
                "max_height": None,
                "mean_width": 0.0,
                "mean_height": 0.0,
            },
        }

        seen_hashes: Set[str] = set()
        widths: List[int] = []
        heights: List[int] = []

        all_files = [
            f for f in self.data_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.valid_extensions
        ]
        report["total_files"] = len(all_files)

        for filepath in all_files:
            rel_parts = filepath.relative_to(self.data_dir).parts
            split_name = rel_parts[0] if len(rel_parts) > 1 else "root"
            class_name = rel_parts[-2] if len(rel_parts) > 1 else "unknown"

            # Track class counts
            report["classes"][class_name] = report["classes"].get(class_name, 0) + 1
            if split_name not in report["splits"]:
                report["splits"][split_name] = {}
            report["splits"][split_name][class_name] = (
                report["splits"][split_name].get(class_name, 0) + 1
            )

            # Check duplicates via MD5 hash
            try:
                file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()
                if file_hash in seen_hashes:
                    report["duplicate_count"] += 1
                else:
                    seen_hashes.add(file_hash)
            except Exception:
                pass

            # Validate readability with OpenCV
            img = cv2.imread(str(filepath))
            if img is None:
                report["corrupted_images"].append(str(filepath))
                continue

            report["valid_images"] += 1
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

        if widths and heights:
            report["dimensions"]["min_width"] = int(min(widths))
            report["dimensions"]["max_width"] = int(max(widths))
            report["dimensions"]["min_height"] = int(min(heights))
            report["dimensions"]["max_height"] = int(max(heights))
            report["dimensions"]["mean_width"] = round(float(np.mean(widths)), 2)
            report["dimensions"]["mean_height"] = round(float(np.mean(heights)), 2)

        return report


def main() -> None:
    """CLI entry point for data audit."""
    parser = argparse.ArgumentParser(description="Audit eye-state dataset")
    parser.add_argument("--data", default="data/eye_state", help="Dataset directory")
    parser.add_argument("--out", default="reports/data_audit_report.json", help="Output JSON path")
    args = parser.parse_args()

    auditor = DatasetAuditor(args.data)
    result = auditor.audit()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Data audit completed. Results written to {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
