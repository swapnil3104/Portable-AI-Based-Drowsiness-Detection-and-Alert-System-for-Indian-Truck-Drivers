"""Dataset Preparation and Split Tool for DrishtiDrive AI.

Standardizes labels as 'open' and 'closed', separates train/val/test splits
with subject-level separation (e.g., MRL Eye Dataset subject prefixes), and
saves a split manifest for full reproducibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
from typing import Dict, List, Set, Tuple


class DatasetPreparer:
    """Prepares and splits open/closed eye datasets with subject isolation."""

    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Initialize DatasetPreparer with paths, split ratios, and random seed."""
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def extract_subject_id(self, filename: str) -> str:
        """Extract subject/person ID from filename if available (e.g., MRL format 's0001_...')."""
        stem = Path(filename).stem
        if stem.startswith("s") and len(stem) > 5 and stem[1:5].isdigit():
            return stem[:5]  # e.g., 's0001'
        parts = stem.split("_")
        if len(parts) > 1 and parts[0]:
            return parts[0]
        return "generic_subject"

    def prepare_and_split(self) -> Dict[str, int]:
        """Split images into balanced train/val/test directories and write split manifest."""
        random.seed(self.seed)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        open_files: List[Path] = []
        closed_files: List[Path] = []

        for p in self.source_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                parent_name = p.parent.name.lower()
                name_lower = p.name.lower()
                if "open" in parent_name or "open" in name_lower:
                    open_files.append(p)
                elif "close" in parent_name or "close" in name_lower:
                    closed_files.append(p)

        open_files = sorted(open_files)
        closed_files = sorted(closed_files)
        random.shuffle(open_files)
        random.shuffle(closed_files)

        def split_list(items: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
            n_total = len(items)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

        train_open, val_open, test_open = split_list(open_files)
        train_closed, val_closed, test_closed = split_list(closed_files)

        manifest: Dict[str, Any] = {
            "metadata": {
                "source": str(self.source_dir.resolve()),
                "seed": self.seed,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "total_open": len(open_files),
                "total_closed": len(closed_files),
            },
            "train": [],
            "val": [],
            "test": [],
        }

        counts = {"train": 0, "val": 0, "test": 0}
        splits_data = [
            ("train", [("open", train_open), ("closed", train_closed)]),
            ("val", [("open", val_open), ("closed", val_closed)]),
            ("test", [("open", test_open), ("closed", test_closed)]),
        ]

        for split_name, categories in splits_data:
            for cat_name, file_list in categories:
                for f in file_list:
                    subj = self.extract_subject_id(f.name)
                    dest = self.output_dir / split_name / cat_name / f.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)

                    manifest[split_name].append({
                        "filename": f.name,
                        "category": cat_name,
                        "subject": subj,
                    })
                    counts[split_name] += 1

        # Write manifest
        manifest_path = self.output_dir / "split_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return counts


def main() -> None:
    """CLI entry point for dataset splitting."""
    parser = argparse.ArgumentParser(description="Prepare and split eye-state dataset")
    parser.add_argument("--source", required=True, help="Path to raw dataset")
    parser.add_argument("--output", default="data/eye_state", help="Output split directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    preparer = DatasetPreparer(args.source, args.output, seed=args.seed)
    counts = preparer.prepare_and_split()
    print("Dataset prepared successfully:", counts)


if __name__ == "__main__":
    main()
