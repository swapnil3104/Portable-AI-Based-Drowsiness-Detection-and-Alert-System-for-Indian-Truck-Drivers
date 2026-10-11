"""Model Benchmark Reporting Utility for DrishtiDrive AI.

Loads real test evaluation metrics from reports/model_comparison.csv
and generates summary markdown or CSV representations for documentation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd


def generate_benchmark_summary(
    csv_path: str | Path = "reports/model_comparison.csv",
    output_md: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load real benchmark results and optionally format as markdown."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Benchmark file {csv_file} does not exist. Run train_compare.py first."
        )

    df = pd.read_csv(csv_file)
    print("\n=== DrishtiDrive AI Model Benchmark Results ===")
    print(df.to_string(index=False))

    if output_md:
        out_path = Path(output_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md_table = df.to_markdown(index=False)
        out_path.write_text(md_table, encoding="utf-8")
        print(f"\nMarkdown summary saved to {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Summary Tool")
    parser.add_argument("--csv", default="reports/model_comparison.csv", help="Input comparison CSV")
    parser.add_argument("--md", default=None, help="Optional output markdown file")
    args = parser.parse_args()
    generate_benchmark_summary(args.csv, args.md)
