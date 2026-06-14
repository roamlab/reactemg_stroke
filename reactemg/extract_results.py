#!/usr/bin/env python3
"""
Extract main experiment results in Table II layout.

Each cell is mean ± std across the 5 held-out test conditions.
Avg columns are the mean across participants.
Participant mapping: s1=S1, s2=S2, s3=S3.

Usage:
    python3 extract_results.py
    python3 extract_results.py --results_dir /custom/path
"""

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ["zero_shot", "stroke_only", "head_only", "lora", "full_finetune"]
VARIANT_LABELS = {
    "zero_shot": "Zero-shot",
    "stroke_only": "Stroke-only",
    "head_only": "Head-only",
    "lora": "LoRA",
    "full_finetune": "Full",
}

PARTICIPANTS = ["s1", "s2", "s3"]
PARTICIPANT_LABELS = {"s1": "S1", "s2": "S2", "s3": "S3"}

CONDITIONS = [
    "mid_session_baseline",
    "end_session_baseline",
    "unseen_posture",
    "sensor_shift",
    "orthosis_actuated",
]

METRICS = [
    ("raw_accuracy", "Raw Accuracy"),
    ("transition_accuracy", "Transition Accuracy"),
]


def collect_cell(results_dir: Path, participant: str, variant: str, metric_key: str):
    """Return list of per-condition values for one (participant, variant, metric) cell."""
    variant_dir = results_dir / participant / variant
    values = []
    for cond in CONDITIONS:
        path = variant_dir / cond / "metrics_summary.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if metric_key in data and data[metric_key] is not None:
            values.append(float(data[metric_key]))
    return values


def cell_stats(values):
    if not values:
        return None, None
    arr = np.array(values)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def build_table(results_dir: Path):
    """out[metric_key][variant] = {"s1": (mean,std), ..., "avg": (mean,None)}"""
    out = {}
    for metric_key, _ in METRICS:
        out[metric_key] = {}
        for variant in VARIANTS:
            out[metric_key][variant] = {}
            participant_means = []
            for p in PARTICIPANTS:
                values = collect_cell(results_dir, p, variant, metric_key)
                mean, std = cell_stats(values)
                out[metric_key][variant][p] = (mean, std)
                if mean is not None:
                    participant_means.append(mean)
            avg = float(np.mean(participant_means)) if participant_means else None
            out[metric_key][variant]["avg"] = (avg, None)
    return out


def format_table(table) -> str:
    """Return Table II as a single aligned plain-text block.

    Layout: Method | Raw[S1 S2 S3] | Trans[S1 S2 S3] | Avg[Raw Trans]
    """
    cell_w = 11
    method_w = 12
    subj_section_w = cell_w * 3
    avg_section_w = cell_w * 2

    title = ("Table II — Intent Detection Performance "
             "(mean ± std over 5 test conditions; s1=S1, s2=S2, s3=S3)")

    top = (f"{'':<{method_w}} | "
           f"{'Raw Accuracy':^{subj_section_w}} | "
           f"{'Transition Accuracy':^{subj_section_w}} | "
           f"{'Avg':^{avg_section_w}}")

    sub = f"{'Method':<{method_w}} | "
    for p in ["S1", "S2", "S3"]:
        sub += f"{p:^{cell_w}}"
    sub += " | "
    for p in ["S1", "S2", "S3"]:
        sub += f"{p:^{cell_w}}"
    sub += " | "
    sub += f"{'Raw':^{cell_w}}{'Trans':^{cell_w}}"

    bar_w = max(len(title), len(top), len(sub))

    lines = ["=" * bar_w, title, "=" * bar_w, top, sub, "-" * bar_w]

    for variant in VARIANTS:
        row = f"{VARIANT_LABELS[variant]:<{method_w}} | "
        for p in PARTICIPANTS:
            mean, std = table["raw_accuracy"][variant][p]
            cell = f"{mean:.2f}±{std:.2f}" if mean is not None else "—"
            row += f"{cell:^{cell_w}}"
        row += " | "
        for p in PARTICIPANTS:
            mean, std = table["transition_accuracy"][variant][p]
            cell = f"{mean:.2f}±{std:.2f}" if mean is not None else "—"
            row += f"{cell:^{cell_w}}"
        row += " | "
        for metric_key, _ in METRICS:
            avg, _ = table[metric_key][variant]["avg"]
            cell = f"{avg:.2f}" if avg is not None else "—"
            row += f"{cell:^{cell_w}}"
        lines.append(row)

    lines.append("=" * bar_w)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Extract main experiment results in Table II layout")
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to results directory (default: ./results/main_experiment)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the formatted table "
             "(default: ./results/main_experiment/table2.txt)",
    )
    args = parser.parse_args()

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else Path(__file__).parent / "results" / "main_experiment"
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "results" / "main_experiment" / "table2.txt"
    )

    table = build_table(results_dir)
    text = format_table(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
