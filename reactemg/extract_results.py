#!/usr/bin/env python3
"""
Extract and summarize results from main experiment.

Usage:
    python3 extract_results.py --participant p4
    python3 extract_results.py --participant p15
    python3 extract_results.py --participant p20
"""

import argparse
import json
import os
from pathlib import Path


VARIANTS = ["zero_shot", "stroke_only", "head_only", "lora", "full_finetune"]
CONDITIONS = [
    "mid_session_baseline",
    "end_session_baseline",
    "unseen_posture",
    "sensor_shift",
    "orthosis_actuated"
]
METRICS = ["raw_accuracy", "transition_accuracy", "average_latency"]


def extract_results(participant: str, results_dir: str = None) -> dict:
    """
    Extract metrics for a participant, averaged across test conditions.

    Returns:
        dict: {variant: {metric: avg_value}}
    """
    if results_dir is None:
        results_dir = Path(__file__).parent / "results" / "main_experiment"
    else:
        results_dir = Path(results_dir)

    participant_dir = results_dir / participant

    if not participant_dir.exists():
        raise FileNotFoundError(f"Results not found for participant: {participant}")

    results = {}

    for variant in VARIANTS:
        variant_dir = participant_dir / variant

        if not variant_dir.exists():
            print(f"Warning: No results for {participant}/{variant}")
            results[variant] = {m: None for m in METRICS}
            continue

        metric_values = {m: [] for m in METRICS}

        for condition in CONDITIONS:
            json_path = variant_dir / condition / "metrics_summary.json"

            if not json_path.exists():
                print(f"Warning: Missing {participant}/{variant}/{condition}")
                continue

            with open(json_path) as f:
                data = json.load(f)

            for metric in METRICS:
                if metric in data and data[metric] is not None:
                    metric_values[metric].append(data[metric])

        # Compute averages
        results[variant] = {}
        for metric in METRICS:
            values = metric_values[metric]
            if values:
                results[variant][metric] = sum(values) / len(values)
            else:
                results[variant][metric] = None

    return results


def print_results(participant: str, results: dict):
    """Print results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"Results for {participant} (averaged across 5 test conditions)")
    print(f"{'='*70}")
    print(f"{'Variant':<15} {'Raw Acc':>12} {'Trans Acc':>12} {'Avg Latency':>12}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    for variant in VARIANTS:
        metrics = results[variant]
        raw_acc = f"{metrics['raw_accuracy']:.4f}" if metrics['raw_accuracy'] is not None else "N/A"
        trans_acc = f"{metrics['transition_accuracy']:.4f}" if metrics['transition_accuracy'] is not None else "N/A"
        latency = f"{metrics['average_latency']:.2f}" if metrics['average_latency'] is not None else "N/A"

        print(f"{variant:<15} {raw_acc:>12} {trans_acc:>12} {latency:>12}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract main experiment results")
    parser.add_argument(
        "--participant", "-p",
        required=True,
        help="Participant ID (e.g., p4, p15, p20)"
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to results directory (default: ./results/main_experiment)"
    )

    args = parser.parse_args()

    results = extract_results(args.participant, args.results_dir)
    print_results(args.participant, results)


if __name__ == "__main__":
    main()
