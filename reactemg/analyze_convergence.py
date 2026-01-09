#!/usr/bin/env python3
"""
Analyze convergence experiment results and generate dual-axis plot.

This script visualizes convergence behavior and catastrophic forgetting
by plotting stroke performance vs healthy performance over training epochs.

Usage:
    python3 analyze_convergence.py --variant lora --participant p4
    python3 analyze_convergence.py --variant lora --participant p15
    python3 analyze_convergence.py --variant head_only --participant p20
    python3 analyze_convergence.py --variant lora --participant p4 --output my_plot.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = [
    "mid_session_baseline",
    "end_session_baseline",
    "unseen_posture",
    "sensor_shift",
    "orthosis_actuated"
]


def load_convergence_data(variant: str, participant: str, results_dir: str = None) -> Dict:
    """
    Load convergence curves data for a participant.

    Args:
        variant: Fine-tuning variant (e.g., 'lora', 'head_only')
        participant: Participant ID (e.g., 'p4', 'p15', 'p20')
        results_dir: Path to results directory

    Returns:
        Dict containing convergence curves data
    """
    if results_dir is None:
        results_dir = Path(__file__).parent / "results" / "convergence"
    else:
        results_dir = Path(results_dir)

    curves_file = results_dir / variant / participant / "convergence_curves.json"

    if not curves_file.exists():
        raise FileNotFoundError(
            f"Convergence results not found for {variant}/{participant}\n"
            f"Expected file: {curves_file}"
        )

    with open(curves_file) as f:
        return json.load(f)


def extract_plot_data(data: Dict) -> Dict:
    """
    Extract data needed for plotting.

    Returns:
        Dict with epochs, stroke_accs, healthy_accs, and baseline values
    """
    # Extract frozen baseline (shown as horizontal lines only)
    frozen = data["frozen_baseline"]
    frozen_stroke = frozen["stroke_avg_transition_acc"]
    frozen_healthy = frozen["healthy_results"]["transition_accuracy"]

    # Extract training epochs (keep original epoch numbers, starting from 1)
    epochs = []
    stroke_accs = []
    healthy_accs = []

    for result in data["epoch_results"]:
        epochs.append(result["epoch"] + 1)  # +1 so first training epoch is 1
        stroke_accs.append(result["stroke_avg_transition_acc"])
        healthy_accs.append(result["healthy_results"]["transition_accuracy"])

    return {
        "epochs": epochs,
        "stroke_accs": stroke_accs,
        "healthy_accs": healthy_accs,
        "frozen_stroke": frozen_stroke,
        "frozen_healthy": frozen_healthy,
        "base_epochs": data["base_epochs"],
        "extended_epochs": data["extended_epochs"],
        "variant": data["variant"],
    }


def create_convergence_plot(
    variant: str,
    participant: str,
    plot_data: Dict,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Create dual-axis convergence plot.

    Args:
        variant: Fine-tuning variant
        participant: Participant ID
        plot_data: Data extracted by extract_plot_data()
        output_path: Path to save the plot (default: results/convergence/{variant}/{participant}/convergence_plot.png)
        show_plot: Whether to display the plot interactively

    Returns:
        Path to the saved plot
    """
    epochs = plot_data["epochs"]
    stroke_accs = plot_data["stroke_accs"]
    healthy_accs = plot_data["healthy_accs"]
    frozen_stroke = plot_data["frozen_stroke"]
    frozen_healthy = plot_data["frozen_healthy"]
    variant = plot_data["variant"]

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Colors
    stroke_color = "#2E86AB"  # Blue
    healthy_color = "#A23B72"  # Magenta/pink

    # Plot stroke performance (left axis)
    line1 = ax1.plot(
        epochs, stroke_accs,
        color=stroke_color, linewidth=2, marker='o', markersize=6,
        label="Stroke"
    )
    ax1.axhline(
        y=frozen_stroke, color=stroke_color, linestyle='--', linewidth=1.5,
        alpha=0.7, label="Frozen healthy model zeroshot - stroke"
    )

    # Plot healthy performance (right axis)
    line2 = ax2.plot(
        epochs, healthy_accs,
        color=healthy_color, linewidth=2, marker='s', markersize=6,
        label="Healthy"
    )
    ax2.axhline(
        y=frozen_healthy, color=healthy_color, linestyle='--', linewidth=1.5,
        alpha=0.7, label="Frozen healthy model zeroshot - healthy"
    )

    # Configure left axis (Stroke)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Stroke Transition Accuracy", fontsize=12, color=stroke_color)
    ax1.tick_params(axis='y', labelcolor=stroke_color)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(min(epochs) - 5, max(epochs) + 5)

    # Configure right axis (Healthy)
    ax2.set_ylabel("Healthy Transition Accuracy", fontsize=12, color=healthy_color)
    ax2.tick_params(axis='y', labelcolor=healthy_color)
    ax2.set_ylim(0, 1.0)

    # Title
    plt.title(
        f"Convergence & Catastrophic Forgetting: {participant} ({variant})",
        fontsize=14, fontweight='bold'
    )

    # Combined legend at the bottom (below the figure)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        lines1 + lines2, labels1 + labels2,
        loc='upper center', ncol=2, fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.02)
    )

    # Tight layout with space for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Determine output path
    if output_path is None:
        output_dir = Path(__file__).parent / "results" / "convergence" / variant / participant
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "convergence_plot.png"
    else:
        output_path = Path(output_path)

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return str(output_path)


def print_summary(participant: str, plot_data: Dict):
    """Print a text summary of the convergence results."""
    epochs = plot_data["epochs"]
    stroke_accs = plot_data["stroke_accs"]
    healthy_accs = plot_data["healthy_accs"]
    frozen_stroke = plot_data["frozen_stroke"]
    frozen_healthy = plot_data["frozen_healthy"]
    variant = plot_data["variant"]

    # Find peak stroke performance
    peak_idx = np.argmax(stroke_accs)
    peak_epoch = epochs[peak_idx]
    peak_stroke = stroke_accs[peak_idx]
    healthy_at_peak = healthy_accs[peak_idx]

    print(f"\n{'='*65}")
    print(f"Convergence Summary: {participant} ({variant})")
    print(f"{'='*65}")

    print(f"\n{'Metric':<30} {'Stroke':>12} {'Healthy':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12}")

    print(f"{'Frozen baseline':<30} {frozen_stroke:>11.1%} {frozen_healthy:>11.1%}")
    print(f"{'Peak stroke (epoch ' + str(peak_epoch) + ')':<30} {peak_stroke:>11.1%} {healthy_at_peak:>11.1%}")
    print(f"{'Final epoch (' + str(epochs[-1]) + ')':<30} {stroke_accs[-1]:>11.1%} {healthy_accs[-1]:>11.1%}")

    print(f"\n{'Change from Frozen Baseline':<30}")
    print(f"{'-'*54}")
    print(f"{'At peak stroke epoch':<30} {peak_stroke - frozen_stroke:>+11.1%} {healthy_at_peak - frozen_healthy:>+11.1%}")

    # Forgetting ratio
    forgetting = (frozen_healthy - healthy_at_peak) / frozen_healthy * 100 if frozen_healthy > 0 else 0
    print(f"\nCatastrophic forgetting at peak: {forgetting:.1f}% of healthy performance lost")

    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze convergence experiment results and generate plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 analyze_convergence.py --variant lora --participant p4
    python3 analyze_convergence.py --variant lora --participant p15 --show
    python3 analyze_convergence.py -v head_only -p p20 --output my_plot.png
        """
    )
    parser.add_argument(
        "--variant", "-v",
        required=True,
        choices=['stroke_only', 'head_only', 'lora', 'full_finetune'],
        help="Fine-tuning variant (stroke_only, head_only, lora, full_finetune)"
    )
    parser.add_argument(
        "--participant", "-p",
        required=True,
        help="Participant ID (e.g., p4, p15, p20)"
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to results directory (default: ./results/convergence)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the plot (default: results/convergence/{variant}/{participant}/convergence_plot.png)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (in addition to saving)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the text summary"
    )

    args = parser.parse_args()

    # Load data
    data = load_convergence_data(args.variant, args.participant, args.results_dir)

    # Extract plot data
    plot_data = extract_plot_data(data)

    # Print summary
    if not args.no_summary:
        print_summary(args.participant, plot_data)

    # Create plot
    output_path = create_convergence_plot(
        variant=args.variant,
        participant=args.participant,
        plot_data=plot_data,
        output_path=args.output,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
