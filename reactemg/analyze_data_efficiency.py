#!/usr/bin/env python3
"""
Analyze data efficiency experiment results.
For a given subject and each N, compute the average transition accuracy across all test conditions.

Usage:
    python3 analyze_data_efficiency.py --variant lora --participant p4
    python3 analyze_data_efficiency.py --variant lora --participant p15
    python3 analyze_data_efficiency.py --variant head_only --participant p20

    # Plot all subjects for a variant:
    python3 analyze_data_efficiency.py --variant lora --plot
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


BUDGETS = [1, 4, 8]
PARTICIPANTS = ["p4", "p15", "p20"]
PARTICIPANT_LABELS = {"p4": "s1", "p15": "s2", "p20": "s3"}

# Zero-shot accuracy (N=0, no fine-tuning)
ZERO_SHOT_ACC = {"p4": 0.05, "p15": 0.22, "p20": 0.13}

# Full data accuracy (all calibration data) - per variant
FULL_DATA_ACC = {
    "lora": {"p4": 0.45, "p15": 0.62, "p20": 0.83},
    "full_finetune": {"p4": 0.40, "p15": 0.62, "p20": 0.82},
    "head_only": {"p4": 0.45, "p15": 0.62, "p20": 0.83},
    "stroke_only": {"p4": 0.45, "p15": 0.62, "p20": 0.83},
}

# X-axis labels and positions (evenly spaced)
X_LABELS = ["0", "1", "4", "8", "All"]
X_POSITIONS = [0, 1, 2, 3, 4]
CONDITIONS = [
    "mid_session_baseline",
    "end_session_baseline",
    "unseen_posture",
    "sensor_shift",
    "orthosis_actuated"
]


def analyze_results(variant: str, participant: str, results_dir: str = None) -> dict:
    """
    Analyze data efficiency results for a participant.

    Returns:
        dict: {N: {"avg_transition_accuracy": float, "std_transition_accuracy": float, "per_condition": {cond: float}}}
    """
    if results_dir is None:
        results_dir = Path(__file__).parent / "results" / "data_efficiency"
    else:
        results_dir = Path(results_dir)

    participant_dir = results_dir / variant / participant

    if not participant_dir.exists():
        raise FileNotFoundError(f"Results not found for {variant}/{participant}")

    results = {}

    for n in BUDGETS:
        agg_file = participant_dir / f"K{n}" / "aggregated_metrics.json"

        if not agg_file.exists():
            print(f"Warning: No results for {participant}/N={n}")
            results[n] = None
            continue

        with open(agg_file) as f:
            data = json.load(f)

        # Extract transition accuracy mean for each test condition
        agg_results = data["aggregated_results"]
        conditions = list(agg_results.keys())

        per_condition = {}
        trans_accs = []
        for cond in conditions:
            acc = agg_results[cond]["transition_accuracy_mean"]
            per_condition[cond] = acc
            trans_accs.append(acc)

        trans_accs_arr = np.array(trans_accs)
        avg_trans_acc = float(np.mean(trans_accs_arr)) if len(trans_accs_arr) > 0 else None
        std_trans_acc = float(np.std(trans_accs_arr)) if len(trans_accs_arr) > 0 else None

        results[n] = {
            "avg_transition_accuracy": avg_trans_acc,
            "std_transition_accuracy": std_trans_acc,
            "per_condition": per_condition
        }

    return results


def print_results(variant: str, participant: str, results: dict):
    """Print results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Data Efficiency Results: {variant}/{participant}")
    print(f"Average Transition Accuracy across 5 test conditions")
    print(f"{'='*60}")

    for n in BUDGETS:
        if results[n] is None:
            print(f"\nN={n}: No results found")
            continue

        avg_acc = results[n]["avg_transition_accuracy"]
        std_acc = results[n]["std_transition_accuracy"]
        per_cond = results[n]["per_condition"]

        print(f"\nN={n}: {avg_acc:.4f} ± {std_acc:.4f} ({avg_acc*100:.2f}% ± {std_acc*100:.2f}%)")
        print(f"{'-'*40}")
        for cond in CONDITIONS:
            if cond in per_cond:
                acc = per_cond[cond]
                print(f"  {cond:<25} {acc:.4f}")

    print(f"\n{'='*60}\n")


def plot_all_subjects(variant: str, results_dir: str = None, output_path: str = None):
    """
    Plot data efficiency results for all three subjects on a single plot.

    Args:
        variant: Fine-tuning variant (stroke_only, head_only, lora, full_finetune)
        results_dir: Path to results directory
        output_path: Path to save the plot (optional)
    """
    # Set seaborn style for clean, publication-quality plots
    sns.set_theme(style="white", context="paper", font_scale=1.2)

    # Use a clean color palette
    palette = sns.color_palette("deep", n_colors=3)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Mapping from budget values to evenly-spaced x positions
    budget_to_xpos = {0: 0, 1: 1, 4: 2, 8: 3, "All": 4}

    # Print header for numerical results
    print(f"\n{'='*70}")
    print(f"Data Efficiency Results: {variant}")
    print(f"{'='*70}")
    print(f"{'Subject':<10} {'N=0':<12} {'N=1':<12} {'N=4':<12} {'N=8':<12} {'All':<12}")
    print(f"{'-'*70}")

    for i, participant in enumerate(PARTICIPANTS):
        # Start with zero-shot accuracy at N=0 (no std available for zero-shot)
        x_vals = [budget_to_xpos[0]]
        y_vals = [ZERO_SHOT_ACC[participant]]
        std_vals = [0.0]  # No std for zero-shot point

        # Add data efficiency results for N=1, 4, 8
        try:
            results = analyze_results(variant, participant, results_dir)
            for n in BUDGETS:
                if results[n] is not None and results[n]["avg_transition_accuracy"] is not None:
                    x_vals.append(budget_to_xpos[n])
                    y_vals.append(results[n]["avg_transition_accuracy"])
                    std_vals.append(results[n]["std_transition_accuracy"] or 0.0)
        except FileNotFoundError:
            print(f"Warning: No results found for {variant}/{participant}, using only zero-shot and full data")

        # Add full data accuracy at "All" (no std available for full data)
        x_vals.append(budget_to_xpos["All"])
        y_vals.append(FULL_DATA_ACC[variant][participant])
        std_vals.append(0.0)  # No std for full data point

        # Print the numerical values for this participant
        label = PARTICIPANT_LABELS[participant]
        row_values = []
        for idx, x_label in enumerate(X_LABELS):
            if idx < len(y_vals):
                if std_vals[idx] > 0:
                    row_values.append(f"{y_vals[idx]:.2f}±{std_vals[idx]:.2f}")
                else:
                    row_values.append(f"{y_vals[idx]:.2f}")
            else:
                row_values.append("N/A")
        print(f"{label:<10} {row_values[0]:<12} {row_values[1]:<12} {row_values[2]:<12} {row_values[3]:<12} {row_values[4]:<12}")

        # Convert to numpy arrays for fill_between
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        std_vals = np.array(std_vals)

        # Plot line with error bars (±1 std)
        ax.errorbar(x_vals, y_vals, yerr=std_vals, marker='o', color=palette[i],
                    label=label, linewidth=2.5, markersize=9, markeredgecolor='white',
                    markeredgewidth=1.5, capsize=4, capthick=1.5)

    print(f"{'='*70}\n")

    ax.set_xlabel("Data Budget (N)", fontsize=13)
    ax.set_ylabel("Average Transition Accuracy", fontsize=13)
    ax.set_title("Data Budget Comparison", fontsize=15, fontweight='semibold', pad=12)
    ax.set_xticks(X_POSITIONS)
    ax.set_xticklabels(X_LABELS)
    ax.set_ylim(0, 1)

    # Remove top and right spines for cleaner look
    sns.despine(ax=ax)

    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)

    # Legend at the bottom, outside the axis, horizontal layout
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=12,
        columnspacing=2.0
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white',
                    edgecolor='none')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze data efficiency experiment results (N = data budget)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single participant
    python3 analyze_data_efficiency.py --variant lora --participant p4
    python3 analyze_data_efficiency.py --variant head_only --participant p15
    python3 analyze_data_efficiency.py -v lora -p p20

    # Plot all subjects (s1, s2, s3) for a variant
    python3 analyze_data_efficiency.py --variant lora --plot
    python3 analyze_data_efficiency.py --variant lora --plot --output plot.png
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
        default=None,
        help="Participant ID (e.g., p4, p15, p20). Required unless --plot is used."
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to results directory (default: ./results/data_efficiency)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot all three subjects (p4=s1, p15=s2, p20=s3) for the given variant"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the plot (optional, only used with --plot)"
    )

    args = parser.parse_args()

    if args.plot:
        plot_all_subjects(args.variant, args.results_dir, args.output)
    else:
        if args.participant is None:
            parser.error("--participant is required unless --plot is used")
        results = analyze_results(args.variant, args.participant, args.results_dir)
        print_results(args.variant, args.participant, results)


if __name__ == "__main__":
    main()
