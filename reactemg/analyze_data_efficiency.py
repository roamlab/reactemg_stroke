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
        k_dir = participant_dir / f"K{n}"

        # For each trial, collapse the 5 conditions into one per-trial score
        # (mean transition accuracy across conditions). Std across the resulting
        # per-trial scores reflects calibration-sample variability, which is the
        # quantity a data-efficiency curve is asking about.
        per_trial_scores = []
        per_condition_trial_accs = {c: [] for c in CONDITIONS}
        for trial_file in sorted(k_dir.glob("trial_*/metrics.json")):
            with open(trial_file) as f:
                td = json.load(f)
            cond_accs = []
            for cond in CONDITIONS:
                if cond in td["results"]:
                    acc = td["results"][cond]["transition_accuracy"]
                    cond_accs.append(acc)
                    per_condition_trial_accs[cond].append(acc)
            if cond_accs:
                per_trial_scores.append(float(np.mean(cond_accs)))

        if not per_trial_scores:
            print(f"Warning: No trial results for {participant}/N={n}")
            results[n] = None
            continue

        per_trial_arr = np.array(per_trial_scores)
        avg_trans_acc = float(per_trial_arr.mean())
        std_trans_acc = float(per_trial_arr.std(ddof=1)) if len(per_trial_arr) > 1 else 0.0

        per_condition = {
            c: float(np.mean(v)) for c, v in per_condition_trial_accs.items() if v
        }

        results[n] = {
            "avg_transition_accuracy": avg_trans_acc,
            "std_transition_accuracy": std_trans_acc,
            "per_condition": per_condition,
            "num_trials": len(per_trial_scores),
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
        n_trials = results[n].get("num_trials", 0)

        print(f"\nN={n}: {avg_acc:.4f} ± {std_acc:.4f} ({avg_acc*100:.2f}% ± {std_acc*100:.2f}%, std over {n_trials} trials)")
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


def read_main_experiment_mean(participant: str, variant: str, metric: str,
                              main_results_dir: Path = None) -> float | None:
    """Mean of `metric` across the 5 test conditions for a final main_experiment model."""
    if main_results_dir is None:
        main_results_dir = Path(__file__).parent / "results" / "main_experiment"
    else:
        main_results_dir = Path(main_results_dir)
    cell_dir = main_results_dir / participant / variant
    values = []
    for cond in CONDITIONS:
        path = cell_dir / cond / "metrics_summary.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        if metric in data and data[metric] is not None:
            values.append(float(data[metric]))
    return float(np.mean(values)) if values else None


VARIANT_STYLES = {
    "lora":          {"label": "LoRA",           "color": "#2A9D8F", "marker": "o", "linestyle": "-"},
    "full_finetune": {"label": "Full Fine-tune", "color": "#E63946", "marker": "s", "linestyle": "--"},
    "head_only":     {"label": "Head-only",      "color": "#457B9D", "marker": "^", "linestyle": "-."},
    "stroke_only":   {"label": "Stroke-only",    "color": "#F4A261", "marker": "D", "linestyle": ":"},
}


def plot_variants_compared(
    variants: list,
    results_dir: str = None,
    main_results_dir: str = None,
    output_path: str = None,
):
    """One panel per subject; grouped bars per calibration budget.

    Groups on the x-axis are budgets N=1,4,8,All. Within each group there is one
    bar per fine-tuning variant. Error bars show std across the 12 per-trial
    averages (N=1,4,8 only; N=All is a single final model, so no error bar).
    Zero-shot (N=0) is drawn as a horizontal reference line per panel.
    """
    sns.set_theme(style="white", context="paper", font_scale=1.35)
    plt.rcParams["font.family"] = "DejaVu Sans"

    budgets = [1, 4, 8, "All"]
    budget_labels = ["1", "4", "8", "All"]
    group_x = np.arange(len(budgets))
    bar_w = 0.8 / len(variants)

    fig, axes = plt.subplots(1, len(PARTICIPANTS), figsize=(12, 4.8), sharey=True)
    if len(PARTICIPANTS) == 1:
        axes = [axes]

    print(f"\n{'='*78}")
    print(f"Data Efficiency (grouped bars): {' vs '.join(VARIANT_STYLES[v]['label'] for v in variants)}")
    print(f"{'='*78}")

    for ax_idx, participant in enumerate(PARTICIPANTS):
        ax = axes[ax_idx]
        subject = PARTICIPANT_LABELS[participant].upper()

        # Zero-shot baseline (N=0) — shared across variants, drawn as a reference line
        n0_val = read_main_experiment_mean(
            participant, "zero_shot", "transition_accuracy", main_results_dir
        )
        print(f"\n  {subject} ({participant}):  Zero-shot = "
              + (f"{n0_val:.3f}" if n0_val is not None else "N/A"))
        if n0_val is not None:
            ax.axhline(
                n0_val, color="#444444", linestyle=(0, (5, 2)), linewidth=1.9,
                zorder=6, label="Zero-shot" if ax_idx == 0 else None,
            )

        for v_idx, variant in enumerate(variants):
            style = VARIANT_STYLES[variant]
            try:
                trial_results = analyze_results(variant, participant, results_dir)
            except FileNotFoundError:
                print(f"    Warning: no data for {variant}/{participant}")
                continue

            n_all = read_main_experiment_mean(
                participant, variant, "transition_accuracy", main_results_dir
            )

            heights, errs = [], []
            for b in budgets:
                if b == "All":
                    heights.append(n_all if n_all is not None else 0.0)
                    errs.append(np.nan)  # single model — no trial std
                else:
                    r = trial_results.get(b)
                    if r and r["avg_transition_accuracy"] is not None:
                        heights.append(r["avg_transition_accuracy"])
                        errs.append(r["std_transition_accuracy"] or 0.0)
                    else:
                        heights.append(0.0)
                        errs.append(np.nan)

            row = f"    {style['label']:<15}: "
            for b in [1, 4, 8]:
                r = trial_results.get(b)
                row += (f"N={b}: {r['avg_transition_accuracy']:.3f}±{r['std_transition_accuracy']:.3f}   "
                        if r and r["avg_transition_accuracy"] is not None else f"N={b}: N/A   ")
            row += f"All: {n_all:.3f}" if n_all is not None else "All: N/A"
            print(row)

            offset = (v_idx - (len(variants) - 1) / 2) * bar_w
            ax.bar(
                group_x + offset, heights, width=bar_w,
                yerr=errs, capsize=3, error_kw={"elinewidth": 1.2, "capthick": 1.2},
                color=style["color"], edgecolor="white", linewidth=0.6,
                label=style["label"] if ax_idx == 0 else None, zorder=3,
            )

        ax.set_title(subject, fontsize=15, fontstyle="italic", pad=8, color="#333333")
        ax.set_xticks(group_x)
        ax.set_xticklabels(budget_labels)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Calibration reps (N)", fontsize=13)
        if ax_idx == 0:
            ax.set_ylabel("Avg. Transition Accuracy", fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        sns.despine(ax=ax)
        ax.tick_params(axis="both", which="major", labelsize=12)

    print(f"\n{'='*78}\n")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles),
        frameon=False,
        fontsize=13,
        columnspacing=1.8,
        handlelength=1.6,
    )

    fig.tight_layout(pad=0.5, w_pad=0.6)
    fig.subplots_adjust(bottom=0.2)

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
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
        default=None,
        choices=['stroke_only', 'head_only', 'lora', 'full_finetune'],
        help="Fine-tuning variant. Required for --plot or per-participant analysis; ignored with --compare."
    )
    parser.add_argument(
        "--participant", "-p",
        default=None,
        help="Participant ID (e.g., p4, p15, p20). Required for per-participant analysis."
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to data_efficiency results dir (default: ./results/data_efficiency)"
    )
    parser.add_argument(
        "--main_results_dir",
        default=None,
        help="Path to main_experiment results dir for N=0 and N=All endpoints "
             "(default: ./results/main_experiment)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot all three subjects (p4=s1, p15=s2, p20=s3) for the given --variant"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare LoRA and Full Fine-tune on one figure (1 panel per subject)"
    )
    parser.add_argument(
        "--compare_variants",
        nargs="+",
        default=["head_only", "lora", "full_finetune"],
        choices=['stroke_only', 'head_only', 'lora', 'full_finetune'],
        help="Variants to overlay in --compare mode (default: head_only lora full_finetune)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the plot"
    )

    args = parser.parse_args()

    if args.compare:
        plot_variants_compared(
            variants=args.compare_variants,
            results_dir=args.results_dir,
            main_results_dir=args.main_results_dir,
            output_path=args.output,
        )
    elif args.plot:
        if args.variant is None:
            parser.error("--variant is required for --plot")
        plot_all_subjects(args.variant, args.results_dir, args.output)
    else:
        if args.variant is None or args.participant is None:
            parser.error("--variant and --participant are required unless --plot or --compare is used")
        results = analyze_results(args.variant, args.participant, args.results_dir)
        print_results(args.variant, args.participant, results)


if __name__ == "__main__":
    main()
