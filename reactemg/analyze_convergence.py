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

    # Compare LoRA vs Full Finetune for p15:
    python3 analyze_convergence.py --compare --output comparison.png

    # Combined plot with all variant stroke curves + LoRA healthy curve:
    python3 analyze_convergence.py --combined -p p15
    python3 analyze_convergence.py --combined -p p15 --output combined_p15.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


PARTICIPANT_LABELS = {"p4": "s1", "p15": "s2", "p20": "s3"}


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
        "cv_best_epochs": data.get("cv_best_epochs"),
        "total_epochs": data.get("total_epochs", 100),
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
        y=frozen_stroke, color=stroke_color, linestyle=(0, (2, 1)), linewidth=1.5,
        alpha=0.7, label="Frozen healthy model zeroshot - stroke"
    )

    # Plot healthy performance (right axis)
    line2 = ax2.plot(
        epochs, healthy_accs,
        color=healthy_color, linewidth=2, marker='s', markersize=6,
        label="Healthy"
    )
    ax2.axhline(
        y=frozen_healthy, color=healthy_color, linestyle=(0, (2, 1)), linewidth=1.5,
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


def create_combined_plot(
    participant: str = "p15",
    results_dir: str = None,
    output_path: str = None,
    show_plot: bool = False,
):
    """
    Create a single plot with all variant stroke curves and LoRA healthy curve.

    Plots:
    - Full fine-tuning stroke curve
    - Head-only stroke curve
    - LoRA stroke curve
    - Stroke-only stroke curve
    - LoRA healthy curve (showing catastrophic forgetting)
    - Reference lines for zero-shot baselines

    Args:
        participant: Participant ID (default: p15)
        results_dir: Path to results directory
        output_path: Path to save the plot
        show_plot: Whether to display the plot interactively
    """
    sns.set_theme(style="white", context="paper", font_scale=2.2)

    # Define variants and their display properties (no "(Stroke)" in labels)
    variants_config = {
        "full_finetune": {"label": "Full Fine-tune", "color": "#E63946", "marker": "o"},
        "head_only": {"label": "Head-only", "color": "#457B9D", "marker": "s"},
        "lora": {"label": "LoRA", "color": "#2A9D8F", "marker": "^"},
        "stroke_only": {"label": "Stroke-only", "color": "#F4A261", "marker": "D"},
    }
    lora_healthy_config = {"label": "Healthy Retention", "color": "#9B59B6", "marker": "v"}

    # Minimum epoch to display (skip early epochs)
    min_epoch = 5

    fig, ax = plt.subplots(figsize=(12, 10))

    # Track stroke zero-shot baseline (will use first available variant's frozen_stroke)
    stroke_zeroshot = None
    lora_plot_data = None

    # Pre-load LoRA data for healthy retention curve
    try:
        lora_data = load_convergence_data("lora", participant, results_dir)
        lora_plot_data = extract_plot_data(lora_data)
    except FileNotFoundError:
        print(f"Note: No LoRA data found for {participant}, cannot plot healthy curve")

    # Plot curves in order: full_finetune, head_only, Healthy Retention, lora, stroke_only
    plot_order = ["full_finetune", "head_only", "healthy_retention", "lora", "stroke_only"]

    for item in plot_order:
        if item == "healthy_retention":
            # Plot LoRA healthy curve
            if lora_plot_data is not None:
                epochs = lora_plot_data["epochs"]
                healthy_accs = lora_plot_data["healthy_accs"]

                filtered_data = [(e, h) for e, h in zip(epochs, healthy_accs) if e >= min_epoch]
                if filtered_data:
                    filtered_epochs, filtered_healthy = zip(*filtered_data)

                    ax.plot(
                        filtered_epochs, filtered_healthy,
                        marker=lora_healthy_config["marker"], color=lora_healthy_config["color"],
                        linewidth=4, markersize=12, markeredgecolor='white',
                        markeredgewidth=2, linestyle='--', dashes=(2, 1),
                        label=lora_healthy_config["label"]
                    )
        else:
            # Plot stroke curve for this variant
            config = variants_config.get(item)
            if config is None:
                continue
            try:
                data = load_convergence_data(item, participant, results_dir)
                plot_data = extract_plot_data(data)

                epochs = plot_data["epochs"]
                stroke_accs = plot_data["stroke_accs"]

                filtered_data = [(e, s) for e, s in zip(epochs, stroke_accs) if e >= min_epoch]
                if filtered_data:
                    filtered_epochs, filtered_stroke = zip(*filtered_data)
                else:
                    continue

                if stroke_zeroshot is None:
                    stroke_zeroshot = plot_data["frozen_stroke"]

                ax.plot(
                    filtered_epochs, filtered_stroke,
                    marker=config["marker"], color=config["color"],
                    linewidth=4, markersize=12, markeredgecolor='white',
                    markeredgewidth=2, label=config["label"]
                )

            except FileNotFoundError:
                print(f"Note: No data found for {item}/{participant}, skipping...")

    # Add reference lines with high-visibility colors
    if lora_plot_data is not None:
        frozen_healthy = lora_plot_data["frozen_healthy"]
        ax.axhline(
            y=frozen_healthy, color="#FF1493",  # Deep pink - highly visible
            linestyle=':', linewidth=3, alpha=1.0,
            label="Healthy Zero-shot Reference"
        )
        if stroke_zeroshot is None:
            stroke_zeroshot = lora_plot_data["frozen_stroke"]

    # Add stroke zero-shot reference line (bright, contrasting color)
    if stroke_zeroshot is not None:
        ax.axhline(
            y=stroke_zeroshot, color="#000000",  # Black - highly visible
            linestyle=':', linewidth=3, alpha=1.0,
            label="Stroke Zero-shot Reference"
        )

    # Configure axes - start x-axis at min_epoch
    ax.set_xlabel("Epoch", fontsize=26)
    ax.set_ylabel("Average Transition Accuracy", fontsize=22)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=min_epoch)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Remove top and right spines
    sns.despine(ax=ax)

    # Title
    subject_label = PARTICIPANT_LABELS.get(participant, participant)
    ax.set_title(
        f"Convergence Comparison ({subject_label})",
        fontsize=24, fontweight='semibold', pad=15
    )

    # Legend - 3 columns for compact layout
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=20,
        columnspacing=1.0,
        markerscale=1.3,
        handlelength=2.5
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38)

    # Save or show
    if output_path:
        output_path = Path(output_path)
    else:
        output_dir = Path(__file__).parent / "results" / "convergence"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"combined_convergence_{participant}.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Combined plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return str(output_path)


def plot_variant_comparison(
    participant: str = "p15",
    results_dir: str = None,
    output_path: str = None,
):
    """
    Create a two-column comparison plot for LoRA vs Full Finetune convergence.

    Args:
        participant: Participant ID (default: p15)
        results_dir: Path to results directory
        output_path: Path to save the plot
    """
    # Set seaborn style for clean, publication-quality plots
    sns.set_theme(style="white", context="paper", font_scale=1.2)

    # Use a clean color palette
    palette = sns.color_palette("deep", n_colors=2)
    stroke_color = palette[0]  # Blue
    healthy_color = palette[1]  # Orange

    variants = ["lora", "full_finetune"]
    variant_titles = {"lora": "LoRA", "full_finetune": "Full Fine-tune"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, variant in enumerate(variants):
        ax = axes[idx]

        try:
            data = load_convergence_data(variant, participant, results_dir)
            plot_data = extract_plot_data(data)
        except FileNotFoundError:
            print(f"Warning: No data found for {variant}/{participant}")
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(variant_titles[variant], fontsize=14, fontweight='semibold', pad=12)
            continue

        epochs = plot_data["epochs"]
        stroke_accs = plot_data["stroke_accs"]
        healthy_accs = plot_data["healthy_accs"]
        frozen_stroke = plot_data["frozen_stroke"]
        frozen_healthy = plot_data["frozen_healthy"]

        # Plot stroke performance
        ax.plot(epochs, stroke_accs, marker='o', color=stroke_color,
                linewidth=2.5, markersize=7, markeredgecolor='white',
                markeredgewidth=1.5, label="Stroke")

        # Plot healthy performance
        ax.plot(epochs, healthy_accs, marker='s', color=healthy_color,
                linewidth=2.5, markersize=7, markeredgecolor='white',
                markeredgewidth=1.5, label="Healthy")

        # Plot frozen baselines as horizontal dotted lines (same color as main lines)
        ax.axhline(y=frozen_stroke, color=stroke_color, linestyle=(0, (2, 1)),
                   linewidth=2, alpha=0.8, label="Frozen baseline (Stroke)")
        ax.axhline(y=frozen_healthy, color=healthy_color, linestyle=(0, (2, 1)),
                   linewidth=2, alpha=0.8, label="Frozen baseline (Healthy)")

        # Configure axes
        ax.set_xlabel("Epoch", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Transition Accuracy", fontsize=13)
        ax.set_title(variant_titles[variant], fontsize=14, fontweight='semibold', pad=12)
        ax.set_ylim(0, 1)

        # Remove top and right spines
        sns.despine(ax=ax)

        # Style tick labels
        ax.tick_params(axis='both', which='major', labelsize=11)

    # Add overall title
    subject_label = PARTICIPANT_LABELS.get(participant, participant)
    fig.suptitle(f"Convergence Comparison ({subject_label})", fontsize=16,
                 fontweight='semibold', y=1.02)

    # Single legend at the bottom, outside the axes, horizontal layout
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.5
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
        description="Analyze convergence experiment results and generate plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single variant analysis
    python3 analyze_convergence.py --variant lora --participant p4
    python3 analyze_convergence.py --variant lora --participant p15 --show
    python3 analyze_convergence.py -v head_only -p p20 --output my_plot.png

    # Compare LoRA vs Full Finetune (default: p15)
    python3 analyze_convergence.py --compare
    python3 analyze_convergence.py --compare -p p4 --output comparison_p4.png
    python3 analyze_convergence.py --compare -p p20

    # Combined plot: all variant stroke curves + LoRA healthy curve
    python3 analyze_convergence.py --combined -p p15
    python3 analyze_convergence.py --combined -p p15 --output combined_p15.png --show
        """
    )
    parser.add_argument(
        "--variant", "-v",
        default=None,
        choices=['stroke_only', 'head_only', 'lora', 'full_finetune'],
        help="Fine-tuning variant (stroke_only, head_only, lora, full_finetune)"
    )
    parser.add_argument(
        "--participant", "-p",
        default=None,
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
        help="Output path for the plot"
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Create two-column comparison plot (LoRA vs Full Finetune). Use with -p to specify participant."
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create combined plot with all variant stroke curves + LoRA healthy curve. Use with -p to specify participant."
    )

    args = parser.parse_args()

    if args.combined:
        # Create combined plot with all variants + LoRA healthy
        participant = args.participant if args.participant else "p15"
        create_combined_plot(
            participant=participant,
            results_dir=args.results_dir,
            output_path=args.output,
            show_plot=args.show,
        )
    elif args.compare:
        # Create comparison plot (default to p15 if no participant specified)
        participant = args.participant if args.participant else "p15"
        plot_variant_comparison(
            participant=participant,
            results_dir=args.results_dir,
            output_path=args.output,
        )
    else:
        # Single variant analysis
        if args.variant is None or args.participant is None:
            parser.error("--variant and --participant are required unless --compare or --combined is used")

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
