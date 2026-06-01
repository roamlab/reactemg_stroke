#!/usr/bin/env python3
"""
Grouped bar chart of Table II main results.

Groups = participants (S1, S2, S3, Avg). Bars within group = adaptation strategies.
Two subplots: Raw Accuracy and Transition Accuracy. Mean only.

Usage:
    python3 plot_main_results.py
    python3 plot_main_results.py -o my_figure.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from extract_results import (
    build_table,
    VARIANTS,
    VARIANT_LABELS,
    PARTICIPANTS,
    PARTICIPANT_LABELS,
    METRICS,
)


VARIANT_COLORS = {
    "zero_shot":     "#888888",
    "stroke_only":   "#F4A261",
    "head_only":     "#457B9D",
    "lora":          "#2A9D8F",
    "full_finetune": "#E63946",
}


def plot_bars(table, output_path: Path):
    sns.set_theme(style="white", context="paper", font_scale=1.35)
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)

    group_labels = [PARTICIPANT_LABELS[p] for p in PARTICIPANTS] + ["Avg"]
    group_positions = np.arange(len(group_labels))
    bar_w = 0.16

    for ax_idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[ax_idx]

        for v_idx, variant in enumerate(VARIANTS):
            values = []
            for p in PARTICIPANTS:
                mean, _ = table[metric_key][variant][p]
                values.append(mean if mean is not None else 0.0)
            avg_mean, _ = table[metric_key][variant]["avg"]
            values.append(avg_mean if avg_mean is not None else 0.0)

            offset = (v_idx - (len(VARIANTS) - 1) / 2) * bar_w
            ax.bar(
                group_positions + offset, values,
                width=bar_w,
                color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant] if ax_idx == 0 else None,
                edgecolor="white",
                linewidth=0.6,
            )

        ax.set_xticks(group_positions)
        ax.set_xticklabels(group_labels)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric_label, fontsize=17, fontstyle="italic", pad=12, color="#333333")
        if ax_idx == 0:
            ax.set_ylabel("Accuracy", fontsize=15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        sns.despine(ax=ax)
        ax.tick_params(axis="both", which="major", labelsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(VARIANTS),
        frameon=False,
        fontsize=14,
        columnspacing=1.8,
        handlelength=1.6,
    )

    fig.tight_layout(pad=0.4, w_pad=0.5)
    fig.subplots_adjust(bottom=0.16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grouped bar chart of Table II main results")
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Path to main_experiment results (default: ./results/main_experiment)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: ./results/main_experiment/table2_bars.png)",
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
        else Path(__file__).parent / "results" / "main_experiment" / "table2_bars.png"
    )

    table = build_table(results_dir)
    plot_bars(table, output_path)


if __name__ == "__main__":
    main()
