"""
Figure generation for HyBench v0.1.

Produces exactly two publication-quality figures:
  Figure 1: Filter Selectivity vs. Query Latency (RQ1)
  Figure 2: Adaptive Selector vs. Fixed Strategies (RQ2)

Usage:
    python analysis/plot_results.py          # both figures
    python analysis/plot_results.py --fig 1  # only Figure 1
    python analysis/plot_results.py --fig 2  # only Figure 2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Colourblind-safe palette (adapted from Wong 2011)
COL_A      = "#0072B2"   # blue   — Strategy A
COL_B      = "#E69F00"   # orange — Strategy B
COL_ADAPT  = "#009E73"   # green  — Adaptive
COL_ORACLE = "#000000"   # black  — Oracle

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi":     100,
})
STYLE = "seaborn-v0_8-whitegrid"


def load_json(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        print(f"[plot] Missing: {p}  (skipping)")
        return {}
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 — Filter Selectivity vs. Query Latency (RQ1)
# ---------------------------------------------------------------------------

def fig_01_latency_vs_selectivity(theta_star: float = None):
    data = load_json("exp_01_selectivity.json")
    if not data:
        return

    rows = sorted(data["results"], key=lambda r: r["actual_selectivity"])
    sels   = [r["actual_selectivity"] * 100 for r in rows]
    a_mean = [r["strategy_a"].get("mean_ms", 0) if r.get("strategy_a") else 0 for r in rows]
    a_std  = [r["strategy_a"].get("std_ms",  0) if r.get("strategy_a") else 0 for r in rows]
    b_mean = [r["strategy_b"].get("mean_ms", 0) if r.get("strategy_b") else 0 for r in rows]
    b_std  = [r["strategy_b"].get("std_ms",  0) if r.get("strategy_b") else 0 for r in rows]
    a_recall = [r["strategy_a"].get("recall_mean", 0) if r.get("strategy_a") else 0 for r in rows]
    b_recall = [r["strategy_b"].get("recall_mean", 0) if r.get("strategy_b") else 0 for r in rows]

    # Find crossover if theta_star not supplied
    if theta_star is None:
        for i in range(len(sels) - 1):
            if (a_mean[i] <= b_mean[i]) != (a_mean[i+1] <= b_mean[i+1]):
                ds = sels[i+1] - sels[i]
                diff_i   = a_mean[i]   - b_mean[i]
                diff_ip1 = a_mean[i+1] - b_mean[i+1]
                if abs(diff_i - diff_ip1) > 1e-9:
                    theta_star = sels[i] + ds * diff_i / (diff_i - diff_ip1)
                break

    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[3, 1.2])

        # --- Top subplot: Latency vs Selectivity ---
        ax1.errorbar(sels, a_mean, yerr=a_std, fmt="o-", color=COL_A,
                     label="Strategy A (vector-first)", capsize=3, linewidth=1.8)
        ax1.errorbar(sels, b_mean, yerr=b_std, fmt="s-", color=COL_B,
                     label="Strategy B (filter-first)", capsize=3, linewidth=1.8)

        if theta_star is not None:
            ax1.axvline(theta_star, linestyle="--", color="#CC79A7", linewidth=1.4,
                        label=f"θ* ≈ {theta_star:.1f}%")
            ax1.annotate(
                f"θ* ≈ {theta_star:.1f}%",
                xy=(theta_star, ax1.get_ylim()[1] * 0.95),
                xytext=(4, 0), textcoords="offset points",
                fontsize=8, color="#CC79A7",
            )

        ax1.set_xscale("log")
        ax1.set_xlabel("Filter Selectivity (%) — log scale")
        ax1.set_ylabel("Mean Query Latency (ms)\n(error bars: ±1 SD, n=50 queries)")
        ax1.set_title(
            "Figure 1 — Filter Selectivity vs. Query Latency (RQ1)\n"
            "50K rows · HNSW m=16 ef_search=40 · K=10 · "
            "Recall@K relative to filtered set"
        )
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax1.legend(loc="upper right")

        # Inset table: actual vs nominal selectivity (repositioned to avoid overlap)
        table_data = [
            [f"{r['target_selectivity']*100:.0f}%", f"{r['actual_selectivity']*100:.1f}%",
             f"{r['n_filtered']:,}"]
            for r in rows
        ]
        table = ax1.table(
            cellText=table_data,
            colLabels=["Target σ", "Actual σ", "Rows"],
            loc="center right",
            bbox=[0.60, 0.35, 0.36, 0.42],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)

        # --- Bottom subplot: Recall@K vs Selectivity ---
        ax2.plot(sels, a_recall, "o-", color=COL_A,
                 label="Strategy A", linewidth=1.8, markersize=5)
        ax2.plot(sels, b_recall, "s-", color=COL_B,
                 label="Strategy B", linewidth=1.8, markersize=5)
        ax2.set_xscale("log")
        ax2.set_xlabel("Filter Selectivity (%) — log scale")
        ax2.set_ylabel("Recall@10")
        ax2.set_ylim(-0.05, 1.15)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax2.legend(loc="lower right", fontsize=8)
        ax2.axhline(1.0, linestyle=":", color="grey", linewidth=0.8, alpha=0.5)

        fig.tight_layout()
        out = FIGURES_DIR / "fig_01_latency_vs_selectivity.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[plot] Figure 1 saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Adaptive Selector vs. Fixed Strategies (RQ2)
# ---------------------------------------------------------------------------

def fig_02_adaptive_vs_fixed():
    data = load_json("exp_02_adaptive.json")
    if not data:
        return

    rows = sorted(data["results"], key=lambda r: r["actual_selectivity"])
    sels    = [r["actual_selectivity"] * 100 for r in rows]
    a_mean  = [r["fixed_a"]["mean_ms"]   for r in rows]
    b_mean  = [r["fixed_b"]["mean_ms"]   for r in rows]
    ad_mean = [r["adaptive"]["mean_ms"]  for r in rows]
    or_mean = [r["oracle"]["mean_ms"]    for r in rows]
    # threshold stored under benchmark_config.threshold (as fraction); convert to %
    threshold_frac = data.get("benchmark_config", {}).get("threshold")
    theta = threshold_frac * 100 if threshold_frac is not None else None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(sels, a_mean,  "o--", color=COL_A,      label="Fixed-A (vector-first)",  linewidth=1.6)
        ax.plot(sels, b_mean,  "s--", color=COL_B,      label="Fixed-B (filter-first)",  linewidth=1.6)
        ax.plot(sels, ad_mean, "D-",  color=COL_ADAPT,  label="Adaptive (selector)",     linewidth=2.0)
        ax.plot(sels, or_mean, "^:",  color=COL_ORACLE,  label="Oracle (lower bound)",    linewidth=1.4)

        # Shaded adaptive gap
        ax.fill_between(sels, or_mean, ad_mean, alpha=0.12, color=COL_ADAPT,
                        label="Adaptive gap")

        if theta is not None:
            ax.axvline(theta, linestyle="--", color="#CC79A7", linewidth=1.4)
            ax.annotate(
                f"θ* ≈ {theta:.1f}%",
                xy=(theta, ax.get_ylim()[1] * 0.85),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color="#CC79A7", fontweight="bold",
            )

        ax.set_xlabel("Filter Selectivity (%)")
        ax.set_ylabel("Mean Query Latency (ms)")
        ax.set_title(
            "Figure 2 — Adaptive Selector vs. Fixed Strategies (RQ2)\n"
            "50K rows · HNSW m=16 ef_search=40 · K=10 · n=50 queries per point"
        )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(loc="upper right")
        fig.tight_layout()

        out = FIGURES_DIR / "fig_02_adaptive_vs_fixed.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[plot] Figure 2 saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HyBench v0.1 figures")
    parser.add_argument(
        "--fig", type=int, choices=[1, 2],
        help="Which figure to generate (1 or 2). Omit to generate both.",
    )
    parser.add_argument(
        "--theta", type=float, default=None,
        help="Crossover threshold theta* (percent) to annotate on Figure 1.",
    )
    args = parser.parse_args()

    if args.fig is None or args.fig == 1:
        fig_01_latency_vs_selectivity(theta_star=args.theta)
    if args.fig is None or args.fig == 2:
        fig_02_adaptive_vs_fixed()

    print(f"\n[plot] Done. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
