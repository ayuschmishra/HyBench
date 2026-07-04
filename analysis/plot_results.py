"""
Figure generation for HyBench.

Produces publication-quality figures:
  Figure 1: Filter Selectivity vs. Query Latency (RQ1)
  Figure 2: Adaptive Selector vs. Fixed Strategies (RQ2)
  Figure 3: ANN Index Comparison — HNSW vs. IVFFlat (v0.2; requires
            results/exp_01_selectivity_ivfflat.json, skipped otherwise)

Usage:
    python analysis/plot_results.py          # all figures with available inputs
    python analysis/plot_results.py --fig 1  # only Figure 1
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
COL_A        = "#0072B2"   # blue       — Strategy A
COL_B        = "#E69F00"   # orange     — Strategy B
COL_ADAPT    = "#009E73"   # green      — Adaptive (COUNT probe)
COL_ADAPT_PG = "#56B4E9"   # sky blue   — Adaptive (pg_stats)      [v0.2]
COL_IVF      = "#D55E00"   # vermillion — IVFFlat                  [v0.2]
COL_ORACLE   = "#000000"   # black      — Oracle


def _run_context(data: dict) -> str:
    """Shared title suffix derived from the run's own config metadata."""
    cfg = data.get("benchmark_config", {})
    n_rows_k = cfg.get("n_rows", 50_000) // 1000
    top_k = cfg.get("top_k", 10)
    ef_eff = max(cfg.get("ef_search", 40), top_k * 100)
    return f"{n_rows_k}K rows · HNSW m=16 · Strategy A ef_search={ef_eff} · K={top_k}"

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
            f"{_run_context(data)} · Recall@K relative to filtered set"
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
    or_mean = [r["oracle"]["mean_ms"]    for r in rows]

    def _series(key):
        return [r[key]["mean_ms"] for r in rows] if all(key in r for r in rows) else None

    ad_mean  = _series("adaptive")           # COUNT-probe estimator
    adp_mean = _series("adaptive_pgstats")   # pg_stats estimator (v0.2)
    # threshold stored under benchmark_config.threshold (as fraction); convert to %
    threshold_frac = data.get("benchmark_config", {}).get("threshold")
    theta = threshold_frac * 100 if threshold_frac is not None else None

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(sels, a_mean,  "o--", color=COL_A,      label="Fixed-A (vector-first)",  linewidth=1.6)
        ax.plot(sels, b_mean,  "s--", color=COL_B,      label="Fixed-B (filter-first)",  linewidth=1.6)
        if ad_mean is not None:
            ax.plot(sels, ad_mean, "D-", color=COL_ADAPT,
                    label="Adaptive (COUNT probe)", linewidth=2.0)
        if adp_mean is not None:
            ax.plot(sels, adp_mean, "v-", color=COL_ADAPT_PG,
                    label="Adaptive (pg_stats)", linewidth=1.8)
        ax.plot(sels, or_mean, "^:",  color=COL_ORACLE,  label="Oracle (lower bound)",    linewidth=1.4)

        # Shaded adaptive gap (against whichever adaptive series exists)
        gap_series = ad_mean if ad_mean is not None else adp_mean
        if gap_series is not None:
            ax.fill_between(sels, or_mean, gap_series, alpha=0.12, color=COL_ADAPT,
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
        n_q = data.get("benchmark_config", {}).get("n_queries", 50)
        ax.set_title(
            "Figure 2 — Adaptive Selector vs. Fixed Strategies (RQ2)\n"
            f"{_run_context(data)} · n={n_q} queries per point"
        )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(loc="upper right")
        fig.tight_layout()

        out = FIGURES_DIR / "fig_02_adaptive_vs_fixed.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[plot] Figure 2 saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 — ANN Index Comparison: HNSW vs. IVFFlat (RQ3, v0.2)
# ---------------------------------------------------------------------------

def fig_03_index_comparison():
    """Strategy A latency + Recall@K for HNSW vs. IVFFlat across selectivity.

    Requires both results/exp_01_selectivity.json (HNSW) and
    results/exp_01_selectivity_ivfflat.json. Skips silently if the IVFFlat
    run is absent, so the default HNSW-only pipeline is unaffected.
    """
    hnsw = load_json("exp_01_selectivity.json")
    ivf = load_json("exp_01_selectivity_ivfflat.json")
    if not hnsw or not ivf:
        print("[plot] Figure 3 skipped (need both HNSW and IVFFlat exp_01 results)")
        return

    def extract(data):
        rows = sorted(data["results"], key=lambda r: r["actual_selectivity"])
        return (
            [r["actual_selectivity"] * 100 for r in rows],
            [r["strategy_a"].get("mean_ms", 0) for r in rows],
            [r["strategy_a"].get("std_ms", 0) for r in rows],
            [r["strategy_a"].get("recall_mean", 0) for r in rows],
        )

    h_sel, h_lat, h_std, h_rec = extract(hnsw)
    i_sel, i_lat, i_std, i_rec = extract(ivf)

    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), height_ratios=[3, 1.2])

        # --- Top: Strategy A latency, HNSW vs IVFFlat ---
        ax1.errorbar(h_sel, h_lat, yerr=h_std, fmt="o-", color=COL_A,
                     label="HNSW (m=16, ef_c=64)", capsize=3, linewidth=1.8)
        ax1.errorbar(i_sel, i_lat, yerr=i_std, fmt="s-", color=COL_IVF,
                     label="IVFFlat (lists≈N/1000)", capsize=3, linewidth=1.8)
        ax1.set_xscale("log")
        ax1.set_xlabel("Filter Selectivity (%) — log scale")
        ax1.set_ylabel("Strategy A Mean Latency (ms)\n(error bars: ±1 SD)")
        h_ctx = _run_context(hnsw)
        ax1.set_title(
            "Figure 3 — ANN Index Comparison: HNSW vs. IVFFlat (Strategy A)\n"
            f"{h_ctx} · vector-first with relational post-filter"
        )
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax1.legend(loc="upper right")

        # --- Bottom: Recall@K, HNSW vs IVFFlat ---
        ax2.plot(h_sel, h_rec, "o-", color=COL_A, label="HNSW", linewidth=1.8, markersize=5)
        ax2.plot(i_sel, i_rec, "s-", color=COL_IVF, label="IVFFlat", linewidth=1.8, markersize=5)
        ax2.set_xscale("log")
        ax2.set_xlabel("Filter Selectivity (%) — log scale")
        ax2.set_ylabel("Recall@10")
        ax2.set_ylim(-0.05, 1.15)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax2.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax2.axhline(1.0, linestyle=":", color="grey", linewidth=0.8, alpha=0.5)
        ax2.legend(loc="lower right", fontsize=8)

        fig.tight_layout()
        out = FIGURES_DIR / "fig_03_index_comparison.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"[plot] Figure 3 saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HyBench figures")
    parser.add_argument(
        "--fig", type=int, choices=[1, 2, 3],
        help="Which figure to generate (1, 2, or 3). Omit to generate all available.",
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
    if args.fig is None or args.fig == 3:
        fig_03_index_comparison()

    print(f"\n[plot] Done. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
