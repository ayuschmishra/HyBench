"""
Statistical summary generator for HyBench v0.1.

Reads exp_01 and exp_02 JSON results and prints a Markdown table
summary ready to paste into the report.

Usage:
    python analysis/summarize.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.metrics import format_stats_table, latency_ratio_label

RESULTS_DIR = Path("results")


def load(name: str) -> dict:
    p = RESULTS_DIR / name
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def summarize_exp01():
    data = load("exp_01_selectivity.json")
    if not data:
        print("## Experiment 1  [NOT RUN]\n")
        return

    print("## Experiment 1 — Filter Selectivity vs. Latency (RQ1)\n")
    rows = sorted(data["results"], key=lambda r: r["actual_selectivity"])
    table_rows = []
    for r in rows:
        a = r.get("strategy_a") or {}
        b = r.get("strategy_b") or {}
        a_mean = a.get("mean_ms", 0)
        b_mean = b.get("mean_ms", 0)
        ratio  = b_mean / max(a_mean, 0.001)
        table_rows.append({
            "Target σ":      f"{r['target_selectivity']*100:.0f}%",
            "Actual σ":      f"{r['actual_selectivity']*100:.1f}%",
            "Filtered rows": r["n_filtered"],
            "A mean (ms)":   round(a_mean, 1),
            "A P95 (ms)":    round(a.get("p95_ms", 0), 1),
            "B mean (ms)":   round(b_mean, 1),
            "B P95 (ms)":    round(b.get("p95_ms", 0), 1),
            "B/A ratio":     round(ratio, 2),
            "Winner":        "A" if a_mean < b_mean else "B",
        })
    print(format_stats_table(table_rows))
    print()


def summarize_exp02():
    data = load("exp_02_adaptive.json")
    if not data:
        print("## Experiment 2  [NOT RUN]\n")
        return

    theta_raw = data.get("benchmark_config", {}).get("threshold", None)
    theta_str = f"{theta_raw*100:.1f}" if isinstance(theta_raw, (int, float)) else "?"
    print(f"## Experiment 2 — Adaptive Selector Evaluation (RQ2)  [θ* = {theta_str}%]\n")
    rows = sorted(data["results"], key=lambda r: r["actual_selectivity"])
    table_rows = []
    for r in rows:
        fa = r["fixed_a"]["mean_ms"]
        fb = r["fixed_b"]["mean_ms"]
        ad = r["adaptive"]["mean_ms"]
        oc = r["oracle"]["mean_ms"]
        gap_pct = 100 * (ad - oc) / max(oc, 0.001)
        probe_stats = r.get("probe_overhead_ms", {})
        probe_ms = probe_stats.get("mean_ms", None)
        choices = r.get("adaptive_choices", {})
        decision = f"A:{choices.get('A', '?')}/B:{choices.get('B', '?')}"
        table_rows.append({
            "Actual σ":       f"{r['actual_selectivity']*100:.1f}%",
            "Fixed-A (ms)":   round(fa, 1),
            "Fixed-B (ms)":   round(fb, 1),
            "Adaptive (ms)":  round(ad, 1),
            "Oracle (ms)":    round(oc, 1),
            "Gap %":          f"{gap_pct:.1f}%",
            "Probe (ms)":     round(probe_ms, 2) if probe_ms is not None else "—",
            "Decision":       decision,
        })
    print(format_stats_table(table_rows))

    # Summary stats
    gaps = [100 * (r["adaptive"]["mean_ms"] - r["oracle"]["mean_ms"]) / max(r["oracle"]["mean_ms"], 0.001)
            for r in rows]
    print(f"\n  Mean adaptive gap: {sum(gaps)/len(gaps):.1f}%")
    print(f"  Max  adaptive gap: {max(gaps):.1f}%")
    print()


def main():
    print("# HyBench v0.1 — Experimental Results Summary\n")
    summarize_exp01()
    summarize_exp02()


if __name__ == "__main__":
    main()
