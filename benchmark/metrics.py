"""
Metrics computation for HyBench.

Provides:
- compute_recall_at_k   : intersection-based recall relative to ground truth
- aggregate_latencies   : mean / median / P95 / IQR statistics
- format_stats_table    : Markdown table for reports
"""

from typing import Dict, List, Optional
import numpy as np


def compute_recall_at_k(
    result_ids: List[int],
    ground_truth_ids: List[int],
    k: int,
) -> float:
    """
    Recall@K = |result_ids[:k] ∩ ground_truth_ids[:k]| / k

    Both lists are treated as ordered; only the first k elements of each
    are considered.  If either list is shorter than k, the available
    elements are used.
    """
    if k == 0 or not ground_truth_ids:
        return 0.0
    gt_set = set(ground_truth_ids[:k])
    result_set = set(result_ids[:k])
    return len(gt_set & result_set) / k


def aggregate_latencies(latencies_s: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics over a list of per-query latencies (seconds).

    Returns a dict with keys:
        mean_ms, median_ms, p95_ms, p99_ms, iqr_ms, min_ms, max_ms, std_ms
    """
    arr = np.array(latencies_s) * 1000.0  # convert to milliseconds
    return {
        "mean_ms":   float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "iqr_ms":    float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min_ms":    float(np.min(arr)),
        "max_ms":    float(np.max(arr)),
        "std_ms":    float(np.std(arr)),
        "n_samples": len(latencies_s),
        "raw_ms":    arr.tolist(),
    }


def compute_speedup(stats_a: Dict, stats_b: Dict, metric: str = "mean_ms") -> float:
    """Return B_latency / A_latency.  > 1 means A is faster."""
    if stats_a.get(metric, 0) == 0:
        return float("inf")
    return stats_b[metric] / stats_a[metric]


def format_stats_table(
    rows: List[Dict],
    columns: Optional[List[str]] = None,
    header_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format a list of dicts as a GitHub-Flavoured Markdown table.

    Args:
        rows       : list of result dicts
        columns    : keys to include (default: all keys from first row)
        header_map : human-readable column header overrides
    """
    if not rows:
        return ""
    if columns is None:
        columns = list(rows[0].keys())
    if header_map is None:
        header_map = {}

    headers = [header_map.get(c, c) for c in columns]
    col_widths = [max(len(h), 8) for h in headers]

    # Determine column widths from data
    for row in rows:
        for i, col in enumerate(columns):
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.3f}"
            col_widths[i] = max(col_widths[i], len(str(val)))

    def fmt_row(values):
        return "| " + " | ".join(
            str(v).ljust(w) for v, w in zip(values, col_widths)
        ) + " |"

    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [fmt_row(headers), separator]
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.3f}"
            cells.append(str(val))
        lines.append(fmt_row(cells))

    return "\n".join(lines)


def latency_ratio_label(ratio: float) -> str:
    """Human-readable description of a latency ratio."""
    if ratio < 0.90:
        pct = (1 - ratio) * 100
        return f"Strategy A is {pct:.1f}% faster"
    if ratio > 1.10:
        pct = (ratio - 1) * 100
        return f"Strategy B is {pct:.1f}% faster"
    return "Strategies perform comparably"
