"""Unit tests for benchmark/metrics.py — pure computation, no DB required."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.metrics import aggregate_latencies, compute_recall_at_k


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_partial_overlap():
    result_ids   = [1, 2, 3]
    ground_truth = [1, 2, 5]
    recall = compute_recall_at_k(result_ids, ground_truth, k=3)
    assert abs(recall - 2 / 3) < 1e-9


def test_recall_at_k_perfect():
    ids = [1, 2, 3, 4, 5]
    assert compute_recall_at_k(ids, ids, k=5) == 1.0


def test_recall_at_k_no_overlap():
    assert compute_recall_at_k([1, 2, 3], [4, 5, 6], k=3) == 0.0


# ---------------------------------------------------------------------------
# aggregate_latencies
# ---------------------------------------------------------------------------

def test_latency_mean():
    latencies_s = [0.001, 0.002, 0.003]  # 1ms, 2ms, 3ms
    stats = aggregate_latencies(latencies_s)
    assert abs(stats["mean_ms"] - 2.0) < 1e-6


def test_latency_std_all_equal():
    latencies_s = [0.005] * 20
    stats = aggregate_latencies(latencies_s)
    assert stats["std_ms"] < 1e-9


def test_latency_p95():
    # 100 latencies: 1ms .. 100ms
    latencies_s = [i / 1000 for i in range(1, 101)]
    stats = aggregate_latencies(latencies_s)
    # numpy's 95th percentile on [1..100] ms
    expected = float(np.percentile(np.array(latencies_s) * 1000, 95))
    assert abs(stats["p95_ms"] - expected) < 1e-6
