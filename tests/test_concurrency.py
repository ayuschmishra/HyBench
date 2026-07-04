"""Unit tests for the v0.5 concurrent workload runner (benchmark/concurrency.py).

All DB-free: aggregate_concurrency is pure, and run_concurrent_workload takes
injectable connection/runner factories so the whole threaded path exercises
without a live PostgreSQL.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.concurrency import aggregate_concurrency, run_concurrent_workload
from benchmark.config import BenchmarkConfig, DBConfig
from benchmark.runner import QueryResult


# ---------------------------------------------------------------------------
# aggregate_concurrency — pure stats over per-query latencies
# ---------------------------------------------------------------------------

def test_aggregate_empty_returns_zeros():
    out = aggregate_concurrency([], wall_s=1.0)
    assert out["n_queries"] == 0
    assert out["qps"] == 0.0
    assert out["mean_ms"] == 0.0
    assert out["p99_ms"] == 0.0


def test_aggregate_converts_seconds_to_ms():
    # 10 ms and 20 ms as seconds → mean 15 ms.
    out = aggregate_concurrency([0.010, 0.020], wall_s=1.0)
    assert out["min_ms"] == pytest.approx(10.0)
    assert out["max_ms"] == pytest.approx(20.0)
    assert out["mean_ms"] == pytest.approx(15.0)
    assert out["median_ms"] == pytest.approx(15.0)


def test_aggregate_qps_is_queries_over_wall():
    # 8 queries completed across a 2 s timed phase → 4 QPS.
    out = aggregate_concurrency([0.001] * 8, wall_s=2.0)
    assert out["n_queries"] == 8
    assert out["qps"] == pytest.approx(4.0)


def test_aggregate_zero_wall_does_not_divide_by_zero():
    out = aggregate_concurrency([0.001, 0.002], wall_s=0.0)
    assert out["qps"] == 0.0
    assert out["n_queries"] == 2  # queries still counted; only QPS guarded


def test_aggregate_percentiles_are_ordered():
    lats = [x / 1000.0 for x in range(1, 101)]  # 1..100 ms
    out = aggregate_concurrency(lats, wall_s=1.0)
    assert out["median_ms"] <= out["p95_ms"] <= out["p99_ms"] <= out["max_ms"]


# ---------------------------------------------------------------------------
# Test doubles for the injectable concurrent path
# ---------------------------------------------------------------------------

class _FakeRunner:
    """Stands in for BenchmarkRunner: returns a fixed-latency QueryResult, or
    raises on every query when `fail` is set (to exercise error counting)."""

    def __init__(self, latency_s: float = 0.002, fail: bool = False):
        self.latency_s = latency_s
        self.fail = fail

    def run_strategy_a(self, qvec, cat, mp, mr, top_k, **kw):
        if self.fail:
            raise RuntimeError("simulated query failure")
        return QueryResult(ids=[1, 2, 3], latency_s=self.latency_s, strategy="A")

    run_strategy_b = run_strategy_a


def _make_factories(latency_s: float = 0.002, fail: bool = False):
    """Return (conn_factory, runner_factory, created_conns) for injection.

    created_conns lets a test assert every connection was closed.
    """
    created_conns = []

    def conn_factory(db_cfg):
        m = MagicMock(name="conn")
        created_conns.append(m)
        return m

    def runner_factory(conn, cfg, db_cfg, total_rows):
        return _FakeRunner(latency_s=latency_s, fail=fail)

    return conn_factory, runner_factory, created_conns


def _run(n_clients, *, latency_s=0.002, fail=False, n_warmup=1, n_queries=3):
    conn_factory, runner_factory, created = _make_factories(latency_s, fail)
    agg = run_concurrent_workload(
        db_cfg=DBConfig(),
        cfg=BenchmarkConfig(),
        total_rows=10_000,
        query_embeddings=[np.zeros(384, dtype=np.float32)],
        category="Laptop",
        max_price=100_000.0,
        min_rating=0.0,
        top_k=10,
        n_clients=n_clients,
        strategy="A",
        n_warmup=n_warmup,
        n_queries=n_queries,
        conn_factory=conn_factory,
        runner_factory=runner_factory,
    )
    return agg, created


# ---------------------------------------------------------------------------
# run_concurrent_workload — single client (no barrier)
# ---------------------------------------------------------------------------

def test_single_client_counts_only_timed_queries():
    agg, _ = _run(1, n_warmup=2, n_queries=3)
    # Warmup queries are discarded: 3 timed queries, not 5.
    assert agg["n_queries"] == 3
    assert agg["n_clients"] == 1
    assert agg["total_errors"] == 0
    assert len(agg["per_client"]) == 1
    assert agg["per_client"][0]["n_queries"] == 3


def test_single_client_closes_connection():
    _, created = _run(1)
    assert len(created) == 1
    assert created[0].close.called


# ---------------------------------------------------------------------------
# run_concurrent_workload — multi client (barrier path)
# ---------------------------------------------------------------------------

def test_multi_client_aggregates_all_queries():
    agg, created = _run(4, n_warmup=1, n_queries=3)
    assert agg["n_clients"] == 4
    assert agg["n_queries"] == 12          # 4 clients × 3 timed queries
    assert agg["total_errors"] == 0
    assert len(agg["per_client"]) == 4
    assert len(created) == 4                # one connection per client
    assert all(c.close.called for c in created)


def test_multi_client_qps_positive():
    agg, _ = _run(2, latency_s=0.001, n_queries=5)
    assert agg["qps"] > 0.0
    assert agg["mean_ms"] == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# run_concurrent_workload — error handling
# ---------------------------------------------------------------------------

def test_failing_queries_are_counted_not_raised():
    # With no warmup, every timed query raises and is caught → errors counted,
    # no latencies recorded, QPS falls back to 0 (empty timed phase).
    agg, created = _run(2, fail=True, n_warmup=0, n_queries=4)
    assert agg["n_queries"] == 0
    assert agg["total_errors"] == 8         # 2 clients × 4 failed queries
    assert agg["qps"] == 0.0
    assert all(c.close.called for c in created)  # cleanup still runs
