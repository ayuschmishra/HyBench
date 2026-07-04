"""
Concurrent multi-client workload runner for HyBench v0.5.

Experiment 4 characterises how hybrid-query throughput (QPS) and tail latency
degrade as concurrent client count rises. PostgreSQL/pgvector serve each client
on its own backend; HNSW graph traversal takes a shared read lock, so aggregate
throughput does not scale linearly with clients. This module runs N independent
client threads — each with its own psycopg2 connection and BenchmarkRunner —
issuing the same fixed workload, then aggregates QPS + latency percentiles.

Threads (not processes) suffice because psycopg2 releases the GIL during the
libpq socket wait, so DB-bound work runs concurrently. Each connection is used
by exactly one thread (psycopg2 connections are not safe to share across
threads).

The connection/runner factories are injectable so the whole concurrent path is
unit-testable without a live database (see tests/test_concurrency.py).
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from benchmark.config import BenchmarkConfig, DBConfig
from benchmark.db import get_connection
from benchmark.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ClientResult:
    client_id: int
    latencies_s: List[float] = field(default_factory=list)
    t_start: float = 0.0          # perf_counter at first timed query
    t_end: float = 0.0            # perf_counter after last timed query
    errors: int = 0

    @property
    def n_queries(self) -> int:
        return len(self.latencies_s)


# ---------------------------------------------------------------------------
# Pure aggregation (DB-free, trivially testable)
# ---------------------------------------------------------------------------

def aggregate_concurrency(latencies_s: List[float], wall_s: float) -> dict:
    """Aggregate per-query latencies + timed-phase wall clock into QPS + stats.

    QPS is total completed queries divided by the wall-clock span of the timed
    phase (max client end − min client start), i.e. real overlapping throughput
    rather than the sum of per-client rates.
    """
    arr = np.asarray(latencies_s, dtype=float) * 1000.0  # → milliseconds
    n = int(arr.size)
    qps = (n / wall_s) if wall_s > 0 else 0.0
    if n == 0:
        return {
            "n_queries": 0, "wall_s": wall_s, "qps": 0.0,
            "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0,
            "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
        }
    return {
        "n_queries": n,
        "wall_s": wall_s,
        "qps": qps,
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Client worker
# ---------------------------------------------------------------------------

def _client_worker(
    client_id: int,
    db_cfg: DBConfig,
    cfg: BenchmarkConfig,
    total_rows: int,
    query_embeddings: List[np.ndarray],
    category: Optional[str],
    max_price: float,
    min_rating: float,
    top_k: int,
    strategy: str,
    n_warmup: int,
    n_queries: int,
    barrier: Optional[threading.Barrier],
    conn_factory: Callable,
    runner_factory: Callable,
) -> ClientResult:
    """One client: own connection, warmup (discarded), then timed queries.

    A shared Barrier releases every client into its timed phase together so the
    measured wall clock reflects genuine concurrent load, not warmup skew.
    """
    conn = conn_factory(db_cfg)
    try:
        runner = runner_factory(conn, cfg, db_cfg, total_rows)
        run_fn = runner.run_strategy_a if strategy == "A" else runner.run_strategy_b
        pool_n = max(len(query_embeddings), 1)

        for i in range(n_warmup):
            run_fn(query_embeddings[i % pool_n], category, max_price, min_rating, top_k)

        # All clients start the timed phase at the same instant.
        if barrier is not None:
            barrier.wait()

        result = ClientResult(client_id=client_id)
        result.t_start = time.perf_counter()
        for i in range(n_queries):
            qvec = query_embeddings[i % pool_n]
            try:
                res = run_fn(qvec, category, max_price, min_rating, top_k)
                result.latencies_s.append(res.latency_s)
            except Exception:
                result.errors += 1
        result.t_end = time.perf_counter()
        return result
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Concurrent workload driver
# ---------------------------------------------------------------------------

def run_concurrent_workload(
    db_cfg: DBConfig,
    cfg: BenchmarkConfig,
    total_rows: int,
    query_embeddings: List[np.ndarray],
    category: Optional[str],
    max_price: float,
    min_rating: float,
    top_k: int,
    n_clients: int,
    strategy: str = "A",
    n_warmup: int = 5,
    n_queries: int = 50,
    conn_factory: Callable = get_connection,
    runner_factory: Callable = BenchmarkRunner,
) -> dict:
    """Run `n_clients` concurrent clients and return aggregated stats.

    Returns a dict with the pooled `aggregate_concurrency` metrics plus a
    `per_client` breakdown and total error count. QPS uses the timed-phase wall
    clock (max end − min start across clients).
    """
    barrier = threading.Barrier(n_clients) if n_clients > 1 else None

    with ThreadPoolExecutor(max_workers=n_clients) as pool:
        futures = [
            pool.submit(
                _client_worker,
                cid, db_cfg, cfg, total_rows, query_embeddings,
                category, max_price, min_rating, top_k, strategy,
                n_warmup, n_queries, barrier, conn_factory, runner_factory,
            )
            for cid in range(n_clients)
        ]
        client_results = [f.result() for f in futures]

    all_latencies: List[float] = []
    for cr in client_results:
        all_latencies.extend(cr.latencies_s)

    starts = [cr.t_start for cr in client_results if cr.n_queries > 0]
    ends = [cr.t_end for cr in client_results if cr.n_queries > 0]
    wall_s = (max(ends) - min(starts)) if starts and ends else 0.0

    agg = aggregate_concurrency(all_latencies, wall_s)
    agg["n_clients"] = n_clients
    agg["total_errors"] = sum(cr.errors for cr in client_results)
    agg["per_client"] = [
        {
            "client_id": cr.client_id,
            "n_queries": cr.n_queries,
            "errors": cr.errors,
            "mean_ms": float(np.mean(np.asarray(cr.latencies_s) * 1000.0))
            if cr.n_queries else 0.0,
        }
        for cr in client_results
    ]
    return agg
