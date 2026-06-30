"""
Lightweight Selectivity-Aware Execution Strategy Selector.

Primary systems contribution of HyBench v0.1.

Design:
- Deterministic: same selectivity estimate → same strategy choice, always.
- Explainable: every decision is a single comparison σ < θ*.
- Single-parameter: θ* is one calibrated constant, not a learned weight.
- v0.1 uses a COUNT(*) probe for selectivity estimation; pg_stats MCV-based
  estimation is the natural v0.2 replacement (see CountProbeEstimator docstring).
"""

from __future__ import annotations

import time
from typing import Optional, Protocol

import numpy as np

from benchmark.config import ADAPTIVE_THRESHOLD
from benchmark.db import get_filtered_row_count, set_session_gucs
from benchmark.runner import BenchmarkRunner, QueryResult


# ---------------------------------------------------------------------------
# Selectivity estimation
# ---------------------------------------------------------------------------

class SelectivityEstimator(Protocol):
    """Protocol so v0.2 can swap in a pg_stats-based estimator."""

    def estimate(
        self,
        conn,
        category: Optional[str],
        max_price: float,
        min_rating: float,
        total_rows: int,
    ) -> float: ...


class CountProbeEstimator:
    """Estimate filter selectivity via SELECT COUNT(*).

    Intentionally simple: chosen for prototype clarity and reproducibility.
    Adds one round-trip per query (~1–5 ms on a local connection).

    In a production system, reading `pg_stats.most_common_vals` and
    `most_common_freqs` would estimate selectivity without a probe query.
    That optimisation is deferred to HyBench v0.2 and documented as the
    natural extension point via this protocol.
    """

    def estimate(
        self,
        conn,
        category: Optional[str],
        max_price: float,
        min_rating: float,
        total_rows: int,
    ) -> float:
        n = get_filtered_row_count(conn, category, max_price, min_rating)
        return n / max(total_rows, 1)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def select_strategy(sigma: float, threshold: float = ADAPTIVE_THRESHOLD) -> str:
    """Return 'B' when sigma < threshold, 'A' otherwise.

    Deterministic. Explainable. Parametrised by a single constant θ*.
    """
    return "B" if sigma < threshold else "A"


# ---------------------------------------------------------------------------
# Adaptive execution
# ---------------------------------------------------------------------------

def execute_adaptive(
    runner: BenchmarkRunner,
    query_embedding: np.ndarray,
    category: Optional[str],
    max_price: float,
    min_rating: float,
    total_rows: int,
    top_k: int = 10,
    threshold: float = ADAPTIVE_THRESHOLD,
    estimator: Optional[SelectivityEstimator] = None,
) -> tuple[QueryResult, float, str, float]:
    """Run one hybrid query using the adaptive strategy selector.

    Returns
    -------
    result       : QueryResult from the selected strategy
    sigma        : estimated filter selectivity
    strategy     : 'A' or 'B'
    probe_time_s : wall-clock time for the COUNT(*) probe
    """
    if estimator is None:
        estimator = CountProbeEstimator()

    t_probe_start = time.perf_counter()
    sigma = estimator.estimate(
        runner.conn, category, max_price, min_rating, total_rows
    )
    probe_time_s = time.perf_counter() - t_probe_start

    strategy = select_strategy(sigma, threshold)

    if strategy == "A":
        result = runner.run_strategy_a(
            query_embedding, category, max_price, min_rating, top_k
        )
    else:
        result = runner.run_strategy_b(
            query_embedding, category, max_price, min_rating, top_k
        )

    return result, sigma, strategy, probe_time_s
