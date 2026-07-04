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


class PgStatsEstimator:
    """Estimate filter selectivity from PostgreSQL's pg_stats — no probe query.

    v0.2 replacement for the COUNT(*) probe, implementing the extension point
    documented in the SelectivityEstimator protocol. Statistics for
    products.(category, price, rating) are fetched once per instance and
    cached; every subsequent estimate is pure Python arithmetic (microseconds
    instead of the probe's ~1–5 ms round-trip).

    Estimation mirrors PostgreSQL's own planner for a conjunctive predicate:
    per-column selectivities from most-common-value lists (categorical) and
    equi-depth histograms (continuous), combined under the
    attribute-independence assumption:

        sel = sel(category = c) * sel(price < p) * sel(rating > r)

    HyBench's price ranges are category-dependent, so independence is
    deliberately wrong in the same way real planner estimates are wrong —
    quantifying the selector-decision impact of that estimation error versus
    the exact COUNT(*) probe is the point of the v0.2 comparison.

    Requires ANALYZE to have populated pg_stats (the data generator and
    ensure_vector_index both run it).
    """

    # anyarray columns can't be adapted by psycopg2; go via text.
    _STATS_SQL = """
        SELECT attname,
               most_common_vals::text::text[]  AS mcv_vals,
               most_common_freqs               AS mcv_freqs,
               histogram_bounds::text::text[]  AS hist_bounds
        FROM pg_stats
        WHERE schemaname = 'public'
          AND tablename  = 'products'
          AND attname IN ('category', 'price', 'rating');
    """

    def __init__(self) -> None:
        self._stats: Optional[dict] = None

    def _load_stats(self, conn) -> dict:
        with conn.cursor() as cur:
            cur.execute(self._STATS_SQL)
            rows = cur.fetchall()
        stats = {
            attname: {
                "mcv_vals": mcv_vals or [],
                "mcv_freqs": mcv_freqs or [],
                "hist": hist or [],
            }
            for attname, mcv_vals, mcv_freqs, hist in rows
        }
        if "price" not in stats:
            raise RuntimeError(
                "pg_stats has no statistics for products.price — "
                "run `ANALYZE products;` before using PgStatsEstimator."
            )
        return stats

    @staticmethod
    def _fraction_below(value: float, col_stats: dict) -> float:
        """P(column < value) from MCV list + equi-depth histogram."""
        freqs = col_stats["mcv_freqs"]
        mcv_total = sum(freqs)
        sel = sum(
            f for v, f in zip(col_stats["mcv_vals"], freqs) if float(v) < value
        )
        bounds = [float(b) for b in col_stats["hist"]]
        if len(bounds) >= 2:
            if value <= bounds[0]:
                hist_frac = 0.0
            elif value >= bounds[-1]:
                hist_frac = 1.0
            else:
                # Equi-depth bins: each of the (len-1) bins holds an equal
                # share of the non-MCV rows; interpolate within the bin.
                i = max(j for j in range(len(bounds) - 1) if bounds[j] <= value)
                width = bounds[i + 1] - bounds[i]
                within = (value - bounds[i]) / width if width > 0 else 0.0
                hist_frac = (i + within) / (len(bounds) - 1)
            sel += hist_frac * (1.0 - mcv_total)
        return min(max(sel, 0.0), 1.0)

    def estimate(
        self,
        conn,
        category: Optional[str],
        max_price: float,
        min_rating: float,
        total_rows: int,
    ) -> float:
        if self._stats is None:
            self._stats = self._load_stats(conn)

        sel = 1.0

        if category is not None:
            cat_stats = self._stats.get("category", {"mcv_vals": [], "mcv_freqs": []})
            matched = [
                f
                for v, f in zip(cat_stats["mcv_vals"], cat_stats["mcv_freqs"])
                if v == category
            ]
            if matched:
                sel *= matched[0]
            else:
                # Not in the MCV list: fall back to the residual (non-MCV)
                # frequency mass — a coarse upper bound, same as the planner's
                # treatment of values outside the MCV list.
                sel *= max(0.0, 1.0 - sum(cat_stats["mcv_freqs"]))

        sel *= self._fraction_below(max_price, self._stats["price"])

        if min_rating > 0 and "rating" in self._stats:
            sel *= 1.0 - self._fraction_below(min_rating, self._stats["rating"])

        return min(max(sel, 0.0), 1.0)


def make_estimator(kind: str) -> SelectivityEstimator:
    """Factory for CLI-selectable estimators: 'count' or 'pg_stats'."""
    if kind == "count":
        return CountProbeEstimator()
    if kind in ("pg_stats", "pgstats"):
        return PgStatsEstimator()
    raise ValueError(f"Unknown estimator kind: {kind!r} (expected 'count' or 'pg_stats')")


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
