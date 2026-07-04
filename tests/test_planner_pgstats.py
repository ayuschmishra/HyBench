"""Unit tests for the v0.2 PgStatsEstimator and make_estimator factory.

No live database: pg_stats rows are supplied through a mocked cursor whose
fetchall() returns (attname, mcv_vals, mcv_freqs, histogram_bounds) tuples in
the same shape as the real catalog query.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.planner import (
    CountProbeEstimator,
    PgStatsEstimator,
    make_estimator,
)


# ---------------------------------------------------------------------------
# Helpers: build a mock connection whose single cursor returns given stats rows
# ---------------------------------------------------------------------------

def make_stats_conn(rows):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows
    return conn, cur


# category: 10 equal categories at 10% each (only two listed as MCV here).
# price:    equi-depth histogram over [0, 100000] in four bins.
# rating:   equi-depth histogram over [2.0, 5.0] in three bins.
STANDARD_ROWS = [
    ("category", ["Laptop", "Mouse"], [0.1, 0.1], None),
    ("price", None, [], ["0", "25000", "50000", "75000", "100000"]),
    ("rating", None, [], ["2.0", "3.0", "4.0", "5.0"]),
]


# ---------------------------------------------------------------------------
# Conjunctive selectivity estimation
# ---------------------------------------------------------------------------

def test_category_times_price_midpoint():
    # P(Laptop) = 0.10 ; P(price < 50000) = midpoint bin = 0.5 ; rating skipped.
    conn, _ = make_stats_conn(STANDARD_ROWS)
    est = PgStatsEstimator()
    sigma = est.estimate(conn, "Laptop", 50_000, 0.0, 50_000)
    assert abs(sigma - 0.05) < 1e-9


def test_price_above_all_bounds_is_full_selectivity():
    # value >= last histogram bound => fraction below is 1.0.
    conn, _ = make_stats_conn(STANDARD_ROWS)
    est = PgStatsEstimator()
    sigma = est.estimate(conn, None, 100_000, 0.0, 50_000)
    assert abs(sigma - 1.0) < 1e-9


def test_min_rating_uses_upper_tail():
    # P(rating > 4.0): rating=4.0 sits at bin boundary 2 of 3 => below=2/3,
    # so upper tail = 1/3. No category, price covers everything.
    conn, _ = make_stats_conn(STANDARD_ROWS)
    est = PgStatsEstimator()
    sigma = est.estimate(conn, None, 100_000, 4.0, 50_000)
    assert abs(sigma - (1.0 / 3.0)) < 1e-9


def test_category_not_in_mcv_falls_back_to_residual():
    # 'Camera' absent from MCV list => residual mass = 1 - (0.1 + 0.1) = 0.8.
    conn, _ = make_stats_conn(STANDARD_ROWS)
    est = PgStatsEstimator()
    sigma = est.estimate(conn, "Camera", 100_000, 0.0, 50_000)
    assert abs(sigma - 0.8) < 1e-9


def test_result_is_clamped_to_unit_interval():
    conn, _ = make_stats_conn(STANDARD_ROWS)
    est = PgStatsEstimator()
    sigma = est.estimate(conn, "Laptop", 1, 0.0, 50_000)  # price below all bounds
    assert 0.0 <= sigma <= 1.0


# ---------------------------------------------------------------------------
# Statistics are cached after the first estimate (the whole point vs. COUNT)
# ---------------------------------------------------------------------------

def test_stats_loaded_once_and_cached():
    conn, cur = make_stats_conn(STANDARD_ROWS)
    est = PgStatsEstimator()
    est.estimate(conn, "Laptop", 50_000, 0.0, 50_000)
    est.estimate(conn, "Mouse", 25_000, 0.0, 50_000)
    est.estimate(conn, None, 75_000, 0.0, 50_000)
    # Only the first estimate should have hit the catalog.
    assert cur.execute.call_count == 1


# ---------------------------------------------------------------------------
# Missing price statistics => actionable error (ANALYZE not run)
# ---------------------------------------------------------------------------

def test_missing_price_stats_raises():
    rows = [("category", ["Laptop"], [0.1], None)]  # no price row
    conn, _ = make_stats_conn(rows)
    est = PgStatsEstimator()
    with pytest.raises(RuntimeError, match="ANALYZE"):
        est.estimate(conn, "Laptop", 50_000, 0.0, 50_000)


# ---------------------------------------------------------------------------
# make_estimator factory
# ---------------------------------------------------------------------------

def test_make_estimator_count():
    assert isinstance(make_estimator("count"), CountProbeEstimator)


def test_make_estimator_pg_stats():
    assert isinstance(make_estimator("pg_stats"), PgStatsEstimator)
    assert isinstance(make_estimator("pgstats"), PgStatsEstimator)


def test_make_estimator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown estimator"):
        make_estimator("bogus")
