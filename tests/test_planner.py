"""Unit tests for benchmark/planner.py — no live database required."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.planner import CountProbeEstimator, execute_adaptive, select_strategy


# ---------------------------------------------------------------------------
# select_strategy — pure function, zero dependencies
# ---------------------------------------------------------------------------

def test_select_strategy_below_threshold():
    assert select_strategy(0.01, 0.05) == "B"


def test_select_strategy_above_threshold():
    assert select_strategy(0.10, 0.05) == "A"


def test_select_strategy_at_threshold():
    # sigma == threshold is NOT strictly less-than → strategy A
    assert select_strategy(0.05, 0.05) == "A"


def test_select_strategy_zero_selectivity():
    assert select_strategy(0.0, 0.05) == "B"


def test_select_strategy_full_selectivity():
    assert select_strategy(1.0, 0.05) == "A"


# ---------------------------------------------------------------------------
# CountProbeEstimator — mocks get_filtered_row_count from benchmark.db
# ---------------------------------------------------------------------------

def test_count_probe_estimator(mock_conn):
    with patch("benchmark.planner.get_filtered_row_count", return_value=500):
        estimator = CountProbeEstimator()
        sigma = estimator.estimate(
            conn=mock_conn,
            category="Laptop",
            max_price=50_000,
            min_rating=4.0,
            total_rows=50_000,
        )
    assert abs(sigma - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# execute_adaptive — mocked runner + estimator, checks strategy dispatch
# ---------------------------------------------------------------------------

def test_execute_adaptive_chooses_b(mock_runner, query_embedding):
    """When sigma < threshold the adaptive path must call run_strategy_b."""
    mock_estimator = type("E", (), {"estimate": lambda self, *a, **kw: 0.01})()
    result, sigma, strategy, probe_t = execute_adaptive(
        runner=mock_runner,
        query_embedding=query_embedding,
        category="Laptop",
        max_price=50_000,
        min_rating=4.0,
        total_rows=50_000,
        threshold=0.05,
        estimator=mock_estimator,
    )
    assert strategy == "B"
    mock_runner.run_strategy_b.assert_called_once()
    mock_runner.run_strategy_a.assert_not_called()


def test_execute_adaptive_chooses_a(mock_runner, query_embedding):
    """When sigma > threshold the adaptive path must call run_strategy_a."""
    mock_estimator = type("E", (), {"estimate": lambda self, *a, **kw: 0.50})()
    result, sigma, strategy, probe_t = execute_adaptive(
        runner=mock_runner,
        query_embedding=query_embedding,
        category=None,
        max_price=80_000,
        min_rating=0.0,
        total_rows=50_000,
        threshold=0.05,
        estimator=mock_estimator,
    )
    assert strategy == "A"
    mock_runner.run_strategy_a.assert_called_once()
    mock_runner.run_strategy_b.assert_not_called()
