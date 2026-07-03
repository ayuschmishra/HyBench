"""Shared pytest fixtures for HyBench unit tests (no live DB required)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.runner import QueryResult


@pytest.fixture
def mock_conn():
    return MagicMock()


@pytest.fixture
def fake_result_a():
    return QueryResult(ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], latency_s=0.007, strategy="A")


@pytest.fixture
def fake_result_b():
    return QueryResult(ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], latency_s=0.003, strategy="B")


@pytest.fixture
def mock_runner(fake_result_a, fake_result_b):
    runner = MagicMock()
    runner.conn = MagicMock()
    runner.run_strategy_a.return_value = fake_result_a
    runner.run_strategy_b.return_value = fake_result_b
    return runner


@pytest.fixture
def mock_estimator():
    estimator = MagicMock()
    estimator.estimate.return_value = 0.01  # low selectivity → strategy B
    return estimator


@pytest.fixture
def query_embedding():
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec
