"""Unit tests for the v0.5 ef_search / probes overrides in BenchmarkRunner.

The Pareto sweep (exp_03) needs run_strategy_a to apply an EXACT ANN search
parameter, bypassing the v0.1/v0.2 floor that raises the parameter up to the
candidate-pool size. These tests confirm the override reaches the session GUC
and that omitting it preserves the legacy floor behaviour — all DB-free by
patching set_session_gucs and execute_timed in the runner module.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import BenchmarkConfig, DBConfig, HNSWConfig, IVFFlatConfig
from benchmark.runner import BenchmarkRunner


TOTAL_ROWS = 50_000
TOP_K = 10
# candidate_multiplier defaults to 100 → n_candidates = 1000.
N_CANDIDATES = TOP_K * 100


def _runner(index_type: str, total_rows: int = TOTAL_ROWS) -> BenchmarkRunner:
    cfg = BenchmarkConfig(
        index_type=index_type, hnsw=HNSWConfig(), ivfflat=IVFFlatConfig()
    )
    return BenchmarkRunner(MagicMock(name="conn"), cfg, DBConfig(), total_rows)


def _qvec() -> np.ndarray:
    v = np.zeros(384, dtype=np.float32)
    v[0] = 1.0
    return v


def _guc_kwargs(mock_set_gucs):
    """Extract the keyword args of the single set_session_gucs call."""
    assert mock_set_gucs.call_count == 1
    return mock_set_gucs.call_args.kwargs


# ---------------------------------------------------------------------------
# HNSW — hnsw.ef_search
# ---------------------------------------------------------------------------

@patch("benchmark.runner.execute_timed", return_value=([], 0.004))
@patch("benchmark.runner.set_session_gucs")
def test_hnsw_override_sets_exact_ef_search(mock_gucs, _mock_exec):
    _runner("hnsw").run_strategy_a(
        _qvec(), "Laptop", 100_000.0, 0.0, TOP_K, ef_search_override=32
    )
    kwargs = _guc_kwargs(mock_gucs)
    assert kwargs["hnsw_ef_search"] == 32       # exact swept value, not floored up
    assert kwargs["enable_seqscan"] is False


@patch("benchmark.runner.execute_timed", return_value=([], 0.004))
@patch("benchmark.runner.set_session_gucs")
def test_hnsw_no_override_floors_to_candidate_pool(mock_gucs, _mock_exec):
    # Legacy behaviour: ef_search = max(cfg.hnsw.ef_search=40, n_candidates=1000).
    _runner("hnsw").run_strategy_a(_qvec(), "Laptop", 100_000.0, 0.0, TOP_K)
    assert _guc_kwargs(mock_gucs)["hnsw_ef_search"] == N_CANDIDATES


@patch("benchmark.runner.execute_timed", return_value=([], 0.004))
@patch("benchmark.runner.set_session_gucs")
def test_hnsw_probes_override_ignored_for_hnsw(mock_gucs, _mock_exec):
    # A probes_override must not leak into the HNSW path.
    _runner("hnsw").run_strategy_a(
        _qvec(), "Laptop", 100_000.0, 0.0, TOP_K,
        ef_search_override=64, probes_override=99,
    )
    kwargs = _guc_kwargs(mock_gucs)
    assert kwargs["hnsw_ef_search"] == 64
    assert "ivfflat_probes" not in kwargs


# ---------------------------------------------------------------------------
# IVFFlat — ivfflat.probes
# ---------------------------------------------------------------------------

@patch("benchmark.runner.execute_timed", return_value=([], 0.004))
@patch("benchmark.runner.set_session_gucs")
def test_ivfflat_override_sets_exact_probes(mock_gucs, _mock_exec):
    _runner("ivfflat").run_strategy_a(
        _qvec(), "Laptop", 100_000.0, 0.0, TOP_K, probes_override=99
    )
    kwargs = _guc_kwargs(mock_gucs)
    assert kwargs["ivfflat_probes"] == 99       # exact swept value, not floored
    assert kwargs["enable_seqscan"] is False


@patch("benchmark.runner.execute_timed", return_value=([], 0.004))
@patch("benchmark.runner.set_session_gucs")
def test_ivfflat_no_override_uses_floor(mock_gucs, _mock_exec):
    # Derived defaults: lists=50 (50k/1000), probes=sqrt(50)=7;
    # min_probes = ceil(1000 * 50 / 50000) = 1 → effective = max(7, 1) = 7.
    _runner("ivfflat").run_strategy_a(_qvec(), "Laptop", 100_000.0, 0.0, TOP_K)
    assert _guc_kwargs(mock_gucs)["ivfflat_probes"] == 7


@patch("benchmark.runner.execute_timed", return_value=([], 0.004))
@patch("benchmark.runner.set_session_gucs")
def test_ivfflat_override_below_floor_still_honoured(mock_gucs, _mock_exec):
    # The whole point of the override is to reach values BELOW the floor so the
    # frontier's low-latency/low-recall end is reachable. probes_override=1
    # must win over the derived floor of 7.
    _runner("ivfflat").run_strategy_a(
        _qvec(), "Laptop", 100_000.0, 0.0, TOP_K, probes_override=1
    )
    assert _guc_kwargs(mock_gucs)["ivfflat_probes"] == 1
