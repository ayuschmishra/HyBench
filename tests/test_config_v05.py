"""Unit tests for v0.5 sweep configs in benchmark/config.py.

Covers ParetoConfig (exp_03) and ConcurrencyConfig (exp_04) defaults, plus the
module-level default lists that experiment scripts and tests share.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import (
    CONCURRENCY_CLIENT_COUNTS,
    PARETO_EF_SEARCH_VALUES,
    PARETO_PROBES_VALUES,
    SELECTIVITY_CONFIGS,
    ConcurrencyConfig,
    ParetoConfig,
)


# ---------------------------------------------------------------------------
# ParetoConfig — recall-latency sweep knobs (exp_03)
# ---------------------------------------------------------------------------

def test_pareto_defaults():
    p = ParetoConfig()
    assert p.selectivity_levels == [0.01, 0.10]
    assert p.ef_search_values == [10, 20, 40, 80, 160, 320, 640]
    assert p.probes_values == [1, 2, 5, 10, 20, 50, 100]


def test_pareto_sweep_values_strictly_increasing():
    # A monotone sweep is what draws a clean frontier: each larger value should
    # buy more recall at more latency. Non-monotone lists would tangle Figure 4.
    p = ParetoConfig()
    assert p.ef_search_values == sorted(p.ef_search_values)
    assert p.probes_values == sorted(p.probes_values)
    assert len(set(p.ef_search_values)) == len(p.ef_search_values)
    assert len(set(p.probes_values)) == len(p.probes_values)


def test_pareto_selectivity_levels_are_valid_keys():
    # The sweep indexes into SELECTIVITY_CONFIGS, so every level must exist.
    for level in ParetoConfig().selectivity_levels:
        assert level in SELECTIVITY_CONFIGS


def test_pareto_default_factory_is_isolated():
    # default_factory (not a shared mutable default) → mutating one instance's
    # list must not bleed into the next instance.
    a = ParetoConfig()
    a.ef_search_values.append(9999)
    b = ParetoConfig()
    assert 9999 not in b.ef_search_values


# ---------------------------------------------------------------------------
# ConcurrencyConfig — multi-client contention sweep (exp_04)
# ---------------------------------------------------------------------------

def test_concurrency_defaults():
    c = ConcurrencyConfig()
    assert c.client_counts == [1, 2, 4, 8]
    assert c.selectivity_level == 0.10
    assert c.queries_per_client == 50
    assert c.warmup_per_client == 5
    assert c.strategy == "A"


def test_concurrency_starts_at_single_client_baseline():
    # exp_04 derives scaling efficiency relative to the first client count, so
    # the sweep must begin at 1 client (the serial baseline).
    assert ConcurrencyConfig().client_counts[0] == 1


def test_concurrency_selectivity_level_is_valid_key():
    assert ConcurrencyConfig().selectivity_level in SELECTIVITY_CONFIGS


def test_concurrency_default_factory_is_isolated():
    a = ConcurrencyConfig()
    a.client_counts.append(16)
    b = ConcurrencyConfig()
    assert 16 not in b.client_counts


# ---------------------------------------------------------------------------
# Module-level defaults are the single source of truth shared with scripts
# ---------------------------------------------------------------------------

def test_module_level_lists_match_dataclass_defaults():
    p = ParetoConfig()
    c = ConcurrencyConfig()
    assert PARETO_EF_SEARCH_VALUES == p.ef_search_values
    assert PARETO_PROBES_VALUES == p.probes_values
    assert CONCURRENCY_CLIENT_COUNTS == c.client_counts
