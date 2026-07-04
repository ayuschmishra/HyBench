"""Unit tests for v0.2 IVFFlat sizing helpers in benchmark/config.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import (
    IVFFlatConfig,
    ivfflat_lists_for,
    ivfflat_probes_for,
)


# ---------------------------------------------------------------------------
# ivfflat_lists_for — pgvector guidance: lists ~= rows/1000, floor of 10
# ---------------------------------------------------------------------------

def test_lists_scales_with_rows():
    assert ivfflat_lists_for(50_000) == 50
    assert ivfflat_lists_for(100_000) == 100


def test_lists_has_minimum_floor():
    # Tiny scratch datasets must still get a usable (>=10) list count.
    assert ivfflat_lists_for(1_000) == 10
    assert ivfflat_lists_for(0) == 10


# ---------------------------------------------------------------------------
# ivfflat_probes_for — pgvector guidance: probes ~= sqrt(lists), floor of 1
# ---------------------------------------------------------------------------

def test_probes_is_sqrt_of_lists():
    assert ivfflat_probes_for(100) == 10
    assert ivfflat_probes_for(50) == 7   # round(sqrt(50)) = round(7.07) = 7


def test_probes_has_minimum_floor():
    assert ivfflat_probes_for(1) == 1
    assert ivfflat_probes_for(0) == 1


# ---------------------------------------------------------------------------
# IVFFlatConfig defaults — None means "derive from dataset size at runtime"
# ---------------------------------------------------------------------------

def test_ivfflat_config_defaults_are_none():
    cfg = IVFFlatConfig()
    assert cfg.lists is None
    assert cfg.probes is None


def test_ivfflat_config_accepts_overrides():
    cfg = IVFFlatConfig(lists=200, probes=15)
    assert cfg.lists == 200
    assert cfg.probes == 15
