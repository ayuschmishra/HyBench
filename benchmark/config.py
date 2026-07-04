"""
Centralised experiment configuration for HyBench.

All tuneable parameters live here. Experiment scripts import from this
module; they never hardcode values themselves.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@dataclass
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "hybench"
    user: str = "hybench"
    password: str = "hybench"

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    n_rows: int = 50_000          # v0.1 default; v0.2 extends to 100K
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64
    random_seed: int = 42
    output_dir: str = "data/synthetic"


# ---------------------------------------------------------------------------
# Index parameters (HNSW since v0.1; IVFFlat added in v0.2)
# ---------------------------------------------------------------------------

@dataclass
class HNSWConfig:
    m: int = 16
    ef_construction: int = 64
    ef_search: int = 40           # session-level GUC, set per-query


@dataclass
class IVFFlatConfig:
    # None means "derive from dataset size" via the helpers below, so a single
    # config works across --scale small/full/large without manual retuning.
    lists: Optional[int] = None    # build-time: number of inverted lists
    probes: Optional[int] = None   # query-time GUC: lists probed per search


# ---------------------------------------------------------------------------
# v0.5 — recall-latency Pareto sweep & concurrency evaluation
# ---------------------------------------------------------------------------

@dataclass
class ParetoConfig:
    """Search-parameter sweep for the recall-latency Pareto frontier (exp_03).

    For HNSW the swept knob is the query-time GUC hnsw.ef_search; for IVFFlat
    it is ivfflat.probes. Each value trades recall against latency: larger =
    higher recall, higher latency. The sweep is what draws the frontier.
    """
    # Fixed selectivity level(s) at which to run the sweep (keys into
    # SELECTIVITY_CONFIGS). Pareto behaviour is cleanest at low selectivity,
    # where Strategy A's ANN search dominates and post-filtering bites hardest.
    selectivity_levels: List[float] = field(default_factory=lambda: [0.01, 0.10])
    ef_search_values: List[int] = field(
        default_factory=lambda: [10, 20, 40, 80, 160, 320, 640]
    )
    probes_values: List[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100]
    )


@dataclass
class ConcurrencyConfig:
    """Multi-client throughput / contention sweep (exp_04).

    Each client is an independent thread with its own psycopg2 connection
    issuing the same fixed workload. Aggregate QPS and latency percentiles are
    reported per client count to characterise contention on HNSW traversal.
    """
    client_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    # Single fixed selectivity for the concurrency sweep: contention, not
    # selectivity, is the independent variable here.
    selectivity_level: float = 0.10
    queries_per_client: int = 50
    warmup_per_client: int = 5
    strategy: str = "A"           # "A" (vector-first) or "B" (filter-first)


# Module-level defaults so experiment scripts and tests share one source.
PARETO_EF_SEARCH_VALUES: List[int] = [10, 20, 40, 80, 160, 320, 640]
PARETO_PROBES_VALUES: List[int] = [1, 2, 5, 10, 20, 50, 100]
CONCURRENCY_CLIENT_COUNTS: List[int] = [1, 2, 4, 8]


def ivfflat_lists_for(n_rows: int) -> int:
    """pgvector guidance: lists ≈ rows/1000 for datasets ≤ 1M rows (min 10)."""
    return max(10, n_rows // 1000)


def ivfflat_probes_for(lists: int) -> int:
    """pgvector guidance: probes ≈ sqrt(lists) (min 1)."""
    return max(1, int(round(lists ** 0.5)))


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    top_k: int = 10
    n_queries: int = 50
    n_warmup: int = 5
    strategy: str = "both"        # "A", "B", or "both"
    index_type: str = "hnsw"      # "hnsw" or "ivfflat"
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    ivfflat: IVFFlatConfig = field(default_factory=IVFFlatConfig)


# ---------------------------------------------------------------------------
# Adaptive selector threshold
# Set to None before Experiment 1 runs; calibrated from exp_01 crossover.
# ---------------------------------------------------------------------------

ADAPTIVE_THRESHOLD: float = 0.05   # derived from exp_01 crossover: B wins below ~5%, A wins above


# ---------------------------------------------------------------------------
# Experiment matrices
# ---------------------------------------------------------------------------

# Six selectivity targets covering the full range.
# 0.75 replaces 0.90: hit cleanly with a cross-category price-cap query (no category filter).
SELECTIVITY_LEVELS: List[float] = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]

# (category_or_None, max_price, min_rating) calibrated against the 50K synthetic dataset (seed=42).
# None category = no category filter (cross-category price-range query).
# Actual selectivity is always verified at runtime via COUNT(*).
SELECTIVITY_CONFIGS = {
    0.01: ("Laptop",  50_000, 0.0),   # ~555 rows   (~1.1%)
    0.05: ("Laptop", 136_000, 0.0),   # ~2500 rows  (~5.0%)
    0.10: ("Laptop", 300_000, 0.0),   # ~5000 rows  (~10.0%)
    0.25: (None,      17_641, 0.0),   # ~12500 rows (~25.0%)
    0.50: (None,      44_249, 0.0),   # ~25000 rows (~50.0%)
    0.75: (None,      82_516, 0.0),   # ~37500 rows (~75.0%)
}


# ---------------------------------------------------------------------------
# Category reference data (shared between generator and selectivity calibration)
# ---------------------------------------------------------------------------

CATEGORY_PROFILES = {
    "Laptop":          {"price_range": (25_000,  2_50_000), "rating_range": (3.0, 5.0)},
    "Smartphone":      {"price_range": (8_000,   1_50_000), "rating_range": (2.5, 5.0)},
    "Tablet":          {"price_range": (12_000,  1_20_000), "rating_range": (3.0, 5.0)},
    "Headphones":      {"price_range": (500,       50_000), "rating_range": (2.0, 5.0)},
    "Gaming Console":  {"price_range": (20_000,    80_000), "rating_range": (3.5, 5.0)},
    "Monitor":         {"price_range": (8_000,    1_00_000), "rating_range": (3.0, 5.0)},
    "Keyboard":        {"price_range": (500,       25_000), "rating_range": (2.5, 5.0)},
    "Mouse":           {"price_range": (300,       15_000), "rating_range": (2.5, 5.0)},
    "Camera":          {"price_range": (15_000,  3_00_000), "rating_range": (3.0, 5.0)},
    "Speaker":         {"price_range": (1_000,     80_000), "rating_range": (2.5, 5.0)},
}

CATEGORY_BRANDS = {
    "Laptop":          ["Dell", "HP", "Lenovo", "ASUS", "Apple", "Acer", "MSI", "Razer"],
    "Smartphone":      ["Samsung", "OnePlus", "Apple", "Realme", "Xiaomi", "Google", "Motorola"],
    "Tablet":          ["Apple", "Samsung", "Lenovo", "Microsoft", "Huawei"],
    "Headphones":      ["Sony", "Bose", "Sennheiser", "JBL", "Audio-Technica", "Jabra"],
    "Gaming Console":  ["Sony", "Microsoft", "Nintendo"],
    "Monitor":         ["LG", "Dell", "Samsung", "BenQ", "ASUS", "Acer", "ViewSonic"],
    "Keyboard":        ["Keychron", "Logitech", "Corsair", "Razer", "SteelSeries", "Ducky"],
    "Mouse":           ["Logitech", "Razer", "SteelSeries", "Corsair", "Zowie"],
    "Camera":          ["Canon", "Nikon", "Sony", "Fujifilm", "Olympus", "Panasonic"],
    "Speaker":         ["Sony", "JBL", "Bose", "Harman", "Marshall", "Sonos"],
}
