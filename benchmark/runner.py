"""
Core benchmark execution engine for HyBench.

Implements two hybrid query strategies and ground-truth computation:

  Strategy A  —  Vector-first (ANN → post-filter)
      Uses a CTE that retrieves a large candidate pool ordered by embedding
      similarity (exploiting an ANN index if present), then applies relational
      predicates as a post-filter.

  Strategy B  —  Filter-first (predicate → exact KNN)
      Uses a CTE that applies relational predicates first (using B-tree
      indexes), then performs exact nearest-neighbour search within the
      resulting filtered set.

Ground truth is computed by Strategy B with sequential scan forced off on
the index scan side — i.e., exact KNN over the filtered set.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import psycopg2

from benchmark.config import BenchmarkConfig, DBConfig
from benchmark.db import (
    execute_timed,
    get_filtered_row_count,
    set_session_gucs,
)
from benchmark.metrics import compute_recall_at_k, aggregate_latencies


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    ids: List[int]
    latency_s: float
    strategy: str
    recall_at_k: Optional[float] = None


@dataclass
class ExperimentResult:
    config: dict
    strategy_a: Dict = field(default_factory=dict)
    strategy_b: Dict = field(default_factory=dict)
    ground_truth_ids: Optional[List[int]] = None
    selectivity_actual: Optional[float] = None
    n_filtered: Optional[int] = None


# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

# Strategy A: retrieve `candidate_multiplier * top_k` rows via ANN index,
# then apply relational filter.  The candidate pool must be large enough
# that enough post-filter rows survive to satisfy top_k.
_STRATEGY_A_SQL = """
WITH vector_candidates AS (
    SELECT
        id, category, price, brand, rating, description,
        embedding <=> %(qvec)s::vector AS distance
    FROM products
    ORDER BY embedding <=> %(qvec)s::vector
    LIMIT %(n_candidates)s
)
SELECT id, category, price, brand, rating, distance
FROM vector_candidates
WHERE price < %(max_price)s
  AND rating > %(min_rating)s
  AND (%(category)s::text IS NULL OR category = %(category)s::text)
ORDER BY distance
LIMIT %(top_k)s;
"""

# Strategy B: apply relational filter first (B-tree indexes), then exact KNN.
# MATERIALIZED forces CTE evaluation before ORDER BY so PostgreSQL cannot
# push the ORDER BY into an HNSW scan (which would bypass the relational filter).
_STRATEGY_B_SQL = """
WITH filtered AS MATERIALIZED (
    SELECT id, category, price, brand, rating, embedding
    FROM products
    WHERE price < %(max_price)s
      AND rating > %(min_rating)s
      AND (%(category)s::text IS NULL OR category = %(category)s::text)
)
SELECT
    id, category, price, brand, rating,
    embedding <=> %(qvec)s::vector AS distance
FROM filtered
ORDER BY embedding <=> %(qvec)s::vector
LIMIT %(top_k)s;
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        cfg: BenchmarkConfig,
        db_cfg: DBConfig,
        total_rows: int,
    ) -> None:
        self.conn = conn
        self.cfg = cfg
        self.db_cfg = db_cfg
        self.total_rows = total_rows

    # ------------------------------------------------------------------
    # Strategy A
    # ------------------------------------------------------------------

    def run_strategy_a(
        self,
        query_embedding: np.ndarray,
        category: str,
        max_price: float,
        min_rating: float,
        top_k: int,
        candidate_multiplier: int = 100,
    ) -> QueryResult:
        # 100x default ensures results even at ~1% selectivity (1000 candidates * 0.01 = 10 expected matches)
        n_candidates = top_k * candidate_multiplier

        if self.cfg.index_type == "ivfflat":
            raise NotImplementedError(
                "IVFFlat support deferred to HyBench v0.2. "
                "Use index_type='hnsw' for v0.1."
            )
        elif self.cfg.index_type == "hnsw":
            # ef_search must be >= n_candidates for HNSW to return enough rows.
            # pgvector caps returned rows at ef_search even if LIMIT > ef_search.
            set_session_gucs(
                self.conn,
                enable_seqscan=False,
                hnsw_ef_search=max(self.cfg.hnsw.ef_search, n_candidates),
            )
        else:
            set_session_gucs(self.conn, enable_seqscan=True)

        params = {
            "qvec": query_embedding.tolist(),
            "max_price": max_price,
            "min_rating": min_rating,
            "category": category,
            "top_k": top_k,
            "n_candidates": n_candidates,
        }

        rows, elapsed = execute_timed(self.conn, _STRATEGY_A_SQL, params)
        ids = [r["id"] for r in rows]
        return QueryResult(ids=ids, latency_s=elapsed, strategy="A")

    # ------------------------------------------------------------------
    # Strategy B
    # ------------------------------------------------------------------

    def run_strategy_b(
        self,
        query_embedding: np.ndarray,
        category: str,
        max_price: float,
        min_rating: float,
        top_k: int,
    ) -> QueryResult:
        # Strategy B always does exact KNN within the filtered set;
        # disable vector index so PostgreSQL uses sequential scan over the CTE.
        set_session_gucs(self.conn, enable_seqscan=True)

        params = {
            "qvec": query_embedding.tolist(),
            "max_price": max_price,
            "min_rating": min_rating,
            "category": category,
            "top_k": top_k,
        }

        rows, elapsed = execute_timed(self.conn, _STRATEGY_B_SQL, params)
        ids = [r["id"] for r in rows]
        return QueryResult(ids=ids, latency_s=elapsed, strategy="B")

    # ------------------------------------------------------------------
    # Ground truth  (exact KNN over filtered set; recall reference)
    # ------------------------------------------------------------------

    def compute_ground_truth(
        self,
        query_embedding: np.ndarray,
        category: str,
        max_price: float,
        min_rating: float,
        top_k: int,
    ) -> List[int]:
        """Return exact top-K IDs within the filtered set (brute force)."""
        set_session_gucs(self.conn, enable_seqscan=True)
        params = {
            "qvec": query_embedding.tolist(),
            "max_price": max_price,
            "min_rating": min_rating,
            "category": category,
            "top_k": top_k,
        }
        rows, _ = execute_timed(self.conn, _STRATEGY_B_SQL, params)
        return [r["id"] for r in rows]

    # ------------------------------------------------------------------
    # Full benchmark run for one configuration
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        query_embeddings: List[np.ndarray],
        category: str,
        max_price: float,
        min_rating: float,
        top_k: int,
        label: str = "",
    ) -> ExperimentResult:
        """
        Run n_warmup + n_queries hybrid queries for both strategies.

        Returns an ExperimentResult with per-strategy latency statistics
        and recall values.
        """
        n_total = self.cfg.n_warmup + self.cfg.n_queries
        if len(query_embeddings) < n_total:
            raise ValueError(
                f"Need {n_total} query embeddings but got {len(query_embeddings)}"
            )

        n_filtered = get_filtered_row_count(
            self.conn, category, max_price, min_rating
        )
        selectivity = n_filtered / max(self.total_rows, 1)

        a_latencies, b_latencies = [], []
        a_recalls, b_recalls = [], []

        for i, qvec in enumerate(query_embeddings[:n_total]):
            gt_ids = self.compute_ground_truth(
                qvec, category, max_price, min_rating, top_k
            )

            if self.cfg.strategy in ("A", "both"):
                res_a = self.run_strategy_a(
                    qvec, category, max_price, min_rating, top_k
                )
                if i >= self.cfg.n_warmup:
                    a_latencies.append(res_a.latency_s)
                    a_recalls.append(compute_recall_at_k(res_a.ids, gt_ids, top_k))

            if self.cfg.strategy in ("B", "both"):
                res_b = self.run_strategy_b(
                    qvec, category, max_price, min_rating, top_k
                )
                if i >= self.cfg.n_warmup:
                    b_latencies.append(res_b.latency_s)
                    b_recalls.append(compute_recall_at_k(res_b.ids, gt_ids, top_k))

        result = ExperimentResult(
            config={
                "label": label,
                "category": category,
                "max_price": max_price,
                "min_rating": min_rating,
                "top_k": top_k,
                "index_type": self.cfg.index_type,
                "n_queries": self.cfg.n_queries,
                "total_rows": self.total_rows,
            },
            selectivity_actual=selectivity,
            n_filtered=n_filtered,
        )

        if a_latencies:
            result.strategy_a = {
                **aggregate_latencies(a_latencies),
                "recall_mean": float(np.mean(a_recalls)),
                "recall_values": a_recalls,
            }
        if b_latencies:
            result.strategy_b = {
                **aggregate_latencies(b_latencies),
                "recall_mean": float(np.mean(b_recalls)),
                "recall_values": b_recalls,
            }

        return result

    # ------------------------------------------------------------------
    # Throughput measurement (sequential burst)
    # ------------------------------------------------------------------

    def measure_throughput(
        self,
        query_embeddings: List[np.ndarray],
        category: str,
        max_price: float,
        min_rating: float,
        top_k: int,
        n_queries: int = 100,
    ) -> Dict[str, float]:
        process = psutil.Process()
        results = {}

        for strategy, run_fn in [
            ("A", lambda q: self.run_strategy_a(q, category, max_price, min_rating, top_k)),
            ("B", lambda q: self.run_strategy_b(q, category, max_price, min_rating, top_k)),
        ]:
            mem_before = process.memory_info().rss
            t0 = time.perf_counter()
            for qvec in query_embeddings[:n_queries]:
                run_fn(qvec)
            wall = time.perf_counter() - t0
            mem_after = process.memory_info().rss

            results[f"strategy_{strategy}_qps"] = n_queries / wall
            results[f"strategy_{strategy}_mem_delta_mb"] = (
                mem_after - mem_before
            ) / 1024**2

        return results
