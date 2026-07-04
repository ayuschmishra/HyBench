"""
Database connection utilities for HyBench.

Provides a thin wrapper around psycopg2 with:
- connection acquisition from a simple pool
- timed query execution that returns (rows, elapsed_seconds)
- EXPLAIN ANALYZE capture
- session-level GUC setting (ivfflat.probes, hnsw.ef_search)
- ANN index lifecycle management (ensure_vector_index)          [v0.2]
- memory profiling: pg_relation_size + client RSS               [v0.2]
"""

import time
import contextlib
from typing import Any, Dict, List, Optional, Tuple

import psutil
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from benchmark.config import (
    DBConfig,
    HNSWConfig,
    IVFFlatConfig,
    ivfflat_lists_for,
)


def get_connection(cfg: DBConfig) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(cfg.dsn)
    conn.autocommit = True
    register_vector(conn)
    return conn


@contextlib.contextmanager
def connection(cfg: DBConfig):
    conn = get_connection(cfg)
    try:
        yield conn
    finally:
        conn.close()


def set_session_gucs(
    conn,
    enable_seqscan: bool = True,
    ivfflat_probes: Optional[int] = None,
    hnsw_ef_search: Optional[int] = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(f"SET enable_seqscan = {'on' if enable_seqscan else 'off'};")
        if ivfflat_probes is not None:
            cur.execute(f"SET ivfflat.probes = {ivfflat_probes};")
        if hnsw_ef_search is not None:
            cur.execute(f"SET hnsw.ef_search = {hnsw_ef_search};")


def execute_timed(
    conn,
    sql: str,
    params: Tuple[Any, ...] = (),
) -> Tuple[List[Any], float]:
    """Execute *sql* with *params* and return (rows, elapsed_seconds)."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        t0 = time.perf_counter()
        cur.execute(sql, params)
        rows = cur.fetchall()
        elapsed = time.perf_counter() - t0
    return rows, elapsed


def execute_explain(conn, sql: str, params: Tuple[Any, ...] = ()) -> str:
    """Return the EXPLAIN ANALYZE output for the given query."""
    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"
    with conn.cursor() as cur:
        cur.execute(explain_sql, params)
        lines = cur.fetchall()
    return "\n".join(line[0] for line in lines)


def get_table_row_count(conn, table: str = "products") -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        return cur.fetchone()[0]


def get_filtered_row_count(
    conn,
    category: Optional[str],
    max_price: float,
    min_rating: float,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM products
            WHERE price < %s
              AND rating > %s
              AND (%s::text IS NULL OR category = %s::text);
            """,
            (max_price, min_rating, category, category),
        )
        return cur.fetchone()[0]


def get_index_sizes(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
            FROM pg_indexes
            WHERE tablename = 'products';
            """
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def get_table_size_bytes(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_total_relation_size('products');")
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# ANN index lifecycle (v0.2)
# ---------------------------------------------------------------------------

VECTOR_INDEX_NAMES = {
    "hnsw": "idx_products_hnsw",
    "ivfflat": "idx_products_ivfflat",
}


def ensure_vector_index(
    conn,
    index_type: str,
    hnsw_cfg: Optional[HNSWConfig] = None,
    ivfflat_cfg: Optional[IVFFlatConfig] = None,
    n_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Ensure exactly ONE ANN index — the requested one — exists on products.embedding.

    The other index type is dropped: with both present, PostgreSQL's planner
    would silently pick whichever it costs cheaper for the ORDER BY, destroying
    per-index attribution. Rebuild is skipped when the requested index already
    exists with matching build parameters. Runs ANALYZE after any rebuild so
    pg_stats stays fresh for the PgStatsEstimator.

    Returns {"index_type", "name", "params", "built", "build_seconds"}.
    """
    if index_type not in VECTOR_INDEX_NAMES:
        raise ValueError(f"index_type must be one of {list(VECTOR_INDEX_NAMES)}, got {index_type!r}")

    hnsw_cfg = hnsw_cfg or HNSWConfig()
    ivfflat_cfg = ivfflat_cfg or IVFFlatConfig()
    if n_rows is None:
        n_rows = get_table_row_count(conn)

    if index_type == "hnsw":
        params = {"m": hnsw_cfg.m, "ef_construction": hnsw_cfg.ef_construction}
        create_sql = (
            f"CREATE INDEX {VECTOR_INDEX_NAMES['hnsw']} "
            f"ON products USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {hnsw_cfg.m}, ef_construction = {hnsw_cfg.ef_construction});"
        )
    else:
        lists = ivfflat_cfg.lists or ivfflat_lists_for(n_rows)
        params = {"lists": lists}
        create_sql = (
            f"CREATE INDEX {VECTOR_INDEX_NAMES['ivfflat']} "
            f"ON products USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {lists});"
        )

    wanted = VECTOR_INDEX_NAMES[index_type]
    other = VECTOR_INDEX_NAMES["ivfflat" if index_type == "hnsw" else "hnsw"]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'products' AND indexname = ANY(%s);",
            ([wanted, other],),
        )
        existing = {row[0]: row[1] for row in cur.fetchall()}

        if other in existing:
            cur.execute(f"DROP INDEX IF EXISTS {other};")

        # indexdef renders options as key='value'; accept both quoting styles.
        def _params_match(indexdef: str) -> bool:
            return all(
                f"{k}='{v}'" in indexdef or f"{k}={v}" in indexdef
                for k, v in params.items()
            )

        if wanted in existing and _params_match(existing[wanted]):
            return {
                "index_type": index_type, "name": wanted, "params": params,
                "built": False, "build_seconds": 0.0,
            }

        cur.execute(f"DROP INDEX IF EXISTS {wanted};")
        t0 = time.perf_counter()
        cur.execute(create_sql)
        build_s = time.perf_counter() - t0
        cur.execute("ANALYZE products;")

    return {
        "index_type": index_type, "name": wanted, "params": params,
        "built": True, "build_seconds": build_s,
    }


# ---------------------------------------------------------------------------
# Memory profiling (v0.2): server-side relation sizes + client RSS
# ---------------------------------------------------------------------------

def get_memory_metrics(conn) -> Dict[str, Any]:
    """Collect memory/storage footprint for the results-JSON metadata block.

    Server side: pg_relation_size for the heap and every products index,
    pg_total_relation_size for the full relation (heap + indexes + TOAST).
    Client side: psutil RSS of this process (peak working set where the
    platform exposes it, e.g. Windows).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size('products'), pg_total_relation_size('products');")
        heap_bytes, total_bytes = cur.fetchone()
        cur.execute(
            """
            SELECT indexname, pg_relation_size(indexname::regclass)
            FROM pg_indexes
            WHERE tablename = 'products';
            """
        )
        index_bytes = {row[0]: row[1] for row in cur.fetchall()}

    mem = psutil.Process().memory_info()
    return {
        "table_heap_bytes": heap_bytes,
        "table_total_bytes": total_bytes,
        "index_bytes": index_bytes,
        "client_rss_bytes": mem.rss,
        "client_peak_rss_bytes": getattr(mem, "peak_wset", None),
    }
