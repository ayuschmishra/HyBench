"""
Database connection utilities for HyBench.

Provides a thin wrapper around psycopg2 with:
- connection acquisition from a simple pool
- timed query execution that returns (rows, elapsed_seconds)
- EXPLAIN ANALYZE capture
- session-level GUC setting (ivfflat.probes, hnsw.ef_search)
"""

import time
import contextlib
from typing import Any, List, Optional, Tuple

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from benchmark.config import DBConfig


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
