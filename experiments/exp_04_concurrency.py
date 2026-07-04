"""
Experiment 4 — Concurrency & Throughput Evaluation (RQ4)

Characterises how hybrid-query throughput (QPS) and tail latency respond to
concurrent client load. pgvector serves each client on its own backend, and
HNSW graph traversal takes a shared read lock, so aggregate throughput does not
scale linearly with client count and per-query latency inflates under
contention.

Independent variable : number of concurrent clients in {1, 2, 4, 8}
Dependent variables  : aggregate QPS, mean / P95 / P99 latency
Controlled           : one fixed selectivity level, one strategy, one index

Each client is a thread with its own psycopg2 connection running the same fixed
workload; a barrier releases all clients into the timed phase together so the
measured wall clock reflects genuine concurrent load (see benchmark/concurrency).

Recall/ground-truth is intentionally omitted: this experiment measures
throughput under contention, and recall is already characterised by exp_01/03.

Output: results/exp_04_concurrency.json
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.concurrency import run_concurrent_workload
from benchmark.config import (
    BenchmarkConfig,
    ConcurrencyConfig,
    DBConfig,
    DataConfig,
    HNSWConfig,
    IVFFlatConfig,
    SELECTIVITY_CONFIGS,
)
from benchmark.db import (
    connection,
    ensure_vector_index,
    get_filtered_row_count,
    get_memory_metrics,
    get_table_row_count,
)

# Compact query pool (cycled by the workload runner). The default 0.10 config
# filters by Laptop, so laptop-relevant queries keep Strategy A returning rows.
QUERY_POOL = [
    "lightweight gaming laptop long battery life",
    "professional laptop for video editing",
    "budget ultrabook for students",
    "high performance workstation laptop",
    "thin and light laptop for travel",
    "gaming laptop with RTX graphics card",
    "MacBook alternative for developers",
    "laptop with best display quality",
    "rugged laptop for outdoor use",
    "laptop with thunderbolt connectivity",
    "laptop for machine learning training",
    "business laptop enterprise security",
    "laptop for data science python",
    "compact 13 inch ultrabook",
    "powerful laptop under 80000 rupees",
    "laptop for graphic design illustration",
    "AMD Ryzen laptop best value",
    "laptop fast SSD NVMe storage",
    "silent fanless laptop for coding",
    "laptop for photo editing lightroom",
]


def _parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    c = ConcurrencyConfig()
    parser = argparse.ArgumentParser(description="Exp 04: Concurrency & Throughput (RQ4)")
    parser.add_argument("--clients",
                        default=",".join(str(n) for n in c.client_counts),
                        help="Comma-separated concurrent client counts (default: 1,2,4,8)")
    parser.add_argument("--strategy", choices=["A", "B"], default=c.strategy,
                        help="Strategy to benchmark under load (default: A)")
    parser.add_argument("--selectivity", type=float, default=c.selectivity_level,
                        help="Fixed selectivity level (SELECTIVITY_CONFIGS key, default: 0.10)")
    parser.add_argument("--queries-per-client", type=int, default=c.queries_per_client)
    parser.add_argument("--warmup-per-client", type=int, default=c.warmup_per_client)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-rows", type=int, default=50_000)
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--output", type=Path,
                        default=Path("results/exp_04_concurrency.json"))
    args = parser.parse_args()

    client_counts = _parse_ints(args.clients)
    if args.selectivity not in SELECTIVITY_CONFIGS:
        print(f"[exp_04] ERROR: --selectivity {args.selectivity} not in "
              f"{sorted(SELECTIVITY_CONFIGS)}")
        sys.exit(1)
    cat, mp, mr = SELECTIVITY_CONFIGS[args.selectivity]

    db_cfg = DBConfig()
    data_cfg = DataConfig(n_rows=args.n_rows)
    bench_cfg = BenchmarkConfig(
        top_k=args.top_k, n_queries=args.queries_per_client,
        n_warmup=args.warmup_per_client, strategy=args.strategy,
        index_type=args.index_type, hnsw=HNSWConfig(), ivfflat=IVFFlatConfig(),
    )

    print("[exp_04] Loading embedding model for query encoding...")
    model = SentenceTransformer(data_cfg.embedding_model, local_files_only=True)
    pool_embs = model.encode(
        QUERY_POOL, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    query_embeddings = [pool_embs[i] for i in range(len(pool_embs))]

    pg_version = ""
    memory = {}
    with connection(db_cfg) as conn:
        with conn.cursor() as _cur:
            _cur.execute("SELECT version()")
            pg_version = _cur.fetchone()[0]
        total_rows = get_table_row_count(conn)
        index_info = ensure_vector_index(
            conn, args.index_type, bench_cfg.hnsw, bench_cfg.ivfflat, total_rows
        )
        n_filtered = get_filtered_row_count(conn, cat, mp, mr)
        actual_sel = n_filtered / max(total_rows, 1)
        print(
            f"[exp_04] Index {index_info['name']} params={index_info['params']} | "
            f"sel target={args.selectivity:.0%} actual={actual_sel*100:.2f}% "
            f"({n_filtered:,} rows) | strategy {args.strategy}"
        )
        memory = get_memory_metrics(conn)

    # Each client count runs its own concurrent workload (own connections).
    results = []
    baseline_qps = None
    for n_clients in client_counts:
        agg = run_concurrent_workload(
            db_cfg=db_cfg, cfg=bench_cfg, total_rows=total_rows,
            query_embeddings=query_embeddings,
            category=cat, max_price=mp, min_rating=mr, top_k=args.top_k,
            n_clients=n_clients, strategy=args.strategy,
            n_warmup=args.warmup_per_client, n_queries=args.queries_per_client,
        )
        if baseline_qps is None:
            baseline_qps = agg["qps"]
        agg["target_selectivity"] = args.selectivity
        agg["actual_selectivity"] = actual_sel
        agg["scaling_efficiency"] = (
            (agg["qps"] / baseline_qps / n_clients) if baseline_qps else 0.0
        )
        results.append(agg)
        print(
            f"[exp_04] clients={n_clients:<2d}  QPS={agg['qps']:8.1f}  "
            f"mean={agg['mean_ms']:7.2f}ms  p95={agg['p95_ms']:7.2f}ms  "
            f"scale_eff={agg['scaling_efficiency']*100:5.1f}%  errors={agg['total_errors']}"
        )

    output = {
        "experiment": "exp_04_concurrency",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_config": {
            "n_rows": data_cfg.n_rows,
            "client_counts": client_counts,
            "strategy": args.strategy,
            "selectivity": args.selectivity,
            "queries_per_client": args.queries_per_client,
            "warmup_per_client": args.warmup_per_client,
            "top_k": args.top_k,
            "index_type": args.index_type,
            "index_params": index_info["params"],
            "platform": platform.platform(),
            "python_version": sys.version,
            "pg_version": pg_version,
        },
        "memory": memory,
        "results": results,
    }

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[exp_04] Results saved to {out_path}")


if __name__ == "__main__":
    main()
