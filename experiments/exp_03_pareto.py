"""
Experiment 3 — Recall-Latency Pareto Frontier / Index Parameter Sensitivity (RQ3)

The primary v0.5 contribution: characterise the recall-vs-latency trade-off of
each ANN index as its query-time search parameter is swept.

  HNSW    : sweep hnsw.ef_search   in {10, 20, 40, 80, 160, 320, 640}
  IVFFlat : sweep ivfflat.probes   in {1, 2, 5, 10, 20, 50, 100}

Each value is run through Strategy A (vector-first + relational post-filter) at
one or more FIXED selectivity levels. Larger search parameter -> more ANN
candidates survive the post-filter -> higher Recall@K but higher latency. The
resulting (latency, recall) points trace the Pareto frontier (Figure 4).

Unlike exp_01, Strategy A here uses the ef_search_override / probes_override
path so the swept value is applied exactly (the exp_01 floor to n_candidates is
bypassed) — that floor is precisely what a frontier sweep must vary.

Ground truth (exact KNN over the filtered set) depends only on (query, filter),
so it is computed once per selectivity level and reused across the whole sweep.

Output: results/exp_03_pareto.json
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

from benchmark.config import (
    BenchmarkConfig,
    DBConfig,
    DataConfig,
    HNSWConfig,
    IVFFlatConfig,
    ParetoConfig,
    SELECTIVITY_CONFIGS,
)
from benchmark.db import (
    connection,
    ensure_vector_index,
    get_filtered_row_count,
    get_memory_metrics,
    get_table_row_count,
)
from benchmark.metrics import aggregate_latencies, compute_recall_at_k
from benchmark.runner import BenchmarkRunner

# Category-relevant query pools. Laptop-filtered selectivity configs need
# laptop-relevant query vectors, else HNSW returns rows from the wrong
# embedding cluster and recall collapses (the v0.1 category-mismatch lesson).
LAPTOP_QUERIES = [
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
    "convertible 2-in-1 laptop tablet",
    "laptop with best keyboard typing",
    "laptop for data science python",
    "compact 13 inch ultrabook",
    "powerful laptop under 80000 rupees",
    "laptop for graphic design illustration",
    "AMD Ryzen laptop best value",
    "laptop fast SSD NVMe storage",
    "silent fanless laptop for coding",
    "laptop with OLED display",
    "17 inch large screen laptop",
    "laptop for software development",
    "affordable gaming laptop budget",
    "laptop for photo editing lightroom",
    "laptop for college engineering student",
    "touchscreen laptop creative work",
    "laptop with good cooling system",
    "slim laptop powerful processor",
    "laptop for competitive programming",
    "laptop for 3D modelling CAD",
    "laptop QHD 2K display resolution",
    "laptop for video conferencing remote work",
    "laptop with fingerprint reader",
]

GENERAL_QUERIES = [
    "flagship smartphone best camera",
    "budget smartphone under 20000",
    "wireless earbuds long battery",
    "noise cancelling headphones office",
    "budget monitor home office",
    "point-and-shoot camera beginners",
    "smart speaker voice assistant",
    "keyboard for mac users",
    "mouse for FPS gaming",
    "tablet for students reading browsing",
    "gaming console family entertainment",
    "4K monitor photo video editing",
    "mechanical keyboard cherry switches",
    "ergonomic mouse for wrist pain",
    "portable bluetooth speaker outdoor",
    "mirrorless camera travel photography",
    "gaming headset microphone quality",
    "ultrawide monitor productivity",
    "wireless keyboard mouse combo",
    "action camera underwater waterproof",
    "curved gaming monitor immersive",
    "studio monitor headphones mixing",
    "compact keyboard travel portable",
    "soundbar home theatre surround",
    "DSLR camera wildlife photography",
    "gaming mouse lightweight competitive",
    "high refresh rate monitor esports",
    "tablet digital art drawing stylus",
    "bookshelf speakers audiophile quality",
    "webcam streaming content creation",
    "gaming keyboard RGB customizable",
    "camera lens portrait photography",
    "true wireless earbuds sport fitness",
    "tablet with cellular 5G connectivity",
    "over-ear headphones comfort long wear",
]


def _parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def encode_pool(model, texts, n):
    embs = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    return [embs[i % len(embs)] for i in range(n)]


def main():
    p = ParetoConfig()
    parser = argparse.ArgumentParser(description="Exp 03: Recall-Latency Pareto (RQ3)")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-rows", type=int, default=50_000,
                        help="Dataset size to benchmark against (default: 50000)")
    parser.add_argument("--index-types", default="hnsw,ivfflat",
                        help="Comma-separated ANN indexes to sweep (default: hnsw,ivfflat)")
    parser.add_argument("--selectivity-levels",
                        default=",".join(str(s) for s in p.selectivity_levels),
                        help="Comma-separated SELECTIVITY_CONFIGS keys to sweep at")
    parser.add_argument("--ef-search-values",
                        default=",".join(str(v) for v in p.ef_search_values),
                        help="HNSW ef_search sweep values")
    parser.add_argument("--probes-values",
                        default=",".join(str(v) for v in p.probes_values),
                        help="IVFFlat probes sweep values")
    parser.add_argument("--output", type=Path,
                        default=Path("results/exp_03_pareto.json"))
    args = parser.parse_args()

    index_types = [t.strip() for t in args.index_types.split(",") if t.strip()]
    invalid = set(index_types) - {"hnsw", "ivfflat"}
    if invalid or not index_types:
        print(f"[exp_03] ERROR: invalid --index-types {args.index_types!r}")
        sys.exit(1)

    sel_levels = _parse_floats(args.selectivity_levels)
    ef_values = _parse_ints(args.ef_search_values)
    probes_values = _parse_ints(args.probes_values)

    db_cfg = DBConfig()
    data_cfg = DataConfig(n_rows=args.n_rows)

    print("[exp_03] Loading embedding model for query encoding...")
    model = SentenceTransformer(data_cfg.embedding_model, local_files_only=True)

    n_total = args.n_warmup + args.n_queries
    laptop_q = encode_pool(model, LAPTOP_QUERIES, n_total)
    general_q = encode_pool(model, GENERAL_QUERIES, n_total)

    results = []
    pg_version = ""

    with connection(db_cfg) as conn:
        with conn.cursor() as _cur:
            _cur.execute("SELECT version()")
            pg_version = _cur.fetchone()[0]

        total_rows = get_table_row_count(conn)

        for index_type in index_types:
            bench_cfg = BenchmarkConfig(
                top_k=args.top_k, n_queries=args.n_queries, n_warmup=args.n_warmup,
                strategy="A", index_type=index_type,
                hnsw=HNSWConfig(), ivfflat=IVFFlatConfig(),
            )
            index_info = ensure_vector_index(
                conn, index_type, bench_cfg.hnsw, bench_cfg.ivfflat, total_rows
            )
            print(
                f"\n[exp_03] Index {index_info['name']} params={index_info['params']} "
                f"({'rebuilt %.1fs' % index_info['build_seconds'] if index_info['built'] else 'reused'})"
            )
            runner = BenchmarkRunner(conn, bench_cfg, db_cfg, total_rows)

            sweep_values = ef_values if index_type == "hnsw" else probes_values
            param_name = "ef_search" if index_type == "hnsw" else "probes"

            for target_sel in sel_levels:
                cat, mp, mr = SELECTIVITY_CONFIGS[target_sel]
                queries = laptop_q if cat == "Laptop" else general_q
                warmup_q = queries[: args.n_warmup]
                run_q = queries[args.n_warmup:]

                n_filtered = get_filtered_row_count(conn, cat, mp, mr)
                actual_sel = n_filtered / max(total_rows, 1)

                # Ground truth is independent of the swept parameter → compute once.
                gt_list = [
                    runner.compute_ground_truth(q, cat, mp, mr, args.top_k)
                    for q in run_q
                ]

                print(
                    f"[exp_03] {index_type} sel target={target_sel:.0%} "
                    f"actual={actual_sel*100:.2f}% ({n_filtered:,} rows)"
                )

                sweep = []
                for pv in sweep_values:
                    kw = ({"ef_search_override": pv} if index_type == "hnsw"
                          else {"probes_override": pv})
                    for q in warmup_q:
                        runner.run_strategy_a(q, cat, mp, mr, args.top_k, **kw)

                    lats, recalls = [], []
                    for q, gt in zip(run_q, gt_list):
                        res = runner.run_strategy_a(q, cat, mp, mr, args.top_k, **kw)
                        lats.append(res.latency_s)
                        recalls.append(compute_recall_at_k(res.ids, gt, args.top_k))

                    agg = aggregate_latencies(lats)
                    point = {
                        "param_value": pv,
                        "mean_ms": agg["mean_ms"],
                        "median_ms": agg["median_ms"],
                        "p95_ms": agg["p95_ms"],
                        "std_ms": agg["std_ms"],
                        "recall_mean": float(np.mean(recalls)),
                    }
                    sweep.append(point)
                    print(
                        f"           {param_name}={pv:<4d} "
                        f"mean={point['mean_ms']:7.2f}ms  recall={point['recall_mean']:.3f}"
                    )

                results.append({
                    "index_type": index_type,
                    "param_name": param_name,
                    "target_selectivity": target_sel,
                    "actual_selectivity": actual_sel,
                    "n_filtered": n_filtered,
                    "total_rows": total_rows,
                    "config": {"category": cat, "max_price": mp, "min_rating": mr},
                    "sweep": sweep,
                })

        memory = get_memory_metrics(conn)

    output = {
        "experiment": "exp_03_pareto",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_config": {
            "n_rows": data_cfg.n_rows,
            "n_queries": args.n_queries,
            "n_warmup": args.n_warmup,
            "top_k": args.top_k,
            "index_types": index_types,
            "ef_search_values": ef_values,
            "probes_values": probes_values,
            "selectivity_levels": sel_levels,
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
    print(f"\n[exp_03] Results saved to {out_path}")


if __name__ == "__main__":
    main()
