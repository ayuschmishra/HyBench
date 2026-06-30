"""
Experiment 2 — Adaptive Strategy Selector Evaluation (RQ2)

Four conditions evaluated per selectivity level:
  Fixed-A   : always execute Strategy A (vector-first)
  Fixed-B   : always execute Strategy B (filter-first)
  Adaptive  : dispatch via Lightweight Selectivity-Aware Execution Strategy Selector
  Oracle    : retrospective min(latency_A, latency_B) per query

Output: results/exp_02_adaptive.json
"""

import argparse
import json
import sys
import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import (
    ADAPTIVE_THRESHOLD,
    BenchmarkConfig,
    DataConfig,
    DBConfig,
    HNSWConfig,
    SELECTIVITY_CONFIGS,
    SELECTIVITY_LEVELS,
)
from benchmark.db import connection, get_filtered_row_count, get_table_row_count
from benchmark.planner import execute_adaptive
from benchmark.runner import BenchmarkRunner

# Laptop-relevant queries (used for selectivity configs that filter by Laptop category)
LAPTOP_QUERIES = [
    "lightweight gaming laptop long battery life",
    "professional laptop video editing",
    "budget ultrabook students",
    "high performance workstation laptop",
    "thin light laptop travel",
    "gaming laptop RTX graphics card",
    "MacBook alternative developers",
    "laptop best display quality",
    "rugged laptop outdoor use",
    "laptop thunderbolt connectivity",
]

# General queries (used for cross-category price-only selectivity configs)
GENERAL_QUERIES = [
    "flagship smartphone best camera",
    "budget smartphone under 20000",
    "wireless earbuds long battery",
    "noise cancelling headphones office",
    "budget monitor home office",
    "point-and-shoot camera beginners",
    "smart speaker voice assistant",
    "keyboard mac users",
    "mouse FPS gaming",
    "tablet students reading browsing",
]


def main():
    parser = argparse.ArgumentParser(description="Exp 02: Adaptive Selector Evaluation (RQ2)")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=ADAPTIVE_THRESHOLD)
    args = parser.parse_args()

    db_cfg = DBConfig()
    data_cfg = DataConfig()

    print(f"[exp_02] Loading embedding model ...")
    model = SentenceTransformer(data_cfg.embedding_model, local_files_only=True)

    # Encode query sets
    laptop_embs = model.encode(LAPTOP_QUERIES, normalize_embeddings=True)
    general_embs = model.encode(GENERAL_QUERIES, normalize_embeddings=True)

    n_q = args.n_queries
    n_all = args.n_warmup + n_q

    # Build per-selectivity query embedding arrays (cycling through the pools)
    def build_query_array(is_laptop: bool, n: int) -> list[np.ndarray]:
        pool = laptop_embs if is_laptop else general_embs
        return [pool[i % len(pool)] for i in range(n)]

    bench_cfg = BenchmarkConfig(
        n_queries=n_q,
        n_warmup=args.n_warmup,
        top_k=args.top_k,
        index_type="hnsw",
        hnsw=HNSWConfig(ef_search=args.ef_search),
    )

    results = []

    with connection(db_cfg) as conn:
        total_rows = get_table_row_count(conn, "products")
        runner = BenchmarkRunner(
            conn=conn, cfg=bench_cfg, db_cfg=db_cfg, total_rows=total_rows
        )

        for target_sel in SELECTIVITY_LEVELS:
            cat, mp, mr = SELECTIVITY_CONFIGS[target_sel]
            is_laptop = cat == "Laptop"

            actual_count = get_filtered_row_count(conn, cat, mp, mr)
            actual_sel = actual_count / total_rows

            print(
                f"[exp_02] sel={target_sel:.0%}  actual={actual_sel*100:.2f}%  ({actual_count:,} rows)"
            )

            queries = build_query_array(is_laptop, n_all)
            warmup_q = queries[: args.n_warmup]
            run_q = queries[args.n_warmup :]

            # Each condition runs in its own pass with its own warmup to prevent
            # buffer cache effects from one condition contaminating another's latency.

            # --- Fixed-A pass ---
            for q in warmup_q:
                runner.run_strategy_a(q, cat, mp, mr, args.top_k)
            lat_a = [
                runner.run_strategy_a(q, cat, mp, mr, args.top_k).latency_s * 1000
                for q in run_q
            ]

            # --- Fixed-B pass ---
            for q in warmup_q:
                runner.run_strategy_b(q, cat, mp, mr, args.top_k)
            lat_b = [
                runner.run_strategy_b(q, cat, mp, mr, args.top_k).latency_s * 1000
                for q in run_q
            ]

            # Oracle: per-query min of the two fixed-condition latencies above
            lat_oracle = [min(a, b) for a, b in zip(lat_a, lat_b)]

            # --- Adaptive pass (separate warmup so no cache carryover) ---
            for q in warmup_q:
                execute_adaptive(runner, q, cat, mp, mr, total_rows, args.top_k, args.threshold)

            lat_adapt, adapt_choices, probe_times = [], [], []
            for q in run_q:
                res_ad, sigma_ad, choice, probe_s = execute_adaptive(
                    runner, q, cat, mp, mr, total_rows, args.top_k, args.threshold
                )
                lat_adapt.append(res_ad.latency_s * 1000)
                adapt_choices.append(choice)
                probe_times.append(probe_s * 1000)

            def stats(vals):
                arr = np.array(vals)
                return {
                    "mean_ms": float(np.mean(arr)),
                    "median_ms": float(np.median(arr)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "std_ms": float(np.std(arr)),
                }

            n_chose_a = adapt_choices.count("A")
            n_chose_b = adapt_choices.count("B")

            row = {
                "target_selectivity": target_sel,
                "actual_selectivity": actual_sel,
                "n_filtered": actual_count,
                "threshold": args.threshold,
                "fixed_a": stats(lat_a),
                "fixed_b": stats(lat_b),
                "adaptive": stats(lat_adapt),
                "oracle": stats(lat_oracle),
                "adaptive_choices": {"A": n_chose_a, "B": n_chose_b},
                "probe_overhead_ms": stats(probe_times),
            }
            results.append(row)

            a_mean = row["fixed_a"]["mean_ms"]
            b_mean = row["fixed_b"]["mean_ms"]
            ad_mean = row["adaptive"]["mean_ms"]
            or_mean = row["oracle"]["mean_ms"]
            gap_pct = (ad_mean - or_mean) / max(or_mean, 0.001) * 100
            print(
                f"         Fixed-A={a_mean:.1f}ms  Fixed-B={b_mean:.1f}ms  "
                f"Adaptive={ad_mean:.1f}ms  Oracle={or_mean:.1f}ms  "
                f"gap={gap_pct:+.1f}%  choices=A:{n_chose_a}/B:{n_chose_b}"
            )

    output = {
        "experiment": "exp_02_adaptive",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmark_config": {
            "n_rows": data_cfg.n_rows,
            "n_queries": n_q,
            "n_warmup": args.n_warmup,
            "top_k": args.top_k,
            "index_type": "hnsw",
            "ef_search": args.ef_search,
            "threshold": args.threshold,
        },
        "results": results,
    }

    out_path = Path("results/exp_02_adaptive.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[exp_02] Results saved to {out_path}")


if __name__ == "__main__":
    main()
