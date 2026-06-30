"""
Experiment 1 — Filter Selectivity vs. Latency  (RQ1)

Fixed variables : dataset = 50K, index = HNSW (ef_search=40), top_k = 10
Varied variable : filter selectivity in {1%, 5%, 10%, 25%, 50%, 75%}

Protocol
--------
For each selectivity target:
  1. Look up the pre-calibrated (category, max_price, min_rating) triple from
     SELECTIVITY_CONFIGS (benchmark/config.py).
  2. Run n_warmup warmup queries (discarded) then n_queries timed queries,
     alternating A/B per query to avoid plan-cache bias.
  3. Record per-query latencies, aggregate statistics, and actual selectivity
     (verified at runtime via COUNT(*)).

Output : results/exp_01_selectivity.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import (
    BenchmarkConfig,
    DBConfig,
    DataConfig,
    HNSWConfig,
    SELECTIVITY_LEVELS,
    SELECTIVITY_CONFIGS,
)
from benchmark.db import (
    connection,
    get_filtered_row_count,
    get_table_row_count,
    set_session_gucs,
)
from benchmark.runner import BenchmarkRunner
from data_gen.generator import load_saved

QUERY_TEXTS = [
    "lightweight gaming laptop with long battery life",
    "professional laptop for video editing",
    "budget ultrabook for students",
    "high performance workstation laptop",
    "thin and light laptop for travel",
    "gaming laptop with RTX graphics card",
    "MacBook alternative for developers",
    "laptop with best display quality",
    "rugged laptop for outdoor use",
    "laptop with thunderbolt connectivity",
    "flagship smartphone with best camera",
    "budget smartphone under 20000",
    "smartphone with longest battery life",
    "premium gaming smartphone",
    "business smartphone with enterprise security",
    "wireless headphones with noise cancellation",
    "true wireless earbuds for sports",
    "audiophile headphones for music production",
    "affordable gaming mouse with high DPI",
    "ergonomic mouse for office work",
    "compact mechanical keyboard for coding",
    "best ultrawide monitor for gaming",
    "4K monitor for photo editing",
    "mirrorless camera for travel photography",
    "portable bluetooth speaker for outdoor use",
    "tablet for digital art and drawing",
    "gaming console for AAA titles",
    "budget monitor for home office",
    "point-and-shoot camera for beginners",
    "smart speaker with voice assistant",
    "keyboard for mac users",
    "monitor with high refresh rate for esports",
    "camera with best video recording quality",
    "speaker for home theatre setup",
    "tablet for students",
    "console for family gaming",
    "laptop with best keyboard",
    "smartphone with best display",
    "headphones for commuting",
    "mouse for FPS gaming",
    "keyboard with wireless connectivity",
    "monitor for programmer",
    "camera for wildlife photography",
    "speaker for outdoor party",
    "tablet for reading and browsing",
    "console for exclusive games",
    "laptop for machine learning",
    "smartphone for content creators",
    "noise cancelling headphones for office",
    "silent mechanical keyboard for open office",
    "trackball mouse for precision work",
    "curved monitor for immersive gaming",
    "mirrorless camera for portraits",
    "waterproof speaker for pool",
    "tablet with cellular connectivity",
]


def calibrate_selectivity_filter(conn, target: float, total_rows: int):
    """Return (category, max_price, min_rating, actual_sel, n_filtered) for target."""
    cat, mp, mr = SELECTIVITY_CONFIGS[target]
    actual_count = get_filtered_row_count(conn, cat, mp, mr)
    actual_sel = actual_count / total_rows
    return cat, mp, mr, actual_sel, actual_count


def main():
    parser = argparse.ArgumentParser(description="Exp 01: Selectivity vs Latency (RQ1)")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--n-warmup",  type=int, default=5)
    parser.add_argument("--top-k",     type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=40)
    args = parser.parse_args()

    db_cfg = DBConfig()
    data_cfg = DataConfig(n_rows=50_000)
    bench_cfg = BenchmarkConfig(
        top_k=args.top_k,
        n_queries=args.n_queries,
        n_warmup=args.n_warmup,
        strategy="both",
        index_type="hnsw",
        hnsw=HNSWConfig(ef_search=args.ef_search),
    )

    print("[exp_01] Loading saved data...")
    df, embeddings = load_saved(data_cfg)

    print("[exp_01] Loading embedding model for query encoding...")
    model = SentenceTransformer(data_cfg.embedding_model)
    query_embeddings = model.encode(
        QUERY_TEXTS[:args.n_queries + args.n_warmup],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    results = []

    with connection(db_cfg) as conn:
        total_rows = get_table_row_count(conn)
        runner = BenchmarkRunner(conn, bench_cfg, db_cfg, total_rows)

        for target_sel in SELECTIVITY_LEVELS:
            cat, mp, mr, actual_sel, n_filtered = calibrate_selectivity_filter(
                conn, target_sel, total_rows
            )
            print(
                f"\n[exp_01] Selectivity target={target_sel*100:.0f}%  "
                f"actual={actual_sel*100:.2f}%  ({n_filtered:,} rows)"
            )

            exp_result = runner.run_experiment(
                query_embeddings=list(query_embeddings),
                category=cat,
                max_price=mp,
                min_rating=mr,
                top_k=args.top_k,
                label=f"selectivity_{target_sel}",
            )

            results.append(
                {
                    "target_selectivity": target_sel,
                    "actual_selectivity": exp_result.selectivity_actual,
                    "n_filtered": exp_result.n_filtered,
                    "total_rows": total_rows,
                    "config": exp_result.config,
                    "strategy_a": exp_result.strategy_a,
                    "strategy_b": exp_result.strategy_b,
                }
            )

            if exp_result.strategy_a and exp_result.strategy_b:
                a_mean = exp_result.strategy_a.get("mean_ms", 0)
                b_mean = exp_result.strategy_b.get("mean_ms", 0)
                print(
                    f"         Strategy A mean={a_mean:.1f}ms  "
                    f"Strategy B mean={b_mean:.1f}ms  "
                    f"speedup={'A' if a_mean < b_mean else 'B'}={max(a_mean,b_mean)/max(min(a_mean,b_mean),0.001):.2f}x"
                )

    output = {
        "experiment": "exp_01_selectivity",
        "timestamp": datetime.utcnow().isoformat(),
        "benchmark_config": {
            "n_rows": data_cfg.n_rows,
            "n_queries": args.n_queries,
            "n_warmup": args.n_warmup,
            "top_k": args.top_k,
            "index_type": "hnsw",
            "ef_search": args.ef_search,
        },
        "results": results,
    }

    out_path = Path("results/exp_01_selectivity.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[exp_01] Results saved to {out_path}")


if __name__ == "__main__":
    main()
