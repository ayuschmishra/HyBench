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
import platform
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
from benchmark.metrics import compute_recall_at_k

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
    "laptop for machine learning training",
    "business laptop enterprise security",
    "laptop with OLED display",
    "silent fanless laptop coding",
    "laptop large RAM capacity",
    "convertible 2-in-1 laptop tablet",
    "laptop long battery life over 12 hours",
    "laptop with best keyboard typing",
    "high refresh rate laptop gaming",
    "laptop for data science Python",
    "compact 13 inch ultrabook",
    "laptop multiple monitor support",
    "powerful laptop under 80000 rupees",
    "laptop for graphic design illustration",
    "AMD Ryzen laptop best value",
    "laptop fast SSD NVMe storage",
    "laptop with webcam quality streaming",
    "quiet laptop library silent use",
    "laptop dual GPU rendering",
    "sturdy laptop build quality premium",
    "laptop with best speakers audio",
    "17 inch large screen laptop",
    "laptop USB-C charging universal",
    "laptop for software development IDE",
    "lightweight laptop under 1.5 kg",
    "laptop for music production DAW",
    "laptop DDR5 fast memory",
    "touchscreen laptop creative work",
    "laptop for competitive programming",
    "laptop with mini LED display",
    "affordable gaming laptop budget",
    "laptop for video conferencing remote work",
    "laptop with good cooling system",
    "laptop for 3D modelling CAD",
    "laptop upgradeable RAM storage",
    "Intel Core Ultra laptop AI",
    "laptop for photo editing Lightroom",
    "laptop matte anti-glare display",
    "laptop for college engineering student",
    "laptop PCIe Gen 5 SSD fast",
    "slim laptop powerful processor",
    "laptop for cybersecurity penetration testing",
    "laptop with fingerprint reader security",
    "laptop for cloud computing development",
    "laptop QHD 2K display resolution",
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
    "gaming console family entertainment",
    "4K monitor photo video editing",
    "mechanical keyboard cherry switches",
    "ergonomic mouse wrist pain",
    "portable bluetooth speaker outdoor",
    "mirrorless camera travel photography",
    "gaming headset microphone quality",
    "ultrawide monitor productivity",
    "wireless keyboard mouse combo",
    "smart home speaker multi-room",
    "action camera underwater waterproof",
    "curved gaming monitor immersive",
    "studio monitor headphones mixing",
    "compact keyboard travel portable",
    "vertical mouse ergonomic design",
    "soundbar home theatre surround",
    "DSLR camera wildlife bird photography",
    "gaming mouse lightweight competitive",
    "high refresh rate monitor esports",
    "tablet digital art drawing stylus",
    "console exclusive games library",
    "bookshelf speakers audiophile quality",
    "webcam streaming content creation",
    "trackball mouse precision CAD",
    "gaming keyboard RGB customizable",
    "smartphone best battery life",
    "camera lens portrait photography",
    "monitor eye care blue light",
    "true wireless earbuds sport fitness",
    "speaker party outdoor waterproof loud",
    "tablet with cellular 5G connectivity",
    "retro gaming console classic games",
    "monitor USB-C docking station",
    "over-ear headphones comfort long wear",
    "silent keyboard office quiet",
    "smartphone foldable flexible display",
    "camera video recording vlogging",
    "monitor HDR content creation",
    "mouse wireless multi-device bluetooth",
    "speaker voice control smart home",
    "tablet productivity office work",
    "headphones spatial audio Dolby Atmos",
    "keyboard hot-swappable switches custom",
    "monitor mini-LED local dimming",
    "camera medium format professional",
]


def main():
    parser = argparse.ArgumentParser(description="Exp 02: Adaptive Selector Evaluation (RQ2)")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=ADAPTIVE_THRESHOLD)
    parser.add_argument("--n-rows", type=int, default=50_000,
                        help="Dataset size to benchmark against (default: 50000)")
    args = parser.parse_args()

    db_cfg = DBConfig()
    data_cfg = DataConfig(n_rows=args.n_rows)

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

    def discard_session(conn) -> None:
        """Clear PostgreSQL session state (prepared plans, GUCs) between passes."""
        with conn.cursor() as _cur:
            _cur.execute("DISCARD ALL")
        conn.commit()

    results = []
    pg_version = ""

    with connection(db_cfg) as conn:
        with conn.cursor() as _cur:
            _cur.execute("SELECT version()")
            pg_version = _cur.fetchone()[0]

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
            # DISCARD ALL clears session GUCs and prepared statement caches between passes.

            # --- Fixed-A pass ---
            discard_session(conn)
            for q in warmup_q:
                runner.run_strategy_a(q, cat, mp, mr, args.top_k)
            lat_a = []
            recall_a = []
            for q in run_q:
                gt_ids = runner.compute_ground_truth(q, cat, mp, mr, args.top_k)
                res_a = runner.run_strategy_a(q, cat, mp, mr, args.top_k)
                lat_a.append(res_a.latency_s * 1000)
                recall_a.append(compute_recall_at_k(res_a.ids, gt_ids, args.top_k))

            # --- Fixed-B pass ---
            discard_session(conn)
            for q in warmup_q:
                runner.run_strategy_b(q, cat, mp, mr, args.top_k)
            lat_b = []
            recall_b = []
            for q in run_q:
                gt_ids = runner.compute_ground_truth(q, cat, mp, mr, args.top_k)
                res_b = runner.run_strategy_b(q, cat, mp, mr, args.top_k)
                lat_b.append(res_b.latency_s * 1000)
                recall_b.append(compute_recall_at_k(res_b.ids, gt_ids, args.top_k))

            # Oracle: per-query min of the two fixed-condition latencies above
            lat_oracle = [min(a, b) for a, b in zip(lat_a, lat_b)]

            # --- Adaptive pass (separate warmup so no cache carryover) ---
            discard_session(conn)
            for q in warmup_q:
                execute_adaptive(runner, q, cat, mp, mr, total_rows, args.top_k, args.threshold)

            lat_adapt, adapt_choices, probe_times, recall_adapt = [], [], [], []
            for q in run_q:
                gt_ids = runner.compute_ground_truth(q, cat, mp, mr, args.top_k)
                res_ad, sigma_ad, choice, probe_s = execute_adaptive(
                    runner, q, cat, mp, mr, total_rows, args.top_k, args.threshold
                )
                lat_adapt.append(res_ad.latency_s * 1000)
                adapt_choices.append(choice)
                probe_times.append(probe_s * 1000)
                recall_adapt.append(compute_recall_at_k(res_ad.ids, gt_ids, args.top_k))

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
                "recall_a": float(np.mean(recall_a)),
                "recall_b": float(np.mean(recall_b)),
                "recall_adaptive": float(np.mean(recall_adapt)),
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
            "platform": platform.platform(),
            "python_version": sys.version,
            "pg_version": pg_version,
        },
        "results": results,
    }

    out_path = Path("results/exp_02_adaptive.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[exp_02] Results saved to {out_path}")


if __name__ == "__main__":
    main()
