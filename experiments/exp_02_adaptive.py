"""
Experiment 2 — Adaptive Strategy Selector Evaluation (RQ2)

Conditions evaluated per selectivity level:
  Fixed-A           : always execute Strategy A (vector-first)
  Fixed-B           : always execute Strategy B (filter-first)
  Adaptive          : selector with exact COUNT(*) probe estimator
  Adaptive-pg_stats : selector with pg_stats statistics estimator [v0.2]
  Oracle            : retrospective min(latency_A, latency_B) per query

The two adaptive conditions isolate the effect of selectivity-estimation
quality: the COUNT(*) probe is exact but costs a round-trip; the pg_stats
estimator is ~free after one catalog read but inherits the planner's
attribute-independence error on correlated predicates.

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
from benchmark.db import (
    connection,
    ensure_vector_index,
    get_filtered_row_count,
    get_memory_metrics,
    get_table_row_count,
)
from benchmark.planner import CountProbeEstimator, PgStatsEstimator, execute_adaptive
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
    parser.add_argument("--estimator", choices=["count", "pg_stats", "both"],
                        default="both",
                        help="Selectivity estimator(s) for the adaptive condition "
                             "(default: both — compare probe vs. pg_stats) [v0.2]")
    parser.add_argument("--output", type=Path,
                        default=Path("results/exp_02_adaptive.json"),
                        help="Results path (default: results/exp_02_adaptive.json)")
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

    # Estimator instances are created once so the PgStatsEstimator's cached
    # catalog read persists across selectivity levels (its realistic steady
    # state); the first adaptive-pg_stats warmup query pays the one-off load.
    estimators: dict[str, object] = {}
    if args.estimator in ("count", "both"):
        estimators["count"] = CountProbeEstimator()
    if args.estimator in ("pg_stats", "both"):
        estimators["pg_stats"] = PgStatsEstimator()

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

        # Selector evaluation runs on HNSW; rebuild it if a prior
        # `exp_01 --index-type ivfflat` run replaced it.
        index_info = ensure_vector_index(
            conn, "hnsw", bench_cfg.hnsw, bench_cfg.ivfflat, total_rows
        )
        print(
            f"[exp_02] Index: {index_info['name']} params={index_info['params']} "
            f"({'rebuilt in %.1fs' % index_info['build_seconds'] if index_info['built'] else 'reused'})"
        )

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

            # --- Adaptive passes, one per estimator (separate warmup each,
            #     so no cache carryover between conditions) ---
            def run_adaptive_pass(estimator):
                discard_session(conn)
                for q in warmup_q:
                    execute_adaptive(runner, q, cat, mp, mr, total_rows,
                                     args.top_k, args.threshold, estimator=estimator)
                lats, choices, overheads, recalls, sigmas = [], [], [], [], []
                for q in run_q:
                    gt_ids = runner.compute_ground_truth(q, cat, mp, mr, args.top_k)
                    res_ad, sigma_ad, choice, probe_s = execute_adaptive(
                        runner, q, cat, mp, mr, total_rows,
                        args.top_k, args.threshold, estimator=estimator
                    )
                    lats.append(res_ad.latency_s * 1000)
                    choices.append(choice)
                    overheads.append(probe_s * 1000)
                    recalls.append(compute_recall_at_k(res_ad.ids, gt_ids, args.top_k))
                    sigmas.append(sigma_ad)
                return lats, choices, overheads, recalls, sigmas

            adaptive_passes = {
                name: run_adaptive_pass(est) for name, est in estimators.items()
            }

            def stats(vals):
                arr = np.array(vals)
                return {
                    "mean_ms": float(np.mean(arr)),
                    "median_ms": float(np.median(arr)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "std_ms": float(np.std(arr)),
                }

            row = {
                "target_selectivity": target_sel,
                "actual_selectivity": actual_sel,
                "n_filtered": actual_count,
                "threshold": args.threshold,
                "fixed_a": stats(lat_a),
                "fixed_b": stats(lat_b),
                "oracle": stats(lat_oracle),
                "recall_a": float(np.mean(recall_a)),
                "recall_b": float(np.mean(recall_b)),
            }
            for name, (lats, choices, overheads, recalls, sigmas) in adaptive_passes.items():
                if name == "count":
                    # v0.1-compatible key names (figures/summary read these).
                    key, overhead_key, recall_key = (
                        "adaptive", "probe_overhead_ms", "recall_adaptive"
                    )
                else:
                    key, overhead_key, recall_key = (
                        "adaptive_pgstats",
                        "estimate_overhead_pgstats_ms",
                        "recall_adaptive_pgstats",
                    )
                    # Constant predicate per level, so this is THE estimate;
                    # compare with actual_selectivity for estimation error.
                    row["sigma_pgstats_mean"] = float(np.mean(sigmas))
                row[key] = stats(lats)
                row[f"{key}_choices"] = {"A": choices.count("A"), "B": choices.count("B")}
                row[overhead_key] = stats(overheads)
                row[recall_key] = float(np.mean(recalls))
            results.append(row)

            a_mean = row["fixed_a"]["mean_ms"]
            b_mean = row["fixed_b"]["mean_ms"]
            or_mean = row["oracle"]["mean_ms"]
            line = (
                f"         Fixed-A={a_mean:.1f}ms  Fixed-B={b_mean:.1f}ms  "
                f"Oracle={or_mean:.1f}ms"
            )
            for key, label in (("adaptive", "Adaptive"),
                               ("adaptive_pgstats", "Adaptive-pgstats")):
                if key in row:
                    m = row[key]["mean_ms"]
                    gap = (m - or_mean) / max(or_mean, 0.001) * 100
                    ch = row[f"{key}_choices"]
                    line += f"  {label}={m:.1f}ms (gap {gap:+.1f}%, A:{ch['A']}/B:{ch['B']})"
            print(line)

        memory = get_memory_metrics(conn)

    output = {
        "experiment": "exp_02_adaptive",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmark_config": {
            "n_rows": data_cfg.n_rows,
            "n_queries": n_q,
            "n_warmup": args.n_warmup,
            "top_k": args.top_k,
            "index_type": "hnsw",
            "index_params": index_info["params"],
            "estimator": args.estimator,
            "ef_search": args.ef_search,
            "threshold": args.threshold,
            "platform": platform.platform(),
            "python_version": sys.version,
            "pg_version": pg_version,
        },
        "results": results,
    }

    output["memory"] = memory

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[exp_02] Results saved to {out_path}")


if __name__ == "__main__":
    main()
