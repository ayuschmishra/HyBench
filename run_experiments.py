"""
HyBench v0.1 — master experiment runner.

Runs the two experiments in sequence and then generates both figures.
θ* is derived from Experiment 1 results and injected into config before
Experiment 2 runs.

Usage:
    python run_experiments.py                   # full run (50K rows)
    python run_experiments.py --skip-data-gen   # skip data generation (data already loaded)
    python run_experiments.py --exp 1           # only Experiment 1
    python run_experiments.py --exp 2           # only Experiment 2 (requires exp_01 output)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"[run] {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable] + cmd, check=True)
    return result


def derive_theta_star(results_path: Path) -> float:
    """
    Estimate θ* from exp_01 by finding the selectivity at which Strategy A
    and Strategy B have equal mean latency (linear interpolation).
    Returns the threshold as a fraction (0–1).
    """
    with open(results_path) as f:
        data = json.load(f)

    rows = sorted(data["results"], key=lambda r: r["actual_selectivity"])
    sels   = [r["actual_selectivity"] for r in rows]
    a_mean = [r["strategy_a"].get("mean_ms", 0) if r.get("strategy_a") else 0 for r in rows]
    b_mean = [r["strategy_b"].get("mean_ms", 0) if r.get("strategy_b") else 0 for r in rows]

    for i in range(len(sels) - 1):
        diff_i   = a_mean[i]   - b_mean[i]
        diff_ip1 = a_mean[i+1] - b_mean[i+1]
        if diff_i * diff_ip1 < 0:  # sign flip → crossover
            ds = sels[i+1] - sels[i]
            theta = sels[i] + ds * diff_i / (diff_i - diff_ip1)
            return theta

    # No crossover found; fall back to midpoint
    print("[run] WARNING: no crossover found in exp_01; using θ* = 0.10")
    return 0.10


def patch_adaptive_threshold(theta: float):
    """Rewrite the ADAPTIVE_THRESHOLD line in benchmark/config.py."""
    cfg_path = Path("benchmark/config.py")
    text = cfg_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("ADAPTIVE_THRESHOLD"):
            lines[i] = f"ADAPTIVE_THRESHOLD: float = {theta:.4f}  # calibrated from exp_01"
            break
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[run] ADAPTIVE_THRESHOLD patched to {theta:.4f} ({theta*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="HyBench v0.1 experiment runner")
    parser.add_argument("--skip-data-gen", action="store_true",
                        help="Skip data generation (products table already loaded)")
    parser.add_argument("--exp", type=int, choices=[1, 2],
                        help="Run only this experiment (1 or 2)")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--n-warmup",  type=int, default=5)
    args = parser.parse_args()

    run_exp1 = args.exp is None or args.exp == 1
    run_exp2 = args.exp is None or args.exp == 2

    if not args.skip_data_gen and run_exp1:
        run(["data_gen/generator.py"], "Data generation (50K rows + embeddings)")

    if run_exp1:
        run(
            ["experiments/exp_01_selectivity.py",
             f"--n-queries={args.n_queries}",
             f"--n-warmup={args.n_warmup}"],
            "Experiment 1 — Filter Selectivity vs. Latency (RQ1)",
        )

    if run_exp2:
        exp01_path = Path("results/exp_01_selectivity.json")
        if not exp01_path.exists():
            print("[run] ERROR: results/exp_01_selectivity.json not found — run Experiment 1 first.")
            sys.exit(1)
        theta = derive_theta_star(exp01_path)
        patch_adaptive_threshold(theta)

        run(
            ["experiments/exp_02_adaptive.py",
             f"--n-queries={args.n_queries}",
             f"--n-warmup={args.n_warmup}"],
            "Experiment 2 — Adaptive Selector Evaluation (RQ2)",
        )

    if args.exp is None:
        run(["analysis/plot_results.py"], "Figure generation (Figure 1 + Figure 2)")
        run(["analysis/summarize.py"],    "Results summary")

    print("\n[run] All done.")


if __name__ == "__main__":
    main()
