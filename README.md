# HyBench v0.1

**A Reproducible Experimental Framework for Hybrid Relational–Vector Query Processing in PostgreSQL**

> *Empirically characterising filter-selectivity effects on hybrid query latency and validating a lightweight selectivity-aware execution strategy selector.*

---

## Problem Statement

Modern AI-native applications issue queries that combine relational predicates with dense vector similarity search. A representative example:

> *"Find highly-rated laptops under ₹90,000 semantically similar to 'ASUS ROG Zephyrus'."*

This decomposes into:
- A relational filter: `category = 'Laptop' AND price < 90000 AND rating > 4.0`
- A semantic retrieval: k-nearest neighbours of a query embedding in 384-dimensional space

PostgreSQL + pgvector can answer both components, but the execution order — apply the vector index first or the relational predicate first — has a profound impact on latency that depends on filter selectivity. HyBench v0.1 studies this empirically.

---

## Research Questions

| ID | Question | Experiment |
|----|----------|------------|
| **RQ1** | How does filter selectivity affect hybrid query latency for Strategy A (vector-first) and Strategy B (filter-first)? | `exp_01_selectivity.py` |
| **RQ2** | Does a lightweight single-threshold selectivity-aware strategy selector achieve near-oracle latency? | `exp_02_adaptive.py` |

---

## Execution Strategies

**Strategy A — Vector-first (HNSW scan -> relational post-filter)**

```sql
WITH vector_candidates AS (
    SELECT id, category, price, brand, rating,
           embedding <=> %(qvec)s::vector AS distance
    FROM products
    ORDER BY embedding <=> %(qvec)s::vector
    LIMIT %(n_candidates)s
)
SELECT * FROM vector_candidates
WHERE price < %(max_price)s
  AND rating > %(min_rating)s
  AND (%(category)s::text IS NULL OR category = %(category)s::text)
ORDER BY distance
LIMIT %(top_k)s;
```

Uses the HNSW index. Fast at low selectivity (few rows match the filter). Degrades when the filtered set is large because many candidates are post-filtered away.

**Strategy B — Filter-first (B-tree predicate -> exact KNN on materialised set)**

```sql
WITH filtered AS MATERIALIZED (
    SELECT id, category, price, brand, rating, embedding
    FROM products
    WHERE price < %(max_price)s
      AND rating > %(min_rating)s
      AND (%(category)s::text IS NULL OR category = %(category)s::text)
)
SELECT id, category, price, brand, rating,
       embedding <=> %(qvec)s::vector AS distance
FROM filtered
ORDER BY embedding <=> %(qvec)s::vector
LIMIT %(top_k)s;
```

Uses B-tree indexes for the predicate, then performs exact KNN on the materialised filtered set. Always returns Recall@K = 1.0. Latency grows linearly with filtered set size.

> **Implementation note:** The `AS MATERIALIZED` hint is critical. Without it, PostgreSQL 16 inlines the CTE and pushes `ORDER BY embedding <=> ...` into the HNSW index, defeating the filter-first design entirely and producing 0 results at low selectivity.

---

## Primary Contribution: Lightweight Strategy Selector

`benchmark/planner.py` implements a deterministic, single-threshold strategy selector:

1. **Estimate selectivity** σ via `SELECT COUNT(*)` with the query predicate
2. **Compare** σ to calibrated threshold θ* (derived from Experiment 1)
3. **Dispatch**: if σ < θ*, choose Strategy B (filter-first); otherwise choose Strategy A (vector-first)

Properties:
- **Deterministic** — same selectivity always produces the same decision
- **Explainable** — every decision is a single comparison: `σ < θ*`
- **Single-parameter** — θ* = 0.05 (5%), one calibrated constant
- **Extensible** — the `SelectivityEstimator` protocol accepts drop-in estimators (e.g., `pg_stats`)

---

## Empirical Results

### Experiment 1 — Selectivity vs. Latency (RQ1)

Dataset: 50K rows · HNSW m=16 ef_search=40 · K=10 · n=50 queries per level · warm cache

| Selectivity | Actual σ | Strategy A | Strategy B | Winner | Speedup |
|-------------|----------|-----------|-----------|--------|---------|
| 1%  | 1.11% | 46.6 ms  | 3.3 ms   | **B** | 14x |
| 5%  | 5.00% | 20.2 ms  | 21.5 ms  | A     | 1.1x |
| 10% | 10.0% | 22.0 ms  | 31.2 ms  | A     | 1.4x |
| 25% | 25.0% | 21.8 ms  | 75.9 ms  | A     | 3.5x |
| 50% | 50.0% | 15.5 ms  | 102.2 ms | A     | 6.6x |
| 75% | 75.0% | 15.9 ms  | 139.4 ms | A     | 8.8x |

**Crossover θ* ≈ 4.6%** (linear interpolation); calibrated threshold set to **5%**.

### Experiment 2 — Adaptive Selector vs. Oracle (RQ2)

Dataset: same setup · 4 conditions: Fixed-A, Fixed-B, Adaptive, Oracle (retrospective min(A,B) per query)

| Selectivity | Fixed-A  | Fixed-B   | **Adaptive** | Oracle  | Oracle gap |
|-------------|----------|-----------|------------|---------|------------|
| 1%  | 7.5 ms   | 2.9 ms    | **2.8 ms** | 2.9 ms  | -3.6% |
| 5%  | 6.8 ms   | 19.9 ms   | **6.9 ms** | 6.8 ms  | +1.0% |
| 10% | 6.5 ms   | 29.8 ms   | **7.6 ms** | 6.5 ms  | +16.1% |
| 25% | 7.4 ms   | 90.9 ms   | **6.9 ms** | 7.4 ms  | -7.2% |
| 50% | 6.3 ms   | 120.1 ms  | **6.1 ms** | 6.3 ms  | -3.4% |
| 75% | 6.2 ms   | 143.5 ms  | **6.4 ms** | 6.2 ms  | +4.0% |

Adaptive made the correct strategy decision at all 6 selectivity levels. Oracle gap is within +/-16% at every level.

---

## Quick Start

### Prerequisites

- PostgreSQL 16 with pgvector 0.7+
- Python 3.10+
- `all-MiniLM-L6-v2` model (downloaded automatically on first run by `sentence-transformers`)

### 1. Install Python dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Create the database

```bash
createdb hybench
psql -d hybench -c "CREATE USER hybench WITH PASSWORD 'hybench'; GRANT ALL ON DATABASE hybench TO hybench;"
psql -d hybench -U hybench -f sql/create_tables.sql
```

### 3. Generate the dataset and build the HNSW index

```bash
python data_gen/generator.py
```

Generates 50,000 synthetic product rows, encodes 384-dimensional embeddings, loads them into PostgreSQL, and builds the HNSW index (m=16, ef_construction=64). Expect 15–45 minutes depending on hardware.

### 4. Run experiments

```bash
# Experiment 1: selectivity vs. latency (RQ1)
python experiments/exp_01_selectivity.py --n-queries 50 --n-warmup 5

# Experiment 2: adaptive selector evaluation (RQ2)
python experiments/exp_02_adaptive.py --n-queries 50 --n-warmup 5
```

### 5. Generate figures

```bash
python analysis/plot_results.py
```

Output saved to `figures/`:
- `fig_01_latency_vs_selectivity.png` — Strategy A vs. B across selectivity levels with θ* crossover annotation
- `fig_02_adaptive_vs_fixed.png` — Adaptive vs. Fixed-A, Fixed-B, and Oracle lower bound

---

## Dataset

Synthetically generated electronics product catalogue (seed = 42, fully reproducible).

| Column | Type | Notes |
|--------|------|-------|
| `id` | SERIAL | Primary key |
| `category` | VARCHAR(50) | 10 categories, 5,000 rows each |
| `price` | DECIMAL(10,2) | Category-specific INR ranges |
| `brand` | VARCHAR(100) | Realistic brand names |
| `rating` | DECIMAL(3,2) | 2.0–5.0 |
| `description` | TEXT | Templated 3–5 sentence product description |
| `embedding` | vector(384) | all-MiniLM-L6-v2 normalised embedding |

Categories (10 × 5,000 rows): Camera, Gaming Console, Headphones, Keyboard, Laptop, Monitor, Mouse, Smartphone, Speaker, Tablet.

Six pre-calibrated `(category, max_price, min_rating)` configurations produce actual selectivity σ in {1.1%, 5%, 10%, 25%, 50%, 75%}. Actual σ is measured via `COUNT(*)` at runtime and recorded in every result file.

---

## Metrics

| Metric | Definition |
|--------|------------|
| **Mean latency** | Average wall-clock time over n=50 queries, warmup excluded |
| **P95 latency** | 95th-percentile execution time |
| **Recall@K** | `|Strategy results ∩ Ground truth| / K`; ground truth = exact KNN within filtered set |
| **Oracle gap** | `(Adaptive − Oracle) / Oracle × 100%` |

---

## Project Structure

```
HyBench/
│
├── benchmark/
│   ├── config.py               All experiment parameters (theta*, HNSW config, data sizes)
│   ├── db.py                   PostgreSQL connection, timed query utilities
│   ├── runner.py               Strategy A and B SQL + ground truth computation
│   ├── planner.py              Selectivity-aware strategy selector
│   └── metrics.py              Recall@K and latency aggregation
│
├── data_gen/
│   └── generator.py            50K synthetic row generation (seed=42)
│
├── sql/
│   └── create_tables.sql       Schema, pgvector extension, B-tree and HNSW indexes
│
├── experiments/
│   ├── exp_01_selectivity.py   RQ1: latency across 6 selectivity levels
│   └── exp_02_adaptive.py      RQ2: adaptive vs. fixed strategies vs. oracle
│
├── analysis/
│   └── plot_results.py         Two 300 DPI publication-quality figures
│
├── data/synthetic/
│   ├── products.csv            50K rows (seed=42)
│   └── embeddings.npy          50K x 384 float32
│
├── results/                    JSON outputs (exp_01_selectivity.json, exp_02_adaptive.json)
├── figures/                    Generated PNG figures
└── report/
    └── RESEARCH_PLAN.md        Full methodology and research plan
```

---

## Configuration Reference

All parameters are in `benchmark/config.py`:

```python
ADAPTIVE_THRESHOLD: float = 0.05   # theta* — calibrated from exp_01 crossover at ~4.6%

class HNSWConfig:
    m: int = 16
    ef_construction: int = 64
    ef_search: int = 40            # overridden per-query as max(40, n_candidates)

class DataConfig:
    n_rows: int = 50_000
    embedding_model: str = "all-MiniLM-L6-v2"
    random_seed: int = 42

class BenchmarkConfig:
    n_queries: int = 50
    n_warmup: int = 5
    top_k: int = 10
    n_candidates: int = 1_000      # Strategy A HNSW scan budget
```

---

## Known Limitations (v0.1 Scope)

| Limitation | Plan |
|-----------|------|
| IVFFlat index not implemented | v0.2 |
| Dataset capped at 50K rows | v0.2 |
| COUNT(*) probe adds ~1–5 ms per query | v0.2: pg_stats estimator |
| No memory profiling | v0.2 |
| No concurrent client benchmarking | v0.5 |
| Synthetic data only | v0.5: real datasets |

---

## Author

Ayush Mishra · IIT Madras · B.S. Data Science · `23f2003585@ds.study.iitm.ac.in`
