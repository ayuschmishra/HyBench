"""
Synthetic product catalogue generator for HyBench.

Generates N product records with:
  - Realistic relational attributes (category, price, brand, rating)
  - Templated natural-language descriptions
  - 384-dimensional sentence embeddings (all-MiniLM-L6-v2)

Output:
  data/synthetic/products.csv       — metadata (no embeddings)
  data/synthetic/embeddings.npy     — float32 array of shape (N, 384)

The CSV and embedding array share the same row order; row i in the CSV
corresponds to row i of the embedding array.
"""

import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from faker import Faker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.config import (
    CATEGORY_BRANDS,
    CATEGORY_PROFILES,
    DataConfig,
    DBConfig,
    HNSWConfig,
    IVFFlatConfig,
)
from benchmark.db import ensure_vector_index, get_connection

fake = Faker("en_IN")

# ---------------------------------------------------------------------------
# Description templates — keyed by category
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Laptop": [
        "{adj} {size}-inch laptop powered by {cpu} with {ram}GB RAM and {storage}GB {storage_type} storage. "
        "Features a {display_type} display at {res} resolution and {battery}Wh battery life. "
        "Designed for {use_case} users who demand {perf_adj} performance.",

        "Ultra-{form} {brand} laptop with {cpu} processor, {ram}GB {ram_type} memory, and {storage}GB {storage_type}. "
        "{display_type} {size}-inch display with {res} resolution and {hz}Hz refresh rate. "
        "{battery}Wh battery, {port} connectivity, and {weight}kg chassis weight.",
    ],
    "Smartphone": [
        "{size}-inch {display_type} smartphone with {cpu} chipset, {ram}GB RAM, and {storage}GB internal storage. "
        "{camera}MP triple camera system and {battery}mAh battery with {charge}W fast charging. "
        "Runs on {os} with {extra} connectivity.",

        "Flagship {brand} smartphone featuring {camera}MP camera, {ram}GB RAM, {storage}GB storage. "
        "{battery}mAh battery and {charge}W charging. {display_type} {size}-inch display at {res}.",
    ],
    "Tablet": [
        "{size}-inch {display_type} tablet with {cpu} chip, {ram}GB RAM, and {storage}GB storage. "
        "Battery rated at {battery}mAh supports up to {bat_hours} hours of video playback. "
        "Supports {stylus} and {keyboard} accessories.",

        "Lightweight {brand} tablet at {weight}kg with {size}-inch {display_type} display and {res} resolution. "
        "{cpu} processor with {ram}GB RAM and {storage}GB storage.",
    ],
    "Headphones": [
        "{type} headphones with {driver}mm drivers, {freq} Hz frequency response, and {anc} noise cancellation. "
        "Up to {battery} hours playback on a single charge. {connect} connectivity with {extra} codec support.",

        "{brand} {type} headphones delivering {sound_sig} sound signature. "
        "{driver}mm drivers, {anc} ANC, {battery}-hour battery, and {connect} pairing.",
    ],
    "Gaming Console": [
        "Next-generation {brand} gaming console with {cpu} custom CPU and {gpu} GPU. "
        "Supports {res} gaming at {hz}Hz with {storage}GB {storage_type} SSD. "
        "Backwards compatible with {compat} titles.",

        "{brand} console delivering {res} at {hz}fps with {storage}GB SSD. "
        "{cpu} processor and {gpu} GPU architecture. {extra} controller included.",
    ],
    "Monitor": [
        "{size}-inch {panel} monitor with {res} resolution and {hz}Hz refresh rate. "
        "{response}ms response time, {brightness}nits brightness, and {contrast}:1 contrast ratio. "
        "{hdr} support and {ports} connectivity.",

        "{brand} {size}-inch gaming monitor with {panel} panel, {hz}Hz refresh, {res} resolution. "
        "{response}ms response time and {sync} synchronisation technology.",
    ],
    "Keyboard": [
        "{type} mechanical keyboard with {switch} switches and {layout} layout. "
        "Features {backlight} backlighting, {connect} connectivity, and {build} construction. "
        "N-key rollover and {macro} macro programmability.",

        "Compact {layout} {brand} keyboard using {switch} switches. "
        "{backlight} RGB lighting, {connect} connectivity, {battery}mAh battery for wireless mode.",
    ],
    "Mouse": [
        "{dpi}DPI optical gaming mouse with {sensor} sensor and {buttons} programmable buttons. "
        "{weight}g lightweight design with {connect} connectivity. "
        "Polling rate of {polling}Hz for {extra} accuracy.",

        "{brand} {type} mouse with {sensor} sensor, {dpi}DPI, {buttons} buttons, {weight}g weight. "
        "{connect} connectivity and {battery}-hour wireless battery life.",
    ],
    "Camera": [
        "{megapixels}MP {type} camera with {sensor_size} sensor and {iso} ISO range. "
        "{fps}fps video recording, {evf} viewfinder, and {stab} image stabilisation. "
        "Compatible with {mount} mount lenses.",

        "{brand} {type} with {megapixels}MP resolution, {fps}fps 4K video, {iso} ISO, "
        "and {stab} stabilisation. {battery} shots per charge.",
    ],
    "Speaker": [
        "{watts}W {type} speaker with {drivers} driver configuration and {freq}Hz frequency response. "
        "{connect} connectivity, {battery}-hour battery, and {extra} waterproofing rating. "
        "Built-in {assistant} voice assistant support.",

        "Portable {brand} speaker delivering {watts}W output. "
        "{connect} audio, {battery}-hour playback, {waterproof} resistance.",
    ],
}

# ---------------------------------------------------------------------------
# Attribute pools per category
# ---------------------------------------------------------------------------

ATTRS = {
    "Laptop": {
        "adj":          ["Professional", "Gaming", "Ultrabook", "Business", "Creator"],
        "size":         [13, 14, 15, 16, 17],
        "cpu":          ["Intel Core i5-13500H", "Intel Core i7-13700H", "AMD Ryzen 5 7640HS",
                         "AMD Ryzen 7 7745HX", "Apple M3", "Apple M3 Pro", "Intel Core Ultra 7"],
        "ram":          [8, 16, 32, 64],
        "ram_type":     ["DDR5", "DDR4", "LPDDR5X"],
        "storage":      [256, 512, 1024, 2048],
        "storage_type": ["NVMe SSD", "PCIe 4.0 SSD", "PCIe 5.0 SSD"],
        "display_type": ["IPS", "OLED", "Mini-LED", "AMOLED"],
        "res":          ["1920x1080", "2560x1440", "3840x2160", "2880x1800"],
        "battery":      [45, 56, 72, 86, 99],
        "use_case":     ["gaming", "creative professionals", "business", "students", "developers"],
        "perf_adj":     ["exceptional", "reliable", "portable", "consistent"],
        "form":         ["slim", "portable", "compact", "lightweight"],
        "port":         ["Thunderbolt 4", "USB-C", "USB-A + HDMI", "full-port"],
        "weight":       [1.1, 1.3, 1.5, 1.8, 2.1, 2.4],
        "hz":           [60, 120, 144, 165, 240],
    },
    "Smartphone": {
        "size":         [6.1, 6.4, 6.6, 6.7, 6.8],
        "display_type": ["AMOLED", "Super AMOLED", "LTPO OLED", "ProMotion OLED"],
        "cpu":          ["Snapdragon 8 Gen 3", "Dimensity 9300", "Apple A17 Pro",
                         "Exynos 2400", "Google Tensor G3"],
        "ram":          [6, 8, 12, 16],
        "storage":      [128, 256, 512],
        "camera":       [50, 64, 108, 200],
        "battery":      [4000, 4500, 5000, 5500],
        "charge":       [25, 45, 67, 100, 120],
        "os":           ["Android 14", "iOS 17", "One UI 6", "ColorOS 14"],
        "extra":        ["5G", "Wi-Fi 7", "NFC", "Bluetooth 5.3"],
        "res":          ["1080x2400", "1440x3200", "1080x2340"],
    },
    "Tablet": {
        "size":         [8.3, 10.2, 10.9, 11, 12.4, 13],
        "display_type": ["Liquid Retina", "AMOLED", "IPS LCD", "OLED"],
        "cpu":          ["Apple M2", "Snapdragon 8cx Gen 3", "MediaTek Dimensity 9000"],
        "ram":          [4, 6, 8, 12, 16],
        "storage":      [64, 128, 256, 512],
        "battery":      [7606, 8000, 10090, 11500],
        "bat_hours":    [8, 10, 12, 15],
        "weight":       [0.45, 0.49, 0.55, 0.61, 0.68],
        "res":          ["2360x1640", "2732x2048", "2800x1752"],
        "stylus":       ["Apple Pencil (2nd gen)", "S Pen", "USI stylus"],
        "keyboard":     ["Magic Keyboard", "Book Cover Keyboard", "detachable keyboard"],
    },
    "Headphones": {
        "type":         ["Over-ear", "On-ear", "In-ear", "True wireless"],
        "driver":       [6, 8, 10, 40, 50],
        "freq":         ["20–20,000", "5–40,000", "10–22,000"],
        "anc":          ["active", "adaptive", "hybrid", "passive"],
        "battery":      [6, 8, 20, 30, 36, 40],
        "connect":      ["Bluetooth 5.3", "Bluetooth 5.2 + 3.5mm", "USB-C"],
        "extra":        ["LDAC", "aptX Adaptive", "AAC + SBC"],
        "sound_sig":    ["neutral", "bass-heavy", "V-shaped", "bright"],
    },
    "Gaming Console": {
        "cpu":          ["AMD Zen 2 8-core", "AMD Zen 2 custom", "NVIDIA Tegra T239"],
        "gpu":          ["AMD RDNA 2", "AMD RDNA 3", "NVIDIA Ampere"],
        "res":          ["4K", "1440p", "1080p"],
        "hz":           [60, 120],
        "storage":      [512, 825, 1024],
        "storage_type": ["NVMe", "custom NVMe"],
        "compat":       ["thousands of PS4/PS5", "Xbox 360/One", "Nintendo Switch"],
        "extra":        ["DualSense wireless", "Xbox wireless", "Joy-Con"],
    },
    "Monitor": {
        "size":         [24, 27, 32, 34, 38, 49],
        "panel":        ["IPS", "VA", "OLED", "Mini-LED", "TN"],
        "res":          ["1920x1080", "2560x1440", "3840x2160", "3440x1440", "5120x1440"],
        "hz":           [60, 75, 144, 165, 240, 360],
        "response":     [0.1, 0.5, 1, 2, 4],
        "brightness":   [250, 350, 400, 600, 1000],
        "contrast":     [1000, 3000, 5000, "1M"],
        "hdr":          ["HDR400", "HDR600", "HDR1000", "DisplayHDR True Black"],
        "ports":        ["HDMI 2.1 + DisplayPort 1.4", "USB-C + HDMI", "Thunderbolt 4"],
        "sync":         ["G-Sync Compatible", "AMD FreeSync Premium", "VESA AdaptiveSync"],
    },
    "Keyboard": {
        "type":         ["Full-size", "TKL", "65%", "75%", "60%"],
        "switch":       ["Cherry MX Red", "Gateron Yellow", "Kailh Box Brown",
                         "Topre 45g", "Optical Linear"],
        "layout":       ["QWERTY US", "QWERTY ISO", "Compact 65%"],
        "backlight":    ["per-key RGB", "single-colour white", "RGB zone"],
        "connect":      ["Bluetooth + USB-C", "2.4GHz wireless", "USB-C wired"],
        "build":        ["aluminium", "polycarbonate", "ABS", "gasket-mounted"],
        "macro":        ["full", "limited", "no"],
        "battery":      [2000, 4000, 6000],
    },
    "Mouse": {
        "dpi":          [400, 800, 1600, 3200, 6400, 12000, 25600],
        "sensor":       ["PixArt PAW3395", "PixArt PMW3335", "Razer Focus Pro 30K"],
        "buttons":      [5, 6, 7, 8, 11],
        "weight":       [45, 55, 68, 79, 95, 110],
        "connect":      ["2.4GHz wireless", "Bluetooth", "USB-C wired", "multi-device"],
        "polling":      [125, 500, 1000, 4000, 8000],
        "extra":        ["sub-millimetre", "flawless", "competition-grade"],
        "type":         ["gaming", "ergonomic", "ambidextrous", "vertical"],
        "battery":      [60, 70, 80, 100, 140],
    },
    "Camera": {
        "megapixels":   [12, 20, 24, 33, 45, 61, 102],
        "type":         ["mirrorless", "DSLR", "compact", "medium-format"],
        "sensor_size":  ["APS-C", "Full-frame", "Micro Four Thirds", "1-inch"],
        "iso":          ["100–51200", "64–102400", "100–25600"],
        "fps":          [24, 30, 60, 120, 240],
        "evf":          ["3.68M-dot OLED", "5.76M-dot OLED", "optical pentaprism"],
        "stab":         ["5-axis IBIS", "optical", "in-body + lens"],
        "mount":        ["Sony E", "Canon RF", "Nikon Z", "L-mount", "Micro 4/3"],
        "battery":      [350, 500, 610, 740, 900],
    },
    "Speaker": {
        "watts":        [5, 10, 20, 30, 50, 100, 200],
        "type":         ["portable Bluetooth", "bookshelf", "soundbar", "smart home"],
        "drivers":      ["2.0", "2.1", "3-way", "5.1"],
        "freq":         ["20–20,000", "45–20,000", "60–20,000"],
        "connect":      ["Bluetooth 5.3", "Wi-Fi + Bluetooth", "AirPlay 2 + Bluetooth"],
        "battery":      [8, 12, 20, 24, 36],
        "extra":        ["IP67", "IPX5", "IP55"],
        "waterproof":   ["IP67", "IPX5", "IP55", "IP44"],
        "assistant":    ["Google Assistant", "Amazon Alexa", "Apple Siri"],
    },
}


def _pick(pool):
    return random.choice(pool)


def _generate_description(category: str, brand: str) -> str:
    template = random.choice(TEMPLATES[category])
    attrs = ATTRS[category]

    ctx = {"brand": brand}
    for k, v in attrs.items():
        ctx[k] = _pick(v)

    try:
        return template.format(**ctx)
    except KeyError:
        return f"{brand} {category} — a premium product with advanced specifications."


def generate_products(cfg: DataConfig) -> pd.DataFrame:
    random.seed(cfg.random_seed)
    fake.seed_instance(cfg.random_seed)

    categories = list(CATEGORY_PROFILES.keys())
    rows_per_cat = cfg.n_rows // len(categories)
    remainder = cfg.n_rows - rows_per_cat * len(categories)

    records = []
    for i, category in enumerate(categories):
        n = rows_per_cat + (1 if i < remainder else 0)
        profile = CATEGORY_PROFILES[category]
        brands = CATEGORY_BRANDS[category]
        p_min, p_max = profile["price_range"]
        r_min, r_max = profile["rating_range"]

        for _ in range(n):
            brand = _pick(brands)
            price = round(random.uniform(p_min, p_max), 2)
            rating = round(random.uniform(r_min, r_max), 2)
            description = _generate_description(category, brand)
            records.append(
                {
                    "category": category,
                    "price": price,
                    "brand": brand,
                    "rating": rating,
                    "description": description,
                }
            )

    random.shuffle(records)
    return pd.DataFrame(records)


def embed_descriptions(
    descriptions: List[str],
    cfg: DataConfig,
) -> np.ndarray:
    # No local_files_only here: data generation is the bootstrap step that
    # downloads/caches the model on first run. Experiments load it afterwards
    # with local_files_only=True so timed runs never touch the network.
    model = SentenceTransformer(cfg.embedding_model)
    embeddings = model.encode(
        descriptions,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def generate_and_save(cfg: DataConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "products.csv"
    emb_path = out_dir / "embeddings.npy"

    print(f"[generator] Generating {cfg.n_rows:,} product records...")
    t0 = time.perf_counter()
    df = generate_products(cfg)
    print(f"[generator] Records generated in {time.perf_counter() - t0:.1f}s")

    print(f"[generator] Embedding {cfg.n_rows:,} descriptions with {cfg.embedding_model}...")
    t1 = time.perf_counter()
    embeddings = embed_descriptions(df["description"].tolist(), cfg)
    print(f"[generator] Embeddings computed in {time.perf_counter() - t1:.1f}s")

    df.to_csv(csv_path, index=False)
    np.save(emb_path, embeddings)
    print(f"[generator] Saved CSV  → {csv_path}")
    print(f"[generator] Saved embs → {emb_path}  shape={embeddings.shape}")

    return df, embeddings


def load_saved(cfg: DataConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    out_dir = Path(cfg.output_dir)
    df = pd.read_csv(out_dir / "products.csv")
    embeddings = np.load(out_dir / "embeddings.npy")
    return df, embeddings


def load_to_db(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    db_cfg: DBConfig,
    index_type: str = "hnsw",
) -> None:
    """Insert generated products + embeddings into the products table."""
    conn = get_connection(db_cfg)
    cur = conn.cursor()

    print("[generator] Truncating products table...")
    cur.execute("TRUNCATE TABLE products RESTART IDENTITY;")

    print("[generator] Dropping ANN indexes for fast bulk load...")
    cur.execute("DROP INDEX IF EXISTS idx_products_hnsw;")
    cur.execute("DROP INDEX IF EXISTS idx_products_ivfflat;")

    print(f"[generator] Inserting {len(df):,} rows (batch size 1000)...")
    t0 = time.perf_counter()

    rows = [
        (
            row["category"],
            float(row["price"]),
            row["brand"],
            float(row["rating"]),
            row["description"],
            embeddings[i].tolist(),
        )
        for i, row in df.iterrows()
    ]

    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO products (category, price, brand, rating, description, embedding)
        VALUES %s
        """,
        rows,
        template="(%s, %s, %s, %s, %s, %s::vector)",
        page_size=1000,
    )
    print(f"[generator] Inserted {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    print(f"[generator] Building {index_type} index...")
    info = ensure_vector_index(
        conn, index_type, HNSWConfig(), IVFFlatConfig(), n_rows=len(df)
    )
    print(
        f"[generator] {info['name']} built in {info['build_seconds']:.1f}s "
        f"(params={info['params']})"
    )

    # ensure_vector_index runs ANALYZE after a rebuild, but run it explicitly
    # so pg_stats (needed by PgStatsEstimator) is populated even on the
    # built=False path.
    cur.execute("ANALYZE products;")

    cur.close()
    conn.close()
    print("[generator] Database load complete.")


def write_checksums(data_dir: Path) -> None:
    """Write SHA-256 checksums of products.csv and embeddings.npy."""
    import hashlib
    checksum_path = data_dir / "checksums.sha256"
    targets = ["products.csv", "embeddings.npy"]
    lines = []
    for name in targets:
        p = data_dir / name
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {name}")
    checksum_path.write_text("\n".join(lines) + "\n")
    print(f"[generator] Checksums written → {checksum_path}")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="Generate HyBench synthetic dataset")
    _parser.add_argument("--n-rows", type=int, default=50_000,
                         help="Number of product rows (default: 50000)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Random seed (default: 42)")
    _parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw",
                         help="ANN index to build after load (default: hnsw)")
    _args = _parser.parse_args()

    cfg = DataConfig(n_rows=_args.n_rows, random_seed=_args.seed)
    db_cfg = DBConfig()
    df, embeddings = generate_and_save(cfg)
    load_to_db(df, embeddings, db_cfg, index_type=_args.index_type)
    write_checksums(Path(cfg.output_dir))
