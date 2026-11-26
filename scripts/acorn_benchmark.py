#!/usr/bin/env python3
"""
ACORN-1 filtered ANN search benchmark using the Qdrant filtered ANN benchmark datasets.

Downloads a small dataset (100K vectors with metadata predicates), builds a standard
HNSW index (kannolo.build), runs filtered search via ACORN-1 and ACORN-γ, and reports recall
and latency.

Usage:
    python acorn_benchmark.py                          # laion-small-clip (default)
    python acorn_benchmark.py --dataset random_ints    # random integer predicates
    python acorn_benchmark.py --k 10 --ef 200          # tune search params
    python acorn_benchmark.py --n-queries 100          # limit number of test queries
    python acorn_benchmark.py --data-dir /tmp/acorn    # custom download directory
"""

import argparse
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "laion-small-clip": {
        "url": "https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-small-clip.tgz",
        "description": "100K LAION CLIP embeddings (512D), range predicates on a float attribute",
        "metric": "dotproduct",   # cosine on normalised vectors = dot product
    },
    "random_ints": {
        "url": "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_ints_100k.tgz",
        "description": "100K random vectors (2048D), integer range predicates",
        "metric": "dotproduct",
    },
    "random_keywords": {
        "url": "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_keywords_100k.tgz",
        "description": "100K random vectors (2048D), keyword / exact-match predicates",
        "metric": "dotproduct",
    },
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)


def download_and_extract(url: str, dest_dir: Path) -> Path:
    """Download a .tgz dataset and extract it under dest_dir. Returns the dataset dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / Path(url).name

    if not archive_path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, archive_path, reporthook=_progress_hook)
        print()  # newline after progress bar
    else:
        print(f"Archive already exists: {archive_path}")

    # Inspect the archive to find the top-level directory (if any).
    with tarfile.open(archive_path) as tf:
        members = tf.getmembers()
        top_level = {m.name.split("/")[0] for m in members}

    if len(top_level) == 1:
        dataset_dir = dest_dir / top_level.pop()
    else:
        # Files are at the archive root — put them in a named subdirectory.
        # Use the stem without the first suffix only (handles .tgz and .tar.gz).
        stem = archive_path.name
        for ext in (".tgz", ".tar.gz", ".tar"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        dataset_dir = dest_dir / stem

    # Only skip extraction if the key output file is already present.
    vectors_npy = dataset_dir / "vectors.npy"
    if vectors_npy.exists():
        print(f"Already extracted: {dataset_dir}")
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to {dataset_dir} ...")
        with tarfile.open(archive_path) as tf:
            if len(top_level) == 1:
                tf.extractall(dest_dir)
            else:
                # Extract flat archives into the named subdirectory.
                tf.extractall(dataset_dir)

    return dataset_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(dataset_dir: Path):
    """
    Load vectors, payloads, and test queries from a Qdrant benchmark directory.

    Returns:
        vectors   – np.ndarray shape (n, dim), float32
        payloads  – list of dicts, one per vector
        tests     – list of dicts with keys: query, conditions, closest_ids, closest_scores
    """
    vectors_path = dataset_dir / "vectors.npy"
    payloads_path = dataset_dir / "payloads.jsonl"
    tests_path = dataset_dir / "tests.jsonl"

    for p in (vectors_path, payloads_path, tests_path):
        if not p.exists():
            # Some datasets put files in a subdirectory
            candidates = list(dataset_dir.rglob(p.name))
            if not candidates:
                raise FileNotFoundError(f"Expected file not found: {p}")
            p = candidates[0]

    print(f"Loading vectors from {vectors_path} ...")
    vectors = np.load(vectors_path).astype(np.float32)

    print(f"Loading payloads from {payloads_path} ...")
    payloads = load_jsonl(payloads_path)

    print(f"Loading tests from {tests_path} ...")
    tests = load_jsonl(tests_path)

    assert len(vectors) == len(payloads), (
        f"Vector count {len(vectors)} != payload count {len(payloads)}"
    )

    return vectors, payloads, tests


# ---------------------------------------------------------------------------
# Predicate builder
# ---------------------------------------------------------------------------

def _check_condition(payload: dict, condition: dict) -> bool:
    """Evaluate a single Qdrant filter condition against a payload dict."""
    for field_name, filter_spec in condition.items():
        value = payload.get(field_name)
        if value is None:
            return False

        if "range" in filter_spec:
            rng = filter_spec["range"]
            if "gt"  in rng and not (value >  rng["gt"]):  return False
            if "gte" in rng and not (value >= rng["gte"]): return False
            if "lt"  in rng and not (value <  rng["lt"]):  return False
            if "lte" in rng and not (value <= rng["lte"]): return False

        elif "match" in filter_spec:
            if value != filter_spec["match"]["value"]:
                return False

        else:
            raise ValueError(f"Unknown filter spec: {filter_spec}")

    return True


def build_predicate(conditions: dict, payloads: list[dict]):
    """
    Convert a Qdrant conditions dict into a Python callable ``(int) -> bool``.

    Supports the ``{"and": [...]}`` top-level structure and per-field
    ``range`` / ``match`` conditions.
    """
    if "and" in conditions:
        sub_conditions = conditions["and"]
    else:
        # Treat the whole dict as a single implicit-AND condition list
        sub_conditions = [{k: v} for k, v in conditions.items()]

    def predicate(vector_id: int) -> bool:
        payload = payloads[vector_id]
        return all(_check_condition(payload, cond) for cond in sub_conditions)

    return predicate


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

def estimate_selectivity(conditions: dict, payloads: list[dict], sample: int = 1000) -> float:
    """Estimate the fraction of vectors passing the predicate (Monte-Carlo sample)."""
    n = len(payloads)
    indices = np.random.choice(n, size=min(sample, n), replace=False)
    pred = build_predicate(conditions, payloads)
    hits = sum(1 for i in indices if pred(int(i)))
    return hits / len(indices)


# --- search strategies ---

def search_post_filter(
    index,
    query: np.ndarray,
    k: int,
    ef_search: int,
    allowed: np.ndarray,
) -> list[int]:
    """
    Standard HNSW search followed by predicate filtering.

    Fetches `ef_search` candidates (the maximum the HNSW is willing to explore),
    then keeps only those satisfying the predicate. This is the naive baseline:
    cheap to implement, but recall collapses when the predicate is selective.
    """
    _distances, ids = index.search(query, ef_search, ef_search=ef_search)
    passing = [int(i) for i in ids if i >= 0 and allowed[i]]
    return passing[:k]


def search_acorn1(
    index,
    query: np.ndarray,
    k: int,
    ef_search: int,
    allowed: np.ndarray,
) -> list[int]:
    """ACORN-1 filtered search (two-hop expansion during graph traversal)."""
    _distances, ids = index.search(query, k, ef_search=ef_search, filter=allowed, acorn_gamma=False)
    return [int(i) for i in ids if i >= 0]


def search_acorn_gamma(
    index,
    query: np.ndarray,
    k: int,
    ef_search: int,
    allowed: np.ndarray,
) -> list[int]:
    """ACORN-γ filtered search (pre-expanded neighbor lists, no two-hop at query time)."""
    _distances, ids = index.search(query, k, ef_search=ef_search, filter=allowed, acorn_gamma=True)
    return [int(i) for i in ids if i >= 0]


def run_all_benchmarks(
    index,
    payloads: list[dict],
    tests: list[dict],
    k: int,
    ef_search: int,
    n_queries: int,
    selectivities: list[float],
) -> list[dict]:
    """
    Run post-filter, ACORN-1, and ACORN-γ on the same queries and return per-method stats.

    Methods:
      - Post-filter  : HNSW search then discard non-matching results
      - ACORN-1      : two-hop predicate-aware graph traversal
      - ACORN-γ      : pre-expanded neighbor lists, no two-hop at query time

    `selectivities` is pre-computed once by the caller so it is shared across
    multiple ef_search sweeps.
    """
    n_queries = min(n_queries, len(tests))

    methods = [
        (
            f"Post-filter  (ef={ef_search})",
            lambda q, pred: search_post_filter(index, q, k, ef_search, pred),
        ),
        (
            f"ACORN-1      (ef={ef_search})",
            lambda q, pred: search_acorn1(index, q, k, ef_search, pred),
        ),
        (
            f"ACORN-gamma  (ef={ef_search})",
            lambda q, pred: search_acorn_gamma(index, q, k, ef_search, pred),
        ),
    ]

    # Run queries method-by-method so progress is easy to follow.
    results = []
    for label, search_fn in methods:
        print(f"\n── {label} ──")

        recalls = []
        latencies_ms = []

        for i, test in enumerate(tests[:n_queries]):
            query = np.array(test["query"], dtype=np.float32)
            ground_truth = set(test["closest_ids"][:k])
            # Evaluate the predicate once, outside the timed region, as a mask over the collection.
            # The search reads the mask.
            predicate = build_predicate(test["conditions"], payloads)
            allowed = np.fromiter((predicate(i) for i in range(len(payloads))), dtype=bool, count=len(payloads))

            t0 = time.perf_counter()
            result_ids = search_fn(query, allowed)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

            result_set = set(result_ids)
            recall = len(result_set & ground_truth) / len(ground_truth) if ground_truth else 1.0
            recalls.append(recall)

            if (i + 1) % 10 == 0 or (i + 1) == n_queries:
                print(
                    f"  [{i+1:3d}/{n_queries}]  recall={recall:.2f}  "
                    f"sel={selectivities[i]:.2%}  lat={elapsed_ms:.1f}ms",
                    flush=True,
                )

        results.append({
            "label": label,
            "mean_recall": float(np.mean(recalls)),
            "median_recall": float(np.median(recalls)),
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "mean_selectivity": float(np.mean(selectivities)),
        })

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ACORN-1 filtered ANN search with kannolo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="laion-small-clip",
        help="Dataset to download and benchmark.",
    )
    parser.add_argument("--data-dir", default="./datasets", help="Directory to store downloaded data.")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbours to retrieve.")
    parser.add_argument(
        "--ef-search", type=int, nargs="+", default=[200],
        metavar="EF",
        help="One or more ef_search values to sweep (e.g. --ef-search 20 50 100 200).",
    )
    parser.add_argument("--m", type=int, default=16, help="HNSW M parameter.")
    parser.add_argument("--ef-construction", type=int, default=200, help="ef_construction parameter.")
    parser.add_argument(
        "--gamma", type=int, nargs="+", default=[2],
        metavar="G",
        help="One or more ACORN-γ expansion factors to sweep (e.g. --gamma 2 4 8).",
    )
    parser.add_argument("--n-queries", type=int, default=50, help="Number of test queries to run.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Import kannolo (give a clear error if not installed)
    # ------------------------------------------------------------------
    try:
        import kannolo
    except ImportError:
        print(
            "ERROR: kannolo is not installed.\n"
            "Build and install it with:  maturin develop --release",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Download / load dataset
    # ------------------------------------------------------------------
    ds_info = DATASETS[args.dataset]
    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset}")
    print(f"          {ds_info['description']}")
    print(f"{'='*60}\n")

    dataset_dir = download_and_extract(ds_info["url"], Path(args.data_dir))
    vectors, payloads, tests = load_dataset(dataset_dir)

    n, dim = vectors.shape
    print(f"\nVectors : {n:,} × {dim}D  |  Tests: {len(tests):,}\n")

    # Show a sample payload and test so the user knows the data format
    print("── Sample payload ──────────────────────────────")
    print(json.dumps(payloads[0], indent=2))
    print("\n── Sample test ─────────────────────────────────")
    sample_test = {k: v for k, v in tests[0].items() if k != "query"}
    sample_test["query"] = f"[{dim}D vector]"
    print(json.dumps(sample_test, indent=2, default=str))
    print()

    # ------------------------------------------------------------------
    # Build HNSW index
    # ------------------------------------------------------------------
    print(f"Building HNSW (M={args.m}, ef_construction={args.ef_construction}) ...")
    t0 = time.perf_counter()
    index = kannolo.build(
        vectors.astype(np.float32),
        metric=ds_info["metric"],
        encoder=kannolo.Plain("f16"),
        index=kannolo.HNSW(m=args.m, ef_construction=args.ef_construction),
    )
    build_time = time.perf_counter() - t0
    print(f"Index built in {build_time:.1f}s\n")

    # ------------------------------------------------------------------
    # Pre-compute selectivities once (shared across all sweeps)
    # ------------------------------------------------------------------
    n_queries = min(args.n_queries, len(tests))
    print(f"Pre-computing selectivities for {n_queries} queries ...")
    selectivities = []
    for test in tests[:n_queries]:
        sel = estimate_selectivity(test["conditions"], payloads, sample=500)
        selectivities.append(sel)
    print(f"  Mean selectivity: {np.mean(selectivities):.2%}\n")

    # ------------------------------------------------------------------
    # Sweep over gamma × ef_search combinations
    # Each gamma requires rebuilding the expanded neighbor lists once;
    # all ef_search values for that gamma reuse the same expansion.
    # ------------------------------------------------------------------
    gamma_values = sorted(set(args.gamma))
    ef_values = sorted(set(args.ef_search))

    # all_tables[(gamma, ef)] = list of per-method stats dicts
    all_tables: dict[tuple[int, int], list[dict]] = {}

    for gamma in gamma_values:
        print(f"\n{'═'*60}")
        print(f"  Building ACORN-γ expanded neighbors (gamma={gamma}) ...")
        t0 = time.perf_counter()
        index.build_acorn_gamma(gamma)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")
        print(f"{'═'*60}")

        for ef in ef_values:
            print(f"\n{'─'*60}")
            print(f"  gamma={gamma}  ef_search={ef}")
            print(f"{'─'*60}")
            all_tables[(gamma, ef)] = run_all_benchmarks(
                index, payloads, tests,
                k=args.k,
                ef_search=ef,
                n_queries=n_queries,
                selectivities=selectivities,
            )

    # ------------------------------------------------------------------
    # Print consolidated summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*88}")
    print(f"Summary  —  k={args.k}  selectivity≈{np.mean(selectivities):.1%}  "
          f"({n_queries} queries)")
    print(f"{'='*88}")
    header = f"{'γ':>4}  {'ef':>6}  {'Method':<26}  {'Recall@k':>9}  {'Mean lat':>9}  {'p95 lat':>8}"
    print(header)
    print("-" * 88)
    for gamma in gamma_values:
        for ef in ef_values:
            stats = all_tables[(gamma, ef)]
            for i, s in enumerate(stats):
                gamma_label = str(gamma) if i == 0 and ef == ef_values[0] else ""
                ef_label = str(ef) if i == 0 else ""
                row = (
                    f"{gamma_label:>4}  "
                    f"{ef_label:>6}  "
                    f"{s['label']:<26}  "
                    f"{s['mean_recall']:>9.4f}  "
                    f"{s['mean_latency_ms']:>8.1f}ms  "
                    f"{s['p95_latency_ms']:>7.1f}ms"
                )
                print(row)
            print()
    print(f"{'='*88}\n")


if __name__ == "__main__":
    main()
