# Kannolo Python API Reference

Kannolo provides a Python interface for vector search with support for:
- **Dense indexes**: Plain and PQ-encoded HNSW
- **Sparse indexes**: Multiple encoding schemes (plain, DotPacking8, FixedU8, FixedU16)
- **Flat (brute-force) indexes**: Dense and sparse exhaustive search
- **Multivector reranking**: Two-stage sparse + multivector retrieval

## Imports

```python
from kannolo import (
    # Dense HNSW
    DensePlainHNSW,      # Dense plain HNSW (no quantization)
    DensePQHNSW,         # Dense product-quantized HNSW
    DenseFlatIndex,      # Dense exhaustive (brute-force) search
    
    # Sparse HNSW
    SparsePlainHNSW,     # Sparse HNSW, dot product, blocked sparse encoder (f16 values)
    SparseDotPacking8HNSW,  # Sparse DotPacking8 HNSW (u16 components, fixed-point u8 values)
    SparseFixedU8HNSW,   # Sparse fixed u8-encoded HNSW
    SparseFixedU16HNSW,  # Sparse fixed u16-encoded HNSW
    SparseFlatIndex,     # Sparse exhaustive (brute-force) search
    
    # Multivector reranking
    SparseMultivecRerankIndex,            # Sparse HNSW + Plain multivector rerank
    SparseMultivecTwoLevelsPQRerankIndex, # Sparse HNSW + Two-level PQ multivector rerank

    # Dense reranking
    DenseRerankHNSW,     # Compressed dense HNSW + f16 or RaBitQ-ext rerank

    # Sparse reranking
    SparseRerankHNSW,    # Sparse HNSW over pruned/compressed vectors + rerank with residuals/full vectors
)
import numpy as np
```

## Index Construction

### HNSW Parameters

All HNSW indexes support:
- `m` (int): Neighbors per node. Typical: 16–64. Default: 32
- `ef_construction` (int): Graph construction effort. Higher = better quality, slower. Default: 200
- `metric` (str): Distance metric. Options: `"euclidean"`, `"dotproduct"` (default)
- `graph_type` (str): How the graph is stored. Leave it out and you get `"standard"`.
  See [GraphCompression.md](GraphCompression.md)
  - `"standard"` (default) — original node order, uncompressed
  - `"compressed"` — reordered *and* compressed: roughly halves the graph portion of the
    index and keeps the query speed-up below. Requires `m <= 128`
  - `"permuted"` — reordered only: faster queries, because nodes that are searched together
    end up stored together, but no size saving (marginally larger than `"standard"`)

  All three return identical search results; they differ in index size and query speed.
  Whatever you pass to the builder must be passed again to `load()`: index files are not
  self-describing, so `load()` cannot detect it for you.

### Dense Plain HNSW

```python
# From .npy file (dtype=float32)
index = DensePlainHNSW.build_from_file(
    "data.npy",
    m=32,
    ef_construction=200,
    metric="dotproduct",
    graph_type="standard"   # or "compressed" / "permuted"
)

# From numpy array. The array must be flattened, and `dim` is required:
# vectors are passed as one contiguous 1-D buffer of n_vectors * dim floats.
data = np.random.randn(10000, 768).astype(np.float32)
index = DensePlainHNSW.build_from_array(data.flatten(), dim=768, m=32, ef_construction=200)
```

### Dense PQ HNSW

```python
# Product-quantized dense index
index = DensePQHNSW.build_from_file(
    "data.npy",
    m_pq=32,           # PQ subspaces: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384
    m=32,              # HNSW neighbors
    ef_construction=200,
    metric="dotproduct"
)
```

`m_pq` must divide the vector dimensionality. Note that the CLI's `--pq-subspaces` accepts a
different set (`4` … `192`); `256` and `384` are Python-only, and `4` is CLI-only.

### Dense Flat Index

```python
# Exhaustive search (no HNSW index, just linear scan)
index = DenseFlatIndex.build_from_file("data.npy", metric="dotproduct")

# Or from array (flattened, as above)
data = np.random.randn(10000, 768).astype(np.float32)
index = DenseFlatIndex.build_from_array(data.flatten(), dim=768, metric="dotproduct")
```

### Sparse Plain HNSW

Dot product only. The collection is stored with the blocked sparse encoder (`f16` values); the
graph is built over the plain `f16` collection, which is then re-encoded. There is no `metric`
argument.

```python
# From binary file (seismic format)
index = SparsePlainHNSW.build_from_file(
    "data.bin",
    m=32,
    ef_construction=200,
)

# From numpy arrays (components and values)
components = np.array([0, 5, 10, 15], dtype=np.int32)
values = np.array([0.5, 0.3, 0.8, 0.2], dtype=np.float32)
offsets = np.array([0, 4], dtype=np.int64)  # one document starting at 0, ending at 4

# Build with these vectors
index = SparsePlainHNSW.build_from_arrays(
    components, values, offsets,
    m=32,
    ef_construction=200,
)
```

### Sparse Variants (DotPacking8, FixedU8, FixedU16)

```python
# Same interface as SparsePlainHNSW, plus `metric`
# SparseDotPacking8HNSW replaces SparseDotVByteHNSW (removed with vectorium's DotVByte encoder);
# it supports only dotproduct and u16 components. Old DotVByte index files must be rebuilt.
index = SparseDotPacking8HNSW.build_from_file("data.bin", m=32, ef_construction=200)
index = SparseFixedU8HNSW.build_from_file("data.bin", m=32, ef_construction=200, metric="dotproduct")
index = SparseFixedU16HNSW.build_from_file("data.bin", m=32, ef_construction=200, metric="dotproduct")
```

### Sparse Flat Index

```python
# Exhaustive sparse search
index = SparseFlatIndex.build_from_file("data.bin")

# Or from arrays
components = np.array([0, 5, 10], dtype=np.int32)
values = np.array([0.5, 0.3, 0.8], dtype=np.float32)
offsets = np.array([0, 3], dtype=np.int64)
index = SparseFlatIndex.build_from_arrays(components, values, offsets)
```

### Dense Reranking

A two-stage dense index: a compressed HNSW first stage over PQ / RaBitQ / RaBitQ-ext codes, reranked against the same collection at plain `f16` (the default) or as RaBitQ-ext codes (`rerank_bits`).

The graph is always built over `f16` and only its *dataset* is then replaced by codes, so the compressed representation never takes part in construction. With an `f16` rerank stage the index holds the collection twice, which trades memory for speed, unlike `DensePQHNSW`; RaBitQ-ext rerank codes make the second copy smaller.

```python
data = np.random.randn(10000, 768).astype(np.float32)

# PQ first stage
index = DenseRerankHNSW.build_from_array(
    data.flatten(), dim=768,
    m=32, ef_construction=200, metric="dotproduct",
    encoder="pq", pq_subspaces=32,
)

# RaBitQ-ext first stage: `rabitq_total_bits` is the document code width (2, 4 or 8)
index = DenseRerankHNSW.build_from_array(
    data.flatten(), dim=768,
    encoder="rabitq-ext", rabitq_total_bits=4,
)

# RaBitQ first stage: `rabitq_query_bits` (1..=8) sets the query-side width
index = DenseRerankHNSW.build_from_array(
    data.flatten(), dim=768,
    encoder="rabitq", rabitq_query_bits=1,
)
```

- `encoder` (str): `"pq"`, `"rabitq"` or `"rabitq-ext"`. Default: `"pq"`
- `pq_subspaces` (int): applies to `"pq"`. Default: 32
- `rabitq_total_bits` (int): document code width for `"rabitq-ext"`. Default: 4
- `rabitq_query_bits` (int): query code width for `"rabitq"`. Default: 1
- `rerank_bits` (int): the rerank stage's encoding. `0` is plain `f16`; `2`, `4` or `8` store
  RaBitQ-ext codes instead, which shrinks the rerank dataset. Default: 0
- `residual_rerank` (bool): encode `x - x̂` (what the first stage left out) in the rerank stage
  and score a candidate as first-stage plus rerank score. Needs `rerank_bits > 0`, a `"rabitq"`
  or `"rabitq-ext"` first stage, and `metric="dotproduct"`. Default: False
- `m`, `ef_construction`, `metric`, `graph_type`: as for every HNSW index above

The graph, the first-stage codes and the candidate lists do not depend on `rerank_bits`: only the
rescoring changes.

```python
# RaBitQ-ext 4-bit first stage, reranked on 8-bit RaBitQ-ext codes instead of f16
index = DenseRerankHNSW.build_from_array(
    data.flatten(), dim=768,
    encoder="rabitq-ext", rabitq_total_bits=4, rerank_bits=8,
)
```

Or straight from a `.npy` collection, where `dim` comes from the file's shape:

```python
index = DenseRerankHNSW.build_from_file(
    "data.npy",
    m=32, ef_construction=200, metric="dotproduct",
    encoder="pq", pq_subspaces=32,
)
```

### Sparse Reranking

A two-stage sparse index (dot product). Both stages are built from one collection: an HNSW
first stage over a reduced version of each vector, and a rerank stage that rescores its
candidates. Three build options decide what each stage holds:

- `prune_percentage` (float, in `[0, 100]`, required): the first stage keeps, per vector, the
  components carrying the top `prune_percentage`% of its L1 mass. 100 means no pruning
- `first_stage_value_type` (str): `"blocked-sparse"`, `"f16"` or `"fixedu8"`. With
  `"blocked-sparse"` both stages are stored with the blocked sparse encoder (`f16` values) and the
  graph is built over the `f16` values; otherwise the rerank stage is `f16`. `graph_type="compressed"`
  needs `"f16"` or `"fixedu8"`. Default: `"blocked-sparse"`
- `residual_rerank` (bool): store only the pruned-away components in the rerank stage and score a
  candidate as first-stage plus rerank score. Needs `prune_percentage < 100`. Default: False

| `prune_percentage` | `residual_rerank` | first stage | index |
|---|---|---|---|
| < 100 | True | `blocked-sparse` or `f16` | L1-pruned first stage, exact rescoring at `f16` precision; each non-zero stored once |
| < 100 | False | any | pruned first stage, rescored on the full vectors |
| < 100 | True | `fixedu8` | pruned and quantized first stage; the summed score is approximate |
| 100 | False | `fixedu8` | quantized first stage, rescored on the full `f16` vectors |

`prune_percentage=100` is rejected with `residual_rerank=True` (the residual would be empty) and
with a `blocked-sparse` or `f16` first stage (the rerank stage would be an identical copy).

```python
from kannolo import SparseRerankHNSW

index = SparseRerankHNSW.build_from_file(
    "data.bin",                  # Seismic binary format
    prune_percentage=60.0, residual_rerank=True,
    m=32, ef_construction=200,
)

distances, ids = index.search(
    query_components, query_values,
    k_candidates=100, k=10, ef_search=100,
)
```

`build_from_arrays(components, values, offsets, ...)` takes the collection as flat arrays, with
the same options. `batch_search` takes the same arguments plus `num_threads`, with queries in the
flattened sparse format described under [Search Operations](#search-operations).

### Multivector Reranking

#### Plain Multivector

```python
# Expects multivec_data_folder with (see MultiVectorUsage.md for the full spec):
# - documents.npy (shape: [n_tokens, token_dim], dtype: uint16, reinterpreted as f16)
#   Note: 2-D and token-major. Documents are delimited by doclens, not by a dimension.
# - doclens.npy (shape: [n_docs], dtype: int32)

index = SparseMultivecRerankIndex.build_from_file(
    sparse_index_path="sparse_index_file",
    multivec_data_folder="/path/to/multivec_data_folder"
)
```

#### Two-Level PQ Multivector

```python
# Expects doclens.npy as above, and instead of documents.npy:
# - centroids.npy       (shape: [n_coarse_centroids, token_dim], dtype: float32)
# - pq_centroids.npy    (shape: [M * 256 * dsub], dtype: float32, dsub = token_dim / M)
# - residuals.npy       (shape: [n_tokens, M], dtype: uint8 -- PQ codes)
# - index_assignment.npy (shape: [n_tokens], dtype: uint64)
# - residual_norms.npy  (shape: [n_tokens], dtype: float32) -- optional, picked up if present

index = SparseMultivecTwoLevelsPQRerankIndex.build_from_file(
    sparse_index_path="sparse_index.bin",
    multivec_data_folder="/path/to/multivec_data",
    pq_subspaces=32  # Must be 8, 16, 32, or 64
)
```

---

## Save / Load

```python
# Save any index
index.save("my_index.bin")

# Load (must specify metric for types that support it)
index = DensePlainHNSW.load("my_index.bin", metric="dotproduct")
index = SparsePlainHNSW.load("my_index.bin")
index = DensePQHNSW.load("my_index.bin", m_pq=32, metric="dotproduct")

# HNSW indexes built with a non-default graph type must be loaded with the same one
index = DensePlainHNSW.load("my_index.bin", metric="dotproduct", graph_type="compressed")

# Flat indexes
index = DenseFlatIndex.load("my_index.bin", metric="dotproduct")
index = SparseFlatIndex.load("my_index.bin")  # dot product only, nothing to disambiguate

# Two-stage indexes save both stages to the one file
index = DenseRerankHNSW.load(
    "my_index.bin",
    metric="dotproduct", graph_type="standard",
    encoder="pq", pq_subspaces=32,
)
index = SparseRerankHNSW.load(
    "my_index.bin", first_stage_value_type="f16", graph_type="standard", residual_rerank=True,
)
index = SparseMultivecRerankIndex.load("my_index.bin")
index = SparseMultivecTwoLevelsPQRerankIndex.load("my_index.bin", pq_subspaces=32)
```

Every index type supports `save` / `load`.

Index files carry no header or type tag, so whatever you passed at build time must be passed
again to `load`: a mismatch decodes into garbage or fails outright rather than reporting itself.

For `DenseRerankHNSW` two arguments behave differently from the rest:

- `rabitq_total_bits` is **not** accepted by `load` — it is stored with the encoder and comes
  back with the index.
- `rabitq_query_bits` **is** accepted, and is free to differ from the build-time value: it
  describes how a query is processed and touches nothing that was stored. One saved index
  therefore serves every query width, and `set_query_bits` can change it again afterwards
  without saving anything.
- `rerank_bits` must be `0` for an index built with the `f16` rerank stage and `2`, `4` or `8`
  otherwise (the exact width is stored), and `residual_rerank` must match the build.

For the two-stage classes, `save`/`load` round-trips the index as it stands, whereas
`build_from_file` reconstructs it from a first-stage index plus the original dataset — so
`load` avoids re-reading and re-encoding the rerank data.

---

## Search Operations

Every index exposes **two** methods, and they are not interchangeable:

- **`search(...)`** — one query. Dense indexes take a 1-D array of exactly `dim` floats and
  raise `ValueError` otherwise; sparse indexes take one `(components, values)` pair and take
  **no** `offsets` argument.
- **`batch_search(...)`** — many queries at once, optionally in parallel. Dense indexes take
  all queries concatenated into one flat array; sparse indexes take the concatenated
  `components`/`values` plus an `offsets` array delimiting each query.

Both return `(distances, ids)` as two 1-D numpy arrays. **Results are always padded to `k`**:
if fewer than `k` neighbors are found, the remainder is filled with `inf` distances and `-1`
ids. `batch_search` returns `num_queries * k` entries, query-major.

Returned ids are always ids into your original dataset, whatever `graph_type` you built with.

### `num_threads` (batch only)

- `0` (default) — rayon's default pool, typically every core.
- `1` — a plain serial loop with no rayon involvement. Use this to reproduce single-threaded
  benchmarks that pin the process with `numactl --physcpubind`.
- `n` — a temporary pool of `n` threads for the duration of the call.

### Dense indexes (`DensePlainHNSW`, `DensePQHNSW`)

```python
# Single query: a 1-D array of exactly `dim` floats.
query = np.random.randn(768).astype(np.float32)
dists, ids = index.search(query, k=10, ef_search=200)

# With early exit threshold (optional, suggested 0.005-0.25)
dists, ids = index.search(query, k=10, ef_search=200, early_exit_threshold=0.1)

# Many queries: flatten them into a single contiguous buffer.
queries = np.random.randn(100, 768).astype(np.float32)
dists, ids = index.batch_search(queries.flatten(), k=10, ef_search=200, num_threads=0)
# dists and ids each hold 100 * 10 entries, query-major.
```

`DensePQHNSW.search` has no default arguments: pass `ef_search` and `early_exit_threshold`
explicitly.

### Dense Flat Index

```python
query = np.random.randn(768).astype(np.float32)
dists, ids = index.search(query, k=10)

queries = np.random.randn(100, 768).astype(np.float32)
dists, ids = index.batch_search(queries.flatten(), k=10, num_threads=0)
```

### Sparse HNSW (`SparsePlainHNSW`, `SparseDotPacking8HNSW`, `SparseFixedU8HNSW`, `SparseFixedU16HNSW`)

```python
# Single query: components and values only, no offsets.
query_components = np.array([0, 5, 10], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3], dtype=np.float32)
dists, ids = index.search(query_components, query_values, k=10, ef_search=200)

# Many queries: concatenate them and delimit with offsets.
query_components = np.array([0, 5, 10, 100, 200], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
offsets = np.array([0, 3, 5], dtype=np.int64)  # Two queries: [0,3) and [3,5)

dists, ids = index.batch_search(
    query_components, query_values, offsets,
    k=10, ef_search=200,
    early_exit_threshold=0.1,   # optional
    num_threads=0,
)
```

### Sparse Flat Index

```python
query_components = np.array([0, 5, 10], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3], dtype=np.float32)
dists, ids = index.search(query_components, query_values, k=10)

# Batch form takes offsets.
offsets = np.array([0, 3], dtype=np.int64)
dists, ids = index.batch_search(query_components, query_values, offsets, k=10, num_threads=0)
```

### Dense Reranking (`DenseRerankHNSW`)

```python
query = np.random.randn(768).astype(np.float32)

dists, ids = index.search(
    query,
    k=10,                       # Final results
    k_candidates=100,           # First-stage candidates passed to reranking (frontier knob)
    ef_search=100,              # First-stage candidate list size
    alpha=0.3,                  # Candidate pruning (optional)
    beta=None,                  # Rerank early exit (optional)
    early_exit_threshold=None,  # First-stage distance-adaptive early exit (optional)
)
```

`k_candidates` is the knob to sweep: it sets how many first-stage candidates get reranked
against the rerank stage. `ef_search` is usually a multiple of it. `alpha` keeps only
candidates whose first-stage score is within a relative slack of the k-th best.

#### Sweeping the RaBitQ query width

Under `encoder="rabitq"` the query-side bit width can be changed on a live index:

```python
index = DenseRerankHNSW.build_from_array(data.flatten(), dim=768, encoder="rabitq")

for bits in (1, 2, 4, 8):
    index.set_query_bits(bits)
    dists, ids = index.search(query, k=10, k_candidates=100, ef_search=100)
```

The width only affects how a *query* is processed and leaves the stored codes at the width they
were encoded with, so one index serves every width — a width ladder is a set of search
arguments, not a set of separate builds of byte-identical indexes. Results are identical to
those from an index built at that width.

It raises `ValueError` for a width outside `1..=8`, and for `encoder="pq"` or
`encoder="rabitq-ext"`, whose queries are not quantized at all, rather than silently doing
nothing.

The width is **not** part of the saved index — see Save / Load above — so a reloaded index
starts at whatever `rabitq_query_bits` was passed to `load`, regardless of what was set when it
was saved.

The batch form takes the queries flattened into one contiguous buffer and returns query-major
results:

```python
queries = np.random.randn(100, 768).astype(np.float32)

dists, ids = index.batch_search(
    queries.flatten(),
    k=10, k_candidates=100, ef_search=100,
    num_threads=0,   # 0 = every core, 1 = serial, n = that many
)
# dists and ids each hold 100 * 10 entries
```

### Sparse Multivector Reranking

The single-query form takes `multivec_query` (singular) and no offsets:

```python
query_components = np.array([0, 5, 10, 100], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3, 0.7], dtype=np.float32)

# One query: 8 tokens of 768 dimensions, flattened.
multivec_query = np.random.randn(8, 768).astype(np.float32).reshape(-1)

dists, ids = index.search(
    query_components, query_values, multivec_query,
    n_tokens=8,
    token_dim=768,
    k_candidates=25,   # First-stage candidates
    k=10,              # Final results
    ef_search=100,
    alpha=0.05,        # First-stage weight (optional)
    beta=2,            # Second-stage early exit (optional)
    residuals=False,   # Sum first- and second-stage scores (optional)
)
```

`residuals=True` scores each candidate as `first_stage_score + rerank_score` instead of the rerank
score alone, for setups where the rerank dataset holds the residual part of a decomposed
representation. Nothing checks that the two scores are summable — that is on the caller.

The batch form adds `sparse_offsets` and takes `multivec_queries` (plural):

```python
sparse_offsets = np.array([0, 4], dtype=np.int64)  # One query
multivec_queries = np.random.randn(1, 8, 768).astype(np.float32).reshape(-1)

dists, ids = index.batch_search(
    query_components, query_values, sparse_offsets, multivec_queries,
    n_tokens=8, token_dim=768,
    k_candidates=25, k=10, ef_search=100,
    num_threads=0,
    residuals=False,   # Sum first- and second-stage scores (optional)
)
```

---

## Index Size

`DensePlainHNSW` and `DenseRerankHNSW` report the bytes they hold:

```python
index.space_usage_bytes()
```

This is the library's own accounting over the dataset and every graph level, not a
resident-set measurement — pair it with process RSS/USS if you need both.

## Filtered Search (ACORN)

Predicate-aware search — "the `k` nearest neighbors for which `predicate(id)` is true" — is
available on **`DensePlainHNSW` only**. It is not a post-filter: the predicate steers the
graph traversal, so recall stays usable even when few vectors qualify.

The predicate is a Python callable receiving an original dataset id and returning `bool`.
It is invoked during traversal, so keep it cheap.

```python
# ACORN-1: two-hop expansion computed on the fly. Nothing to prepare.
dists, ids = index.search_filtered(query, k=10, predicate=lambda i: i % 2 == 0, ef_search=100)

# ACORN-gamma: precompute the expansion once, then search repeatedly.
# Trades memory for speed; gamma of 2-4 is a sensible starting point.
index.build_acorn_gamma(gamma=3)
dists, ids = index.search_filtered_gamma(query, k=10, predicate=lambda i: i % 2 == 0, ef_search=100)
```

`build_acorn_gamma` must be called before `search_filtered_gamma`, which otherwise raises
`RuntimeError`. The expanded lists live on the index object in memory and are **not** written
by `save()`: call `build_acorn_gamma` again after `load()`.

---

## Index Selection Guide

| Use Case | Index | Notes |
|----------|-------|-------|
| Dense vectors, high accuracy | `DensePlainHNSW` | Default choice |
| Dense vectors, memory limited | `DensePQHNSW` | Quantized, faster search |
| Dense vectors, ground truth/exhaustive search | `DenseFlatIndex` | Exhaustive, exact neighbors |
| Sparse vectors, standard | `SparsePlainHNSW` | Blocked sparse encoding (`f16` values), dot product |
| Sparse vectors, memory limited | `SparseFixedU8HNSW` or `SparseDotPacking8HNSW` | Compressed |
| Sparse vectors, ground truth/exhaustive search | `SparseFlatIndex` | Exhaustive, exact |
| Dense vectors, compressed first stage + rerank | `DenseRerankHNSW` | Compressed HNSW + `f16` or RaBitQ-ext rerank |
| Sparse vectors, pruned or quantized first stage + rerank | `SparseRerankHNSW` | L1-pruned HNSW + residual or full-vector rerank |
| Multivector retrieval | `SparseMultivecRerankIndex` | Sparse first-stage + multivec rerank |
| Multivector + quantization | `SparseMultivecTwoLevelsPQRerankIndex` | Sparse + PQ rerank |

---

## Additional Resources

- **Notebooks**: See [notebooks/](../notebooks/) for end-to-end examples
- **Rust API**: See [RustUsage.md](RustUsage.md)
- **Running Experiments**: See [RunExperiments.md](RunExperiments.md)
