# Kannolo Python API Reference

`kannolo.build` builds an index from independent choices, one per axis: the representation
(dense, sparse or multi-vector), the metric, the encoder, the index structure (Flat, HNSW or
IVF) and optional rescoring stages. Every composition in Table 7 of the kANNolo paper is one
`kannolo.build` call. `kannolo.load` reads any saved index back.

## Composable API

Choose the representation, metric, encoder, index structure and rescoring stages
independently. The representation (dense, sparse or multi-vector) is inferred from the data.

```python
import numpy as np
import kannolo

data = np.random.randn(10000, 768).astype(np.float32)        # (n, dim)

index = kannolo.build(
    data,
    metric="ip",                                  # default; "l2", or "maxsim" (multi-vector)
    encoder=kannolo.PQ(subspaces=64),             # default: kannolo.Plain()
    index=kannolo.HNSW(graph="compressed"),       # m=32, ef_construction=200 by default; or Flat(), IVF(...)
    rerank=kannolo.Rerank(kannolo.Plain("f16")),  # optional rescoring stage(s)
)

queries = np.random.randn(100, 768).astype(np.float32)
distances, ids = index.search(queries, k=10, ef_search=100, k_candidates=200)   # (100, 10)
distances, ids = index.search(queries[0], k=10)                                 # one query: (10,)

index.save("my.index")
index = kannolo.load("my.index")   # no arguments: the file records how it was built
print(index)                  # kannolo.Index(n=10000, dim=768, Dense(), metric='ip', ...)
```

### Data

| Representation | Build data | Queries (batch) | One query |
|---|---|---|---|
| dense | 2-D float array `(n, dim)`, or a `.npy` path | 2-D array | 1-D array |
| sparse | `scipy.sparse` matrix, `(components, values, offsets)`, or a `.bin` (seismic) path | same | `(components, values)` |
| multi-vector | list of `(n_tokens, dim)` arrays, or `(tokens, lengths)` | same | one `(n_tokens, dim)` array |

Sparse components need not be sorted within a vector. Repeated components are rejected. Query
components that no indexed vector has are ignored (they cannot change an inner product).

A sparse `.bin` file uses the seismic binary format, little-endian. It starts with the number of
vectors as a `u32`. Then, for each vector, it stores the number of nonzeros `n` as a `u32`, the
`n` components as `u32` (strictly ascending) and the `n` values as `f32`.
`scripts/convert_npy_arrays_to_bin.py` and `scripts/convert_bin_to_npy_arrays.py` convert between
this format and `.npy` arrays.

### Options per axis

| Axis | Options |
|---|---|
| `representation` | inferred; or `kannolo.Dense()`, `kannolo.Sparse(components="auto" \| "u16" \| "u32")`, `kannolo.MultiVector()` |
| `metric` | `"ip"` (alias `"dotproduct"`), `"l2"` (squared Euclidean, alias `"euclidean"`), `"maxsim"` (multi-vector). Default: `"ip"`, or `"maxsim"` for multi-vector data |
| `encoder` | see the table below |
| `index` | `kannolo.Flat()`, `kannolo.HNSW(m=32, ef_construction=200, graph="standard" \| "permuted" \| "compressed")`, `kannolo.IVF(n_lists=1024, residual=False, centroids=kannolo.Flat() \| kannolo.HNSW(...), ...)` |
| `transform` | `kannolo.L1Prune(percent)`: the first stage indexes the largest components holding `percent`% of each sparse vector's L1 mass |
| `rerank` | `kannolo.Rerank(encoder, data=None, metric=None, residual=False)`, or a list of them (a cascade) |

| Encoder | Dense | Sparse | Multi-vector | Notes |
|---|---|---|---|---|
| `kannolo.Plain(dtype="f32" \| "f16" \| "bf16")` | ✓ | ✓ (f32, f16) | ✓ (f32, f16) | uncompressed (a cast for f16/bf16) |
| `kannolo.FixedPoint(bits=8 \| 16)` | ✓ | ✓ | | fixed-point values |
| `kannolo.Scalar(bits=8 \| 16)` | | ✓ | | one trained step per component |
| `kannolo.PQ(subspaces)` | ✓ | | ✓ | dense: 8, 16, 32, 64, 128 subspaces (for other code sizes, use RaBitQ); multi-vector: 8, 16, 32, 64 per token; must divide the dimension |
| `kannolo.RaBitQ(bits=1, seed=42, rotate=True)` | ✓ | | | bits ∈ {1, 2, 4, 8}; dimension a multiple of 64 |
| `kannolo.Blocked(values="f16" \| "scalar_u8" \| "scalar_u16")` | | ✓ | | inner product, `u16` components |
| `kannolo.DotPacking8(values=...)` | | ✓ | | inner product; `fixed_u8`, `scalar_u8`, `scalar_u16`, `f16`, `scalar_4bit` |
| `kannolo.TwoLevelPQ(subspaces=16, n_coarse=1024)` | | | ✓ | coarse centroid + PQ residual per token |

IVF stores dense clusters as `Plain("f32"/"f16")`, `PQ` or `RaBitQ`, optionally as residuals
against their centroid. It stores sparse clusters as `Plain("f32"/"f16")` (inner product, `u16`).
IVF is not available for multi-vector data. Its k-means is configured with `kmeans_epochs`,
`kmeans_sample_size`, `seed`, `spherical` and `n_redo`. `centroid_nnz_factor` prunes sparse
centroids.

`kannolo.check(...)` validates a combination without building anything. A combination that does
not exist raises a `ValueError` that says why:

```python
kannolo.check(representation=kannolo.Sparse(), metric="l2", encoder=kannolo.DotPacking8())
# ValueError: DotPacking8 scores with inner product only; use metric="ip"
```

### Rescoring stages

A rescoring stage rescores the candidates of the stage before it. It has its own encoding and
metric, and optionally its own representation and data. The data must hold the same items in the
same order as the first stage.

```python
# Dense: 1-bit RaBitQ graph, rescored on f16 (the two-stage index of the paper's Figure 2).
index = kannolo.build(x, encoder=kannolo.RaBitQ(1), index=kannolo.HNSW(m=16, graph="permuted"),
                 rerank=kannolo.Rerank(kannolo.Plain("f16")))

# Hybrid: sparse first stage, rescored with multi-vector MaxSim.
index = kannolo.build(splade_docs, encoder=kannolo.DotPacking8(), index=kannolo.HNSW(),
                 rerank=kannolo.Rerank(kannolo.TwoLevelPQ(16), data=colbert_docs))
index.search(splade_queries, k=10, rerank_queries=colbert_queries, k_candidates=100)

# Residuals: the first stage holds 60% of each vector's L1 mass, the rescoring stage the rest.
index = kannolo.build(splade_docs, encoder=kannolo.Blocked(), index=kannolo.HNSW(),
                 transform=kannolo.L1Prune(60), rerank=kannolo.Rerank(kannolo.Plain("f16"), residual=True))

# Cascade: RaBitQ(1) graph -> 8-bit RaBitQ -> f32.
index = kannolo.build(x, encoder=kannolo.RaBitQ(1), index=kannolo.HNSW(),
                 rerank=[kannolo.Rerank(kannolo.RaBitQ(8)), kannolo.Rerank(kannolo.Plain("f32"))])
index.search(q, k=10, k_candidates=[400, 50])
```

`residual=True` stores what the first stage leaves out, and scores a candidate as the sum of the
two stages. For sparse data, the residual is the components `L1Prune` dropped. For dense data with
a Flat or HNSW first stage, it is `x - x̂`, where `x̂` is the first stage's RaBitQ reconstruction.
Residuals support inner product only.

### Composing an index from separately built parts

`kannolo.compose(index, rerank=...)` adds rescoring stages to an existing index, either built or
loaded. The rescoring stages are built from in-memory data, in the same input formats as `build`.
The index is shared without copying and is left unchanged. One first stage can therefore be
combined with several rescoring datasets. For example, three stored first stages and one stored
collection, rescored with three encoders, give nine indexes in memory:

```python
firsts = [kannolo.load(p) for p in ("hnsw_rabitq.index", "hnsw_pq.index", "ivf.index")]
data = np.load("collection.npy")
encoders = [kannolo.Plain("f32"), kannolo.Plain("f16"), kannolo.RaBitQ(bits=8)]
indexes = [kannolo.compose(f, kannolo.Rerank(e, data=data)) for f in firsts for e in encoders]

# Hybrid: a sparse first stage with multi-vector rescoring
sparse_first = kannolo.load("sparse_first.index")
hybrid = kannolo.compose(sparse_first, kannolo.Rerank(kannolo.TwoLevelPQ(subspaces=16), data=(tokens, doclens)))
```

Each `Rerank` needs its `data`, which holds the same items in the same order as the index. For
`residual=True`, `data` is the first stage's original collection. The residual is computed from it
in the same way as in `build`. Composing onto an index that already has rescoring stages appends a
stage (a cascade). A composed index saves and loads like any other, as one file.

### Filtered search (ACORN)

`filter=` returns only the neighbors a filter allows. On an HNSW index, ACORN applies the filter
during the graph traversal and does not post-filter the results. Recall therefore stays high when
few vectors qualify. Filtered search currently applies only to single-stage HNSW indexes (any
representation, encoder and graph layout). Other indexes raise `ValueError`.

```python
allowed = prices < 100                              # a boolean mask over the collection
dists, ids = index.search(queries, k=10, filter=allowed)
index.search(queries, k=10, filter=np.array([3, 17, 42]))   # or the allowed ids
index.search(queries, k=10, filter=lambda i: i % 2 == 0)    # or a callable, evaluated once per id
index.search(queries, k=10, filter=[mask_q0, mask_q1, ...]) # or one filter per query

# ACORN-γ: precompute expanded neighbor lists once, for very selective filters
index.build_acorn_gamma(gamma=3)
index.search(queries, k=10, filter=allowed)                    # now uses ACORN-γ
index.search(queries, k=10, filter=allowed, acorn_gamma=False) # ACORN-1
```

ACORN-1 expands two-hop neighbors on the fly and needs no preparation. With very selective
filters (around 1% of the collection), its recall drops. ACORN-γ is meant for this case. The
ACORN-γ neighbor lists cost memory and build time (a `gamma` of 2–4 is a good starting value).
`save` does not write them, so call `build_acorn_gamma` again after `load`. A callable filter is
evaluated in Python for every id before the search, so pass a mask when you filter repeatedly.
With a mask, batch filtered search runs on every core.

### Search parameters

`index.search(queries, k=10, *, rerank_queries=None, ef_search=None, nprobe=None,
early_exit_threshold=None, query_bits=None, k_candidates=None, alpha=None, beta=None,
rerank_query_bits=None, filter=None, acorn_gamma=None, num_threads=0)`

| Parameter | Applies to | Default |
|---|---|---|
| `ef_search` | HNSW first stage, IVF with HNSW centroids | 100 |
| `early_exit_threshold` | the same (distance-adaptive termination, λ) | off |
| `nprobe` | IVF first stage | 16 |
| `query_bits` | first stage with `RaBitQ(bits=1)` (query quantized to 1–8 bits) | 1 |
| `k_candidates` | each rescoring stage (value or list) | `max(10k, 100)` |
| `alpha` | each rescoring stage: candidate pruning | off |
| `beta` | each rescoring stage: early exit after `beta` stalls | off |
| `rerank_query_bits` | rescoring stage with `RaBitQ(bits=1)` | 1 |
| `rerank_queries` | rescoring stages in another representation (a dict if several) | — |
| `filter` | single-stage HNSW: filtered search (see [Filtered search](#filtered-search-acorn)) | off |
| `acorn_gamma` | filtered search: ACORN-γ (`True`) or ACORN-1 (`False`) | ACORN-γ once built |
| `num_threads` | `0`: every core, `1`: serial, `n`: `n` threads | 0 |

A parameter that does not apply to the index raises a `ValueError`. Missing results (fewer than
`k` candidates) are padded with distance `inf` and id `-1`. Under `"l2"`, an IVF reports
`‖q − x‖² − ‖q‖²`. It leaves out the term that is constant for the query, so the ranking is still
exact.

### Properties

`len(index)`, `index.dim` (the first stage's; the token dimension for multi-vector data),
`index.nbytes` (bytes held by every stage) and `index.spec` (the `kannolo.IndexSpec` it was built with).

### Table 7 of the paper, row by row

Every composition in Table 7 of the kANNolo paper is one `kannolo.build` call. `python/tests/test_table7.py`
builds each of them and checks it against an exact ground truth.

| Table 7 row | `kannolo.build(...)` |
|---|---|
| Dense, Flat, scalar / PQ / RaBitQ | `index=kannolo.Flat(), encoder=kannolo.Plain("f16")` / `kannolo.PQ(m)` / `kannolo.RaBitQ(b)` |
| Dense, Flat, PQ or RaBitQ + rescoring | `index=kannolo.Flat(), encoder=kannolo.PQ(m), rerank=kannolo.Rerank(kannolo.Plain())` |
| Dense, HNSW, scalar / PQ / RaBitQ | `index=kannolo.HNSW(), encoder=...` |
| HNSWPQ + refine, HNSW-RaBitQ | `index=kannolo.HNSW(), encoder=kannolo.PQ(m)` or `kannolo.RaBitQ(1)`, `rerank=kannolo.Rerank(kannolo.Plain("f16"))` (or PQ, RaBitQ) |
| IVFFlat, IVFPQ, IVF-HNSW, IVF-RaBitQ | `index=kannolo.IVF(n_lists, centroids=kannolo.Flat() \| kannolo.HNSW(), residual=...), encoder=kannolo.Plain() \| kannolo.PQ(m) \| kannolo.RaBitQ(b)` |
| IVF + rescoring | the same, plus `rerank=kannolo.Rerank(...)` |
| Sparse, Flat / HNSW, scalar / blocked / DotPacking8 | `index=kannolo.Flat()` or `kannolo.HNSW()`, `encoder=kannolo.Plain() \| kannolo.FixedPoint() \| kannolo.Scalar() \| kannolo.Blocked() \| kannolo.DotPacking8()` |
| Sparse, IVF | `index=kannolo.IVF(...), encoder=kannolo.Plain("f32" \| "f16")` |
| Sparse + scalar rescoring | `rerank=kannolo.Rerank(kannolo.Plain())` |
| Sparse, top-a% of L1 mass, scalar or blocked residuals | `transform=kannolo.L1Prune(a), rerank=kannolo.Rerank(kannolo.Plain("f16") \| kannolo.Blocked(), residual=True)` |
| Multi-vector, Flat / HNSW, scalar / PQ / two-level PQ | `index=kannolo.Flat()` or `kannolo.HNSW()`, `encoder=kannolo.Plain() \| kannolo.PQ(m) \| kannolo.TwoLevelPQ(m)` |
| Multi-vector + rescoring | `rerank=kannolo.Rerank(kannolo.Plain())` (or PQ, two-level PQ) |
| Hybrid: sparse → dense / multi-vector; dense → multi-vector | `rerank=kannolo.Rerank(encoder, data=other_representation)`, searched with `rerank_queries=` |
| Cascade | `rerank=[kannolo.Rerank(...), kannolo.Rerank(...)]` |

Not available from Python: an IVF as the centroid index of an IVF (`centroids=kannolo.IVF(...)`),
and IVF over sparse clusters stored as DotPacking8.

### Building a smaller library

Every combination is compiled ahead of time, so the library contains all of them. The published
wheels contain all of them. A build from source can leave out what it does not need with the Cargo
features `dense`, `sparse`, `multivec` and `ivf`. The default Python build enables all four:

| Feature | Adds |
|---|---|
| `dense` | dense data: Flat, HNSW and rescoring stages over dense vectors |
| `sparse` | sparse data: Flat, HNSW and rescoring stages over sparse vectors |
| `multivec` | multi-vector data, scored with MaxSim. In practice an add-on to `sparse` or `dense`: multi-vector data is mostly used as the rescoring stage of a hybrid pipeline |
| `ivf` | IVF first stages, for whichever of `dense` and `sparse` is enabled |

```bash
# Sparse indexes (Flat, HNSW, rescoring), no IVF:
maturin build --release --no-default-features --features python,sparse
# A sparse first stage rescored with multi-vector MaxSim (the hybrid pipeline):
maturin build --release --no-default-features --features python,sparse,multivec
# Dense HNSW/Flat and IVF:
maturin build --release --no-default-features --features python,dense,ivf
```

`kannolo.compiled_features()` lists what a build contains. Building, loading or searching something
the build left out raises a `ValueError` that names the feature to enable. Index files have the
same format in every build. Any build that contains a representation reads the files of that
representation written by any other build.

### Performance

Every stage is compiled ahead of time and fully specialized. The index is dispatched once per
stage in each `search` call. The search loops, rescoring loops and distance kernels are the same
code that a Rust program runs. Stages are combined at run time, so any first stage composes with
any rescoring dataset without compiling every pair.

A file written by `Index.save` starts with a small header that holds the spec, followed by the
payload of each stage. The CLI binaries (`hnsw_build`, `ivf_build`, ...) write their own
headerless files. `kannolo.load` does not read these files, and the CLI binaries do not read
the files with a header.

## Additional Resources

- **Notebooks** in [notebooks/](../notebooks/):
  - `dense_euclidean_benchmark.ipynb`, `sparse_plain_benchmark.ipynb`: ground truth with a Flat
    index, then an HNSW swept over `ef_search` and early termination
  - `sparse_multivec_rerank_benchmark.ipynb`: a sparse first stage composed with multi-vector
    (MaxSim) rescoring, evaluated with `ir_measures`
  - `filtered_search_acorn.ipynb`: post-filtering, ACORN-1 and ACORN-γ on a filtered benchmark
  - `compose_indexes.ipynb`: three first stages and three rescoring encodings, stored as parts and
    composed into nine indexes

- **Rust API**: See [RustUsage.md](RustUsage.md)
- **Running Experiments**: See [RunExperiments.md](RunExperiments.md)
