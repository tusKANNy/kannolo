# Changelog

## 0.18.0

### Breaking changes

- **The specific Python classes are removed**: `DensePlainHNSW`, `DensePQHNSW`, `DenseFlatIndex`,
  `SparsePlainHNSW`, `SparseDotPacking8HNSW`, `SparseFixedU8HNSW`, `SparseFixedU16HNSW`,
  `SparseFlatIndex`, `DenseRerankHNSW`, `SparseRerankHNSW`, `SparseMultivecRerankIndex` and
  `SparseMultivecTwoLevelsPQRerankIndex`. Each is now one `kannolo.build` call. For example,
  `DensePlainHNSW` is `kannolo.build(x, encoder=kannolo.Plain("f16"), index=kannolo.HNSW())`.
  `SparseFixedU8HNSW` is `encoder=kannolo.FixedPoint(8)`, `SparsePlainHNSW` is
  `encoder=kannolo.Blocked("f16")`, `DenseRerankHNSW` is an HNSW over `kannolo.PQ`/`kannolo.RaBitQ` with
  `rerank=kannolo.Rerank(kannolo.Plain("f16"))`, and `SparseRerankHNSW` is `transform=kannolo.L1Prune(p)` with
  `rerank=kannolo.Rerank(..., residual=True)`. `kannolo.load` does not read index files saved by
  these classes; rebuild them. ACORN filtered search (`search_filtered`, `build_acorn_gamma`,
  `search_filtered_gamma`) is now `search(..., filter=..., acorn_gamma=...)` and
  `Index.build_acorn_gamma`. Not carried over: loading precomputed ColBERTv2 multi-vector
  artifacts from a folder (`kannolo.TwoLevelPQ` trains its codebooks from the token vectors
  instead).

### Added

- **Composable Python API**: `kannolo.build(data, metric=..., encoder=..., index=...,
  rerank=...)` builds any composition of Table 7 of the kANNolo paper. It accepts dense, sparse
  (`u16`/`u32`) or multi-vector data; `ip`, `l2` or `maxsim`; the `Plain`, `FixedPoint`,
  `Scalar`, `PQ`, `RaBitQ`, `Blocked`, `DotPacking8` and `TwoLevelPQ` encoders; `Flat`, `HNSW`
  (`standard`/`permuted`/`compressed` graphs) and `IVF` (flat or HNSW centroids, optional
  residual clusters) first stages; and an optional `L1Prune` transform. Any number of rescoring
  stages (`kannolo.Rerank`) can follow, each with its own encoder, metric, representation and
  data. They give two-stage, hybrid (e.g. sparse → multi-vector) and cascade indexes, including
  residual rescoring. Data comes from numpy arrays, CSR-style tuples, `scipy.sparse` matrices, lists of
  token arrays, or `.npy`/`.bin` files. One `Index` class covers every combination: `search`
  (single query or batch), `save`, `kannolo.load(path)`. `kannolo.check(...)` validates a
  combination without building it. All stages share one Rust interface, called once per batch
  of queries, so any first stage composes with any rescoring stage without compiling every
  pair. The search and rescoring loops are still compiled for the concrete types. See
  [docs/PythonUsage.md](docs/PythonUsage.md#composable-api).
- Filtered search with ACORN-1 and ACORN-γ for single-stage HNSW indexes of any
  representation, encoder and graph layout (before, `DensePlainHNSW` only). The API is
  `Index.search(..., filter=...)`, with a boolean mask, an id array, a callable or one filter per
  query, and `Index.build_acorn_gamma(gamma)`. With a mask, batch filtered search runs in
  parallel.
- `kannolo.compose(index, rerank=...)`: adds rescoring stages, built from in-memory data, to an
  existing (built or loaded) index. The new index shares the existing one without copying it.
  First stages and rescoring datasets can be stored separately and combined at load time.
- `rerank_index::rerank_candidates` and `rerank_candidates_with_early_exit`: the rescoring of
  `RerankIndex`, as free functions over any rerank dataset.
- `IVF::centroid_index`, `IVF::cluster_dataset` and `IVF::metadata_space_usage_bytes`.
- Cargo features `dense`, `sparse` and `ivf` (with the existing `multivec`) select what the
  composable Python API compiles. Builds from source with fewer features are smaller and faster
  to compile. The default Python build enables all of them. `kannolo.compiled_features()` lists
  what a build contains. Using a part that the build left out raises an error that names the
  feature.
- Files written by `Index.save` are self-describing: a header records the spec, so `load` takes only the path and no
  arguments.
- The `dev-python` Cargo profile for faster local builds (`maturin develop --profile
  dev-python`): 16 codegen units, about 3x faster to compile and about 2% slower search.
  `kannolo.build_profile` names the profile a build used. `import kannolo` warns on a
  `dev-python` build, so that it is not used for benchmarks by mistake.
- Notebooks `filtered_search_acorn.ipynb` (post-filtering, ACORN-1 and ACORN-γ on a filtered
  benchmark) and `compose_indexes.ipynb` (first stages and rescoring stages stored separately and
  composed). The three existing notebooks use `kannolo.build` and `kannolo.compose`.
- CI: the `Test Python Package` workflow builds the bindings with `maturin develop --release`,
  checks the smaller feature builds, and runs the Python tests.

### Changes

- The Python package is now a mixed Rust/Python package: the native module is
  `kannolo._kannolo`, wrapped by the pure-Python `kannolo` package.
- The `python` feature no longer depends on `ndarray` and `ndarray-npy` (the `cli` feature still
  does).
- `scripts/acorn_benchmark.py` uses the composable API, and turns each predicate into a mask
  outside the timed region.
- Python dense PQ (`kannolo.PQ`) is compiled for 8, 16, 32, 64 and 128 subspaces (the removed
  `DensePQHNSW` took ten counts up to 384). For other code sizes, the error suggests RaBitQ, with
  1, 2, 4 or 8 bits per component.

## 0.17.0

### Breaking changes

- **`--kmeans-hnsw` is replaced by `--kmeans-ann auto|on|off`** (default `auto`), and
  `KMEANS_HNSW_THRESHOLD` (32,768) by `KMEANS_ANN_THRESHOLD`. `run_experiments.py` reads
  `kmeans-ann` and still maps an old `kmeans-hnsw = true` to `--kmeans-ann on`.
- **`IVFBuildParams` gains `ann_assignment: AnnAssignment`**, and `prepare_ivf`'s
  `kmeans_hnsw: bool` parameter is renamed `ann_assignment`.
- **`prepare_ivf` and `IVF::build_index` require `G::Owned: IvfCentroids`** for the centroid type.
  It is implemented for the dense (`f32`/`f16`) and sparse (`u16`/`u32`) scalar datasets, under
  squared Euclidean distance and dot product.

### Added

- `IVF::over_centroids` builds a plain IVF over vectors filed under a given coarse quantizer;
  `IVF::refile` re-files new vectors under an existing IVF's coarse quantizer.
  `assign_to_centroids` returns the clustering (`BuildArtifacts`) of a dataset against fixed
  centroids.
- `IvfCentroidIndexBuilder` (a vectorium `CentroidIndexBuilder`) and the `IvfCentroids` trait,
  which use an `IVF` as the centroid index of k-means.

### Changes

- **`auto` turns ANN assignment on from 4,096 clusters** (was 32,768). On SIFT1M, k-means +
  assignment at 4,096 clusters drops from 17.4 to 4.6 s (Lloyd, 10 epochs) and from 5.0 to 1.6 s
  (e-means, 2 epochs), with IVF recall@10 unchanged (within 0.2 points at nprobe 4–64). Builds
  with 4,096–32,767 clusters therefore produce different (approximately assigned) clusters.
- **The ANN index is chosen by solver.** Lloyd keeps assigning through an HNSW over the
  centroids, rebuilt every iteration. E-means now assigns through kANNolo's own `IVF` built over
  the centroids (`IvfCentroidIndexBuilder`): it trains its coarse quantizer once (4·√k lists,
  rounded up to a power of two) and, after every mini-batch, only re-files the moved
  centroids. E-means refreshes the index tens of times per epoch, and rebuilding an HNSW each
  time cost more than the clustering itself (vectorium PR #3, E3). On SIFT1M with 32,768
  clusters, e-means k-means takes 28.0 s with an HNSW and 5.1 s with the IVF (Lloyd with HNSW:
  10.6 s). On FiQA ColBERTv2 (65,536 centroids, e-means 2 epochs, nprobe 16), 256, 512 and
  1,024 lists take 50, 28 and 20 s at about the same clustering error, hence 4·√k lists. The
  final assignment of every vector still uses an HNSW.

## 0.16.0

Moves to vectorium 0.12.0.

- **k-means solver for IVF.** `IVFBuildParams` replaces `kmeans_n_iter` with `kmeans_solver: Solver`
  (`Solver::default()`, i.e. vectorium's default) and `kmeans_epochs: Option<f64>` (`None` = the
  solver's default budget). `ivf_build` gains `--kmeans-solver lloyd|emeans` (default: vectorium's
  default solver) and `--kmeans-epochs` (default 25 for Lloyd, 5 for e-means); `--kmeans-n-iter` is
  kept as an alias of `--kmeans-epochs`.

The Python wheel is built with the `multivec` feature, so `SparseMultivecRerankIndex` and
`SparseMultivecTwoLevelsPQRerankIndex` are available from `pip install kannolo`.

## 0.15.1

Pins the Rust toolchain to `nightly-2026-06-19` (`rustc 1.98.0-nightly (bc2112ed5 2026-06-18)`). No code changes.

## 0.15.0

The blocked sparse encoder (vectorium 0.11.1, `f16` values) becomes the default for sparse data
under dot product. It is usually faster than plain `f16`, and at worst a few percent slower.

### Breaking changes

- **Defaults change.**

  | Where | Before | Now |
  |---|---|---|
  | `hnsw_build` / `hnsw_search`, sparse | `--value-type f32` | `--value-type blocked-sparse` |
  | `hnsw_rerank_{build,search}_sparse --first-stage-value-type` | `f16` | `blocked-sparse` |
  | Python `SparseRerankHNSW(first_stage_value_type=...)` | `"f16"` | `"blocked-sparse"` |
  | Rust `SparseRerankConfig::l1_pruned` | `SparseFirstStage::F16` | `SparseFirstStage::Blocked` |
  | Python `SparsePlainHNSW` | plain `f16`, `metric` argument | blocked sparse, dot product only (`metric` removed) |

  The blocked encoder supports only dot product and `u16` components: sparse Euclidean distance or
  `u32` components in the binaries need an explicit `--value-type` (e.g. `f16`).
- **Index files do not carry their value type.** Two-stage indexes built with the old default load
  only if `f16` is passed explicitly. Python `SparsePlainHNSW` dot-product indexes saved by 0.14
  do not load and must be rebuilt.

### New

- **Blocked two-stage index.** `SparseFirstStage::Blocked` (`blocked-sparse` in the binaries and
  Python) stores both stages blocked. The graph is built over the `f16` first stage and both
  datasets are then re-encoded, so the graph is the one an `f16` build gives and every score is the
  same `f16` product. It supports the standard and permuted graph layouts.
- **Blocked single-stage graphs are built over `f16`.** `hnsw_build --value-type blocked-sparse`
  (and Python `SparsePlainHNSW`) build the graph on the `f16` collection, then re-encode it; 0.14
  built it on `f32`.
- **Python sparse searches sort each query by component.** The blocked kernel walks a short query
  (fewer than 33 non-zeros) in component order; numpy queries need not be sorted.

## 0.14.0

kANNolo moves to vectorium 0.11.1. That release removes DotVByte and adds the
DotPacking8, blocked and step-scalar sparse encoders, all with x86_64 (AVX2) and aarch64 (NEON)
kernels.

### Breaking changes

- **DotVByte is removed.** Use `dotpacking8-fixedu8` instead. It keeps DotVByte's fixed-point
  8-bit values with the DotPacking8 component packing.

  | Removed | Use instead |
  |---|---|
  | `hnsw_build` / `hnsw_search --encoder dotvbyte` | `--value-type dotpacking8-fixedu8` (with `--encoder plain`, the default) |
  | Python `SparseDotVByteHNSW` | `SparseDotPacking8HNSW` (same methods; u16 components, dot product) |
  | Rust `PackedSparseDataset<DotVByteFixedU8Encoder>` | `vectorium::DotPacking8U16FixedU8Dataset` |

  Passing `dotvbyte` to the binaries is an error, and the message names the replacement.
- **DotVByte indexes must be rebuilt.** Index files built with `dotvbyte` do not load.
- **`Dataset::prefetch_with_range` takes a byte cap** (vectorium 0.11.1). The HNSW search passes
  256 bytes per neighbour, as before; vectorium 0.11.0's fixed 512 made HNSW search 3-6% slower.
  Custom `Dataset` implementations need the extra `max_bytes` argument.

### Sparse value types

Sparse `hnsw_build` / `hnsw_search` accept every sparse dataset type of vectorium 0.11 through
`--value-type`. The full table is in [docs/Binaries.md](docs/Binaries.md#sparse-value-types).

| Value type | Component types | Distances |
|---|---|---|
| `f32`, `f16`, `fixedu8`, `fixedu16` | `u16`, `u32` | dotproduct, euclidean |
| `scalaru8`, `scalaru16` (new: one quantization step per component) | `u16`, `u32` | dotproduct, euclidean |
| `blocked-sparse`, `blocked-sparse-scalaru8`, `blocked-sparse-scalaru16` (new) | `u16` | dotproduct |
| `dotpacking8-fixedu8`, `dotpacking8-f16`, `dotpacking8-scalar4bit` (new) | `u16` | dotproduct |
| `dotpacking8-scalaru8`, `dotpacking8-scalaru16` (new) | `u16`, `u32` (bridged) | dotproduct |

The index is built the same way as before:
- `f32` and `f16` build the graph on the stored values.
- Every other value type builds the graph on `f32`, then re-encodes the dataset.

Measured on 100k MS MARCO cocondenser documents (`m = 16`, `ef_search = 100`, 1000 queries):

| Value type | Dataset bytes | Recall@10 |
|---|---|---|
| `f32` | 72.9 MB | 0.931 |
| `scalaru8` | 37.0 MB | 0.928 |
| `fixedu8` | 36.8 MB | 0.926 |
| `dotpacking8-fixedu8` | 25.8 MB | 0.926 |
| `dotpacking8-scalaru8` | 25.9 MB | 0.927 |

### Unchanged

- The sparse rerank first stage still uses `FixedU8Q`.
- IVF still stores clusters as plain `f32` / `f16`.
