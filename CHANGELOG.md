# Changelog

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
