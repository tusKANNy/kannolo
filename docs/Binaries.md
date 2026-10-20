# Unified Build, Search, and Convert Binaries

This document describes the current CLI surface for:
- `hnsw_build`
- `hnsw_search`
- `hnsw_rerank_build_dense`, `hnsw_rerank_search_dense`
- `hnsw_rerank_build_sparse`, `hnsw_rerank_search_sparse`
- `ivf_build`
- `ivf_search`

`hnsw_rerank_search` is documented separately in [MultiVectorUsage.md](MultiVectorUsage.md).

All binaries are behind the `cli` feature (`hnsw_rerank_search` additionally needs `multivec`),
so a plain `cargo build` produces none of them:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
```

All examples and option names below are aligned with the current binaries.

## `hnsw_build`

```bash
Usage: hnsw_build [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE> --dataset-type <DATASET_TYPE>

Options:
  -d, --data-file <DATA_FILE>
  -o, --output-file <OUTPUT_FILE>
      --dataset-type <DATASET_TYPE>      [possible values: dense, sparse]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: see "Sparse value types" below]
      --component-type <COMPONENT_TYPE>  [default: u16] [possible values: u16, u32]
      --encoder <ENCODER>                [default: plain] [possible values: plain, pq, rabitq, rabitq-ext]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
      --m <M>                            [default: 16]
      --ef-construction <EF_CONSTRUCTION> [default: 150]
      --distance <DISTANCE>              [default: dotproduct]
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
      --nbits <NBITS>                    [default: 8] (ignored by vectorium PQ)
      --sample-size <SAMPLE_SIZE>        [default: 100000] (ignored by vectorium PQ)
      --rabitq-seed <RABITQ_SEED>        [default: 42] Seed for the random orthogonal rotation
      --rabitq-no-rotate                 Skip the rotation: faster, but expect recall to drop
      --rabitq-total-bits <BITS>         [default: 4] Document code width, rabitq-ext only [2, 4, 8]
      --rabitq-exact-quant               Per-vector rescale search instead of the trained constant
```

## `hnsw_search`

```bash
Usage: hnsw_search [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE> --dataset-type <DATASET_TYPE> --distance <DISTANCE>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>
  -o, --output-path <OUTPUT_PATH>
      --dataset-type <DATASET_TYPE>      [possible values: dense, sparse]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: see "Sparse value types" below]
      --component-type <COMPONENT_TYPE>  [default: u16] [possible values: u16, u32]
      --encoder <ENCODER>                [default: plain] [possible values: plain, pq, rabitq, rabitq-ext]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
      --distance <DISTANCE>
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
  -k, --k <K>                            [default: 10]
      --ef-search <EF_SEARCH>            [default: 40]
      --query-bits <QUERY_BITS>          [default: 1] Query code width for --encoder rabitq (1..=8)
      --early-termination <EARLY_TERMINATION> [default: none] [possible values: none, distance-adaptive]
      --lambda <LAMBDA>                  [default: 1]
      --num-runs <NUM_RUNS>              [default: 1]
```

## Two-stage indexes: `hnsw_rerank_{build,search}_{dense,sparse}`

A two-stage index has an HNSW first stage that retrieves `--k-candidates` candidates and a rerank
dataset that rescores them. The build binaries take one collection, build both stages, and save
them to one file; the search binaries load that file. The files are the same ones the Python
classes `DenseRerankHNSW` and `SparseRerankHNSW` save and load, so either side can build and the
other search.

Index files carry no header, so the search binary must be given the build options that select the
index type: they are listed for each pair below.

Both search binaries share the search options: `--k-candidates` (first-stage candidates to
rerank), `--ef-search` (first-stage candidate list size), `--alpha` (candidate pruning on the
first-stage score), `--beta` (early exit during reranking), `--early-termination` and `--lambda`,
and `--num-runs`. Results are written as `query_id  doc_id  rank  score`.

### Dense

The first stage is a graph built over `f16` whose dataset is then encoded with PQ, RaBitQ or
RaBitQ-ext. The rerank stage is the collection at plain `f16`, or RaBitQ-ext codes with `--rerank-bits 2|4|8`. `--residual-rerank` encodes `x − x̂` instead, and scores a candidate as first-stage plus rerank score; it needs `--rerank-bits`, a `rabitq` or `rabitq-ext` first stage, and dot product.

```bash
Usage: hnsw_rerank_build_dense [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE> --encoder <ENCODER>

Options:
  -d, --data-file <DATA_FILE>            f32 .npy of shape (n, dim)
  -o, --output-file <OUTPUT_FILE>
      --encoder <ENCODER>                [possible values: pq, rabitq, rabitq-ext]
      --pq-subspaces <PQ_SUBSPACES>      [default: 32]
      --rabitq-total-bits <BITS>         [default: 4] rabitq-ext only [2, 4, 8]
      --rerank-bits <RERANK_BITS>        [default: 0] 0 for f16, or 2, 4, 8 for RaBitQ-ext
      --residual-rerank
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, permuted, streamvbyte]
      --distance <DISTANCE>              [default: dotproduct]
      --m <M>                            [default: 32]
      --ef-construction <EF_CONSTRUCTION> [default: 200]
```

```bash
Usage: hnsw_rerank_search_dense [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE> --encoder <ENCODER>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>          f32 .npy
  -o, --output-path <OUTPUT_PATH>
      --encoder <ENCODER>                [possible values: pq, rabitq, rabitq-ext]
      --pq-subspaces <PQ_SUBSPACES>      [default: 32]
      --rabitq-query-bits <BITS>         [default: 1] rabitq only (1..=8)
      --rerank-bits <RERANK_BITS>        [default: 0]
      --residual-rerank
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, permuted, streamvbyte]
      --distance <DISTANCE>              [default: dotproduct]
  -k, --k <K>                            [default: 10]
      --k-candidates <K_CANDIDATES>      [default: 100]
      --ef-search <EF_SEARCH>            [default: 100]
      --alpha <ALPHA>
      --beta <BETA>
      --early-termination <EARLY_TERMINATION> [default: none] [possible values: none, distance-adaptive]
      --lambda <LAMBDA>
      --num-runs <NUM_RUNS>              [default: 1]
```

Must match the build: `--encoder`, `--pq-subspaces`, `--distance`, `--graph-type`,
`--residual-rerank`, and whether `--rerank-bits` is 0 or one of 2, 4, 8. RaBitQ-ext's total bits and the exact
rerank bits are stored with the index. `--rabitq-query-bits` only changes how the query is encoded,
so it can differ from the build.

```bash
hnsw_rerank_build_dense --data-file train.npy --output-file idx.bin \
  --encoder rabitq-ext --rabitq-total-bits 4 --rerank-bits 8 --graph-type permuted
hnsw_rerank_search_dense --index-file idx.bin --query-file queries.npy \
  --encoder rabitq-ext --rerank-bits 8 --graph-type permuted \
  -k 10 --k-candidates 50 --ef-search 50 --output-path results.tsv
```

### Sparse

Dot product only, over collections in Seismic binary format. Three build options decide what each
stage holds:

- `--prune-percentage` below 100 L1-prunes the first stage to the components carrying that
  percentage of each vector's L1 mass.
- `--first-stage-value-type` stores the first stage as `f16` or `fixedu8`. The rerank stage is
  always `f16`.
- `--residual-rerank` stores only the pruned-away components in the rerank stage and sums the two
  scores; with an `f16` first stage this is the exact score at `f16` precision, and each non-zero
  is stored once.
  Without it the rerank stage holds the full vectors.

No pruning with `--residual-rerank` (the residual would be empty) and no pruning with an `f16`
first stage (the rerank stage would be an identical copy) are rejected.

```bash
Usage: hnsw_rerank_build_sparse [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE>

Options:
  -d, --data-file <DATA_FILE>
  -o, --output-file <OUTPUT_FILE>
      --prune-percentage <PRUNE_PERCENTAGE>  required; 100 means no pruning
      --residual-rerank
      --first-stage-value-type <TYPE>    [default: f16] [possible values: f16, fixedu8]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, permuted, streamvbyte]
      --m <M>                            [default: 32]
      --ef-construction <EF_CONSTRUCTION> [default: 200]
```

```bash
Usage: hnsw_rerank_search_sparse [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>          Seismic binary format
  -o, --output-path <OUTPUT_PATH>
      --residual-rerank
      --first-stage-value-type <TYPE>    [default: f16] [possible values: f16, fixedu8]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, permuted, streamvbyte]
  -k, --k <K>                            [default: 10]
      --k-candidates <K_CANDIDATES>      [default: 100]
      --ef-search <EF_SEARCH>            [default: 100]
      --alpha <ALPHA>
      --beta <BETA>
      --early-termination <EARLY_TERMINATION> [default: none] [possible values: none, distance-adaptive]
      --lambda <LAMBDA>
      --num-runs <NUM_RUNS>              [default: 1]
```

Must match the build: `--first-stage-value-type`, `--graph-type` and `--residual-rerank`.

```bash
hnsw_rerank_build_sparse --data-file documents.bin --output-file idx.bin \
  --prune-percentage 60 --residual-rerank --m 32 --ef-construction 1000
hnsw_rerank_search_sparse --index-file idx.bin --query-file queries.bin --residual-rerank \
  -k 10 --k-candidates 100 --ef-search 100 --output-path results.tsv
```

## `ivf_build`

Builds an inverted-file index. Dense only.

```bash
Usage: ivf_build [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE>

Options:
  -d, --data-file <DATA_FILE>
  -o, --output-file <OUTPUT_FILE>
      --distance <DISTANCE>              [default: euclidean] [possible values: euclidean, dotproduct]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f32, f16]
      --n-clusters <N_CLUSTERS>          [default: 1024]
      --kmeans-n-iter <KMEANS_N_ITER>    [default: 25]
      --kmeans-n-redo <KMEANS_N_REDO>    [default: 1]
      --kmeans-sample-size <SIZE>        (optional; defaults to the whole dataset)
      --kmeans-hnsw                      Use an HNSW index to speed up k-means assignment
      --kmeans-spherical                 L2-normalize centroids each iteration
      --residuals                        Encode vectors as residuals from their centroid
      --hnsw                             Index the centroids with HNSW instead of scanning them
      --m-hnsw <M_HNSW>                  [default: 32] (alias: --m) Only with --hnsw
      --ef-construction <EF_CONSTRUCTION> [default: 200] Only with --hnsw
      --m-pq <M_PQ>                      (optional) PQ-encode the vectors with this many subspaces
```

## `ivf_search`

```bash
Usage: ivf_search [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>
  -o, --output-path <OUTPUT_PATH>
  -k, --k <K>                            [default: 10]
      --n-probe <N_PROBE>                [default: 32] Clusters visited per query
      --distance <DISTANCE>              [default: euclidean] [possible values: euclidean, dotproduct]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f32, f16]
      --hnsw                             Must match the index
      --m-pq <M_PQ>                      Must match the index
      --ef-search <EF_SEARCH>            [default: 40] Only with --hnsw
      --lambda <LAMBDA>                  [default: 0]
      --num-runs <NUM_RUNS>              [default: 1]
```

As with HNSW, `--distance`, `--value-type`, `--hnsw` and `--m-pq` are properties baked into the
index at build time and must be repeated identically at search time: each one names part of the
concrete type the index has to be loaded as. `ivf_build --residuals` has no search-time
counterpart — residual-vs-plain is runtime state stored in the index, so it is read back from the
index rather than restated.

## Examples

Dense plain:

```bash
./hnsw_build --data-file data.npy --output-file index.bin \
  --dataset-type dense --encoder plain --value-type f32 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Dense PQ:

```bash
./hnsw_build --data-file data.npy --output-file index.bin \
  --dataset-type dense --encoder pq --pq-subspaces 16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse plain with explicit sparse component type:

```bash
./hnsw_build --data-file data.bin --output-file index.bin \
  --dataset-type sparse --encoder plain --value-type f16 --component-type u16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse DotPacking8 (the replacement of the removed `dotvbyte` encoder):

```bash
./hnsw_build --data-file data.bin --output-file index.bin \
  --dataset-type sparse --value-type dotpacking8-fixedu8 --component-type u16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse DotPacking8 search (`--value-type` and `--component-type` must match the build):

```bash
./hnsw_search --index-file index.bin --query-file queries.bin \
  --dataset-type sparse --value-type dotpacking8-fixedu8 --component-type u16 \
  --distance dotproduct --k 10 --ef-search 40 --output-path results.tsv
```

Compressed graph (see [GraphCompression.md](GraphCompression.md)):

```bash
./hnsw_build --data-file data.npy --output-file index_svb.bin \
  --dataset-type dense --encoder plain --value-type f32 --graph-type streamvbyte \
  --m 16 --ef-construction 150 --distance dotproduct

./hnsw_search --index-file index_svb.bin --query-file queries.npy \
  --dataset-type dense --encoder plain --value-type f32 --graph-type streamvbyte \
  --distance dotproduct --k 10 --ef-search 40 --output-path results.tsv
```

IVF:

```bash
./ivf_build --data-file data.npy --output-file ivf.bin \
  --distance euclidean --value-type f32 --n-clusters 1024

./ivf_search --index-file ivf.bin --query-file queries.npy \
  --distance euclidean --value-type f32 --n-probe 32 --k 10 --output-path results.tsv
```

## Sparse value types

For sparse datasets, `--value-type` selects how the dataset is stored (`--encoder` stays `plain`).
`hnsw_search` must be given the same `--value-type` and `--component-type` used at build time.

| Value type | Component types | Distances | Storage |
|---|---|---|---|
| `f32`, `f16` | `u16`, `u32` | dotproduct, euclidean | plain values; the graph is built on them directly |
| `fixedu8`, `fixedu16` | `u16`, `u32` | dotproduct, euclidean | fixed-point values (one scale for all components) |
| `scalaru8`, `scalaru16` | `u16`, `u32` | dotproduct, euclidean | 8/16-bit codes with one trained step per component |
| `blocked-sparse` | `u16` | dotproduct | 64-byte blocks of 16 entries, `f16` values |
| `blocked-sparse-scalaru8`, `blocked-sparse-scalaru16` | `u16` | dotproduct | blocks with 8/16-bit per-component codes |
| `dotpacking8-fixedu8` | `u16` | dotproduct | DotPacking8, fixed-point 8-bit values; replaces `dotvbyte` |
| `dotpacking8-scalaru8`, `dotpacking8-scalaru16` | `u16`, `u32` (bridged) | dotproduct | DotPacking8, 8/16-bit per-component codes |
| `dotpacking8-f16` | `u16` | dotproduct | DotPacking8, `f16` values |
| `dotpacking8-scalar4bit` | `u16` | dotproduct | DotPacking8, 4-bit per-component codes |

Every value type except `f32` and `f16` builds the graph on the `f32` dataset and then
re-encodes the dataset. The blocked layouts need AVX2/F16C/FMA (build with
`RUSTFLAGS="-C target-cpu=native"`) and are not compiled on other architectures. Indexes built
with the removed `dotvbyte` encoder cannot be loaded and must be rebuilt.

## Validation Rules

The binaries reject invalid combinations:

1. `pq` is dense-only.
2. `rabitq` and `rabitq-ext` are dense-only.
3. Dense datasets accept only the `f32` and `f16` value types; every other value type is sparse-only.
4. `component-type` is sparse-only.
5. The blocked and DotPacking8 value types support only `dotproduct`, and only `component-type = u16` except `dotpacking8-scalaru8` and `dotpacking8-scalaru16`.
6. `dotvbyte` (as `--encoder` or `--value-type`) was removed; the binaries exit naming `dotpacking8-fixedu8` instead.
7. `pq-subspaces` must be one of `4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256` and must divide the vector dimensionality.
8. For PQ, `--nbits` and `--sample-size` are accepted for compatibility but ignored by vectorium.
9. `--rabitq-total-bits` must be 2, 4 or 8; for a 1-bit code use `--encoder rabitq` instead.
10. `--graph-type streamvbyte` caps the ground level at 256 neighbors per node, so it requires `--m` of at most 128. This is checked before the build starts, not after.

Beyond these, `dataset-type`, `value-type`, `component-type`, `encoder`, `distance` and `graph-type` are all baked into the index at build time. `hnsw_search` cannot detect them — index files carry no header — so passing a different value produces a decode error rather than a helpful message. Repeat the build flags exactly.

`hnsw_search --query-bits` is the exception: it scalar-quantizes the *query* for `--encoder rabitq`
and is not stored in the index, so one index serves every setting and it can be varied without
rebuilding. Every other encoder ignores it, `rabitq-ext` included — that one's query width follows
the document code width stored with the index, which is why there is no search-time
`--rabitq-total-bits`.
