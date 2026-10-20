# HNSW Graph Compression

kANNolo can reorder and compress the HNSW ground-level neighbor lists to shrink the index on disk, with no loss of search accuracy. This is controlled by the `--graph-type` flag (CLI) or the `graph_type` argument (Python), independently of the dataset/encoder/value-type choices documented in [Binaries.md](Binaries.md) and [PythonUsage.md](PythonUsage.md).

## How It Works

Two independent techniques are combined:

- **EGB (Enhanced Graph Bisection) permutation**: nodes are reassigned new IDs so that nodes with overlapping neighborhoods end up with adjacent IDs. This shrinks the numeric gaps between a node's neighbor IDs, which is what makes delta-based compression effective.
- **StreamVByte compression**: each node's neighbor list is sorted, delta-encoded (each ID stored as the difference from the previous one), and byte-packed with StreamVByte.

Permutation and compression only change how the graph is *stored*; they never change what a search returns. This was verified end-to-end: the Rust test suite includes a dedicated test asserting byte-identical search results between the baseline graph and its permuted/compressed counterparts, and manual CLI and Python runs confirmed identical result sets (IDs and distances) across all graph types on the same data.

Because reordering happens internally, result vector IDs are always reported against the **original** external IDs — the index stores the inverse permutation and applies it transparently when returning results. You never need to think about the internal node order as a caller.

## The Four Graph Types

The CLI and the Python API name these differently: the CLI names the *storage format*, Python names the *effect*. They are the same four (three, in Python) code paths.

| CLI `--graph-type` | Python `graph_type` | Node order | Neighbor storage | Reduces size? |
|---|---|---|---|---|
| `standard` | `"standard"` | Original (insertion order) | Plain, uncompressed | — (baseline) |
| `fixed-degree` | *not exposed* | Original (insertion order) | Fixed-width padded | No (usually larger) |
| `permuted` | `"permuted"` | EGB-reordered | Plain, uncompressed | No — slightly larger than `standard` |
| `streamvbyte` | `"compressed"` | EGB-reordered | Delta + StreamVByte compressed | **Yes** |

All four return identical search results. They differ only in index size and query speed.

`fixed-degree` is a third neighbor-storage backend alongside plain and StreamVByte, one that pads every node's list to a common stride so a list can be located without an offsets array. 

The EGB reordering is worth having on its own: it clusters nodes that are traversed together, so a beam search touches fewer cache lines and **queries get faster**. That is what `permuted` gives you. Any reordered index also stores a ground-level inverse permutation so results come back as original dataset IDs, which costs about `ceil(log2(n))` bits per vector — so `permuted` comes out marginally *larger* than `standard`, and it buys query speed rather than space.

`streamvbyte` adds compression on top of that reordering — in local testing it reduced the graph portion of the index by roughly half compared to `standard`, with the exact ratio depending on `M` and the data's neighbor-ID locality after permutation. The per-node decode it costs is small enough that the locality gain pays for it, so a compressed index is not slower to query than `standard`.

## CLI Usage

Both `hnsw_build` and `hnsw_search` accept:

```bash
--graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
```

`--graph-type` must match between build and search — an index built with `streamvbyte` must also be searched with `--graph-type streamvbyte`.

Build a compressed index:

```bash
./hnsw_build --data-file data.npy --output-file index.bin \
  --dataset-type dense --encoder plain --value-type f32 \
  --graph-type streamvbyte \
  --m 16 --ef-construction 150 --distance dotproduct
```

Search it:

```bash
./hnsw_search --index-file index.bin --query-file queries.npy \
  --dataset-type dense --encoder plain --value-type f32 \
  --graph-type streamvbyte \
  --distance dotproduct --k 10 --ef-search 40
```

`permuted` and `streamvbyte` are supported for every dataset/encoder combination documented in [Binaries.md](Binaries.md) (dense plain, dense PQ, dense RaBitQ, and every sparse value type).

## Python Usage

The `graph_type` keyword is available on `build_from_file`, `build_from_array`/`build_from_arrays`, and `load` for every HNSW-backed class: `DensePlainHNSW`, `DensePQHNSW`, `SparsePlainHNSW`, `SparseDotPacking8HNSW`, `SparseFixedU8HNSW`, `SparseFixedU16HNSW`. It is not available on the flat (brute-force) or multivector reranking indexes, since they have no HNSW graph to compress.

**You can ignore it entirely.** Omitting it gives you `"standard"`, the plain uncompressed index:

```python
from kannolo import DensePlainHNSW
import numpy as np

data = np.random.rand(10000, 768).astype(np.float32)

# No graph_type: a standard index.
index = DensePlainHNSW.build_from_array(data.flatten(), dim=768)
```

Pass `graph_type="compressed"` when you want the index to take less space. Nothing else about your code changes — the results are the same and the IDs are still the original dataset IDs:

```python
index = DensePlainHNSW.build_from_array(
    data.flatten(), dim=768,
    m=32, ef_construction=200, metric="dotproduct",
    graph_type="compressed",
)

index.save("index.bin")

# graph_type must match what the index was built with: the file does not record it.
loaded = DensePlainHNSW.load("index.bin", metric="dotproduct", graph_type="compressed")

query = np.random.rand(768).astype(np.float32)
distances, ids = loaded.search(query, k=10, ef_search=100)
```

`graph_type="permuted"` is the third option: it applies the reordering without the compression, so queries get faster but the index does not get smaller.

Note the naming difference from the CLI: Python's `"compressed"` is the CLI's `streamvbyte`. Python does not expose `fixed-degree`.

## Constraint: Neighbor List Length

StreamVByte blocks support adjacency lists of **at most 256 neighbors per node**. The ground level's max degree is `2 × M`, so `M` must be at most 128 when using `--graph-type streamvbyte` / `graph_type="compressed"`. The defaults (`M = 16`/`32`) are well within range.

Both front ends check this before building rather than letting it fail at the end: the CLI exits with an error, and Python raises `ValueError` from the build call.

## Index format breaks

Indexes are encoded with bincode's fixed-int configuration, which is positional and not self-describing: any change to a serialized struct's fields is a hard break in both directions, failing with a decode error rather than a diagnosable message. Two such breaks have happened.

**When permutation landed (0.8.0).** Permuted indexes carry a ground-level inverse permutation that the serialized `HNSW` did not previously have. This broke every graph type, not just `permuted`/`streamvbyte`.

**When fixed-degree became a backend (0.10.0).** `GraphFixedDegree` used to be its own type with its own storage and its own ID mapping; it is now `Graph<FixedDegreeNeighbors>`, so its serialized layout is the one every other graph type already used. **Only `fixed-degree` indexes are affected** — files written by 0.9.2 as `standard`, `permuted`, or `streamvbyte` still load, since the `HNSW` and `Graph` structs are byte-for-byte the same as before. Rebuild any `fixed-degree` index.

The same non-self-describing format is why `load` cannot infer `graph_type` (or `metric`) and why passing the wrong one fails inside the decoder. The Python `load` wraps that failure with a message naming both as the likely cause.

## See Also

- [Binaries.md](Binaries.md) — full CLI flag reference for `hnsw_build`/`hnsw_search`
- [PythonUsage.md](PythonUsage.md) — full Python API reference
