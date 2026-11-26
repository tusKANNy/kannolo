# HNSW Graph Compression

kANNolo can reorder and compress the HNSW ground-level neighbor lists to shrink the index on disk, with no loss of search accuracy. The `--graph-type` flag (CLI) or the `graph` argument of `kannolo.HNSW` (Python) controls this. It is independent of the dataset, encoder and value-type choices documented in [Binaries.md](Binaries.md) and [PythonUsage.md](PythonUsage.md).

## How It Works

Two independent techniques are combined:

- **EGB (Enhanced Graph Bisection) permutation**: nodes are reassigned new IDs so that nodes with overlapping neighborhoods end up with adjacent IDs. This shrinks the numeric gaps between a node's neighbor IDs, and small gaps make delta-based compression effective.
- **StreamVByte compression**: each node's neighbor list is sorted, delta-encoded (each ID stored as the difference from the previous one), and byte-packed with StreamVByte.

Permutation and compression change only how the graph is *stored*. They never change what a search returns. This was verified end to end. The Rust test suite includes a test that asserts byte-identical search results between the baseline graph and its permuted and compressed versions. Manual CLI and Python runs also gave identical result sets (IDs and distances) across all graph types on the same data.

Because reordering happens internally, search results always report the **original** external IDs. The index stores the inverse permutation and applies it when it returns results, so callers never need to know the internal node order.

## The Four Graph Types

The CLI and the Python API name these differently: the CLI names the *storage format*, and Python names the *effect*. Both run the same four code paths (three in Python).

| CLI `--graph-type` | Python `kannolo.HNSW(graph=...)` | Node order | Neighbor storage | Reduces size? |
|---|---|---|---|---|
| `standard` | `"standard"` | Original (insertion order) | Plain, uncompressed | — (baseline) |
| `fixed-degree` | *not exposed* | Original (insertion order) | Fixed-width padded | No (usually larger) |
| `permuted` | `"permuted"` | EGB-reordered | Plain, uncompressed | No — slightly larger than `standard` |
| `streamvbyte` | `"compressed"` | EGB-reordered | Delta + StreamVByte compressed | **Yes** |

All four return identical search results. They differ only in index size and query speed.

`fixed-degree` is a third neighbor-storage backend, next to plain and StreamVByte. It pads every node's list to a common stride, so a list can be located without an offsets array.

The EGB reordering is useful on its own. It groups nodes that are traversed together, so a beam search touches fewer cache lines and **queries get faster**. `permuted` gives you this speedup. Any reordered index also stores a ground-level inverse permutation, so that results come back as original dataset IDs. The permutation costs about `ceil(log2(n))` bits per vector, so a `permuted` index is marginally *larger* than a `standard` one: it improves query speed, but not size.

`streamvbyte` adds compression on top of that reordering. In local testing, it reduced the graph portion of the index by roughly half compared with `standard`. The exact ratio depends on `M` and on the locality of the neighbor IDs after the permutation. Decoding each node's list adds a small cost, which the locality gain offsets, so a compressed index is not slower to query than `standard`.

## CLI Usage

Both `hnsw_build` and `hnsw_search` accept:

```bash
--graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
```

`--graph-type` must be the same at build and at search time: an index built with `streamvbyte` must also be searched with `--graph-type streamvbyte`.

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

The `graph` argument of `kannolo.HNSW` selects the layout. It works with every representation
and encoder (dense, sparse and multi-vector; plain or compressed vectors). It applies to HNSW
first stages. Flat and IVF indexes have no graph to compress (the HNSW centroid index of an IVF
uses `"standard"`).

**You can leave it out.** Omitting it gives you `"standard"`, the plain uncompressed index:

```python
import kannolo
import numpy as np

data = np.random.rand(10000, 768).astype(np.float32)

# No graph: a standard index.
index = kannolo.build(data, index=kannolo.HNSW())
```

Pass `graph="compressed"` when you want the index to take less space. Nothing else about your code changes: the results are the same and the IDs are still the original dataset IDs:

```python
index = kannolo.build(data, metric="ip", index=kannolo.HNSW(m=32, ef_construction=200, graph="compressed"))

index.save("index.kannolo")

# The file records how the index was built, graph layout included.
loaded = kannolo.load("index.kannolo")

query = np.random.rand(768).astype(np.float32)
distances, ids = loaded.search(query, k=10, ef_search=100)
```

`graph="permuted"` is the third option: it applies the reordering without the compression, so queries get faster but the index does not get smaller.

Python and the CLI use different names: Python's `"compressed"` is the CLI's `streamvbyte`. Python does not expose `fixed-degree`.

## Constraint: Neighbor List Length

StreamVByte blocks support adjacency lists of **at most 256 neighbors per node**. The ground level's max degree is `2 × M`, so `M` must be at most 128 when using `--graph-type streamvbyte` / `graph="compressed"`. The defaults (`M = 16`/`32`) are well within range.

Both front ends check this before building, so that the build does not fail at the end. The CLI exits with an error, and Python raises `ValueError` from the build call.

## Index format breaks

Indexes are encoded with bincode's fixed-int configuration, which is positional and not self-describing. Any change to the fields of a serialized struct breaks compatibility in both directions. Loading then fails with a decode error that does not explain the cause. This has happened twice.

**When permutation was added (0.8.0).** Permuted indexes carry a ground-level inverse permutation, which the serialized `HNSW` did not have before. This change broke every graph type, including those that are not permuted.

**When fixed-degree became a backend (0.10.0).** `GraphFixedDegree` used to be its own type, with its own storage and its own ID mapping. It is now `Graph<FixedDegreeNeighbors>`, so its serialized layout is the one every other graph type already used. **Only `fixed-degree` indexes are affected.** Files written by 0.9.2 as `standard`, `permuted` or `streamvbyte` still load, because the `HNSW` and `Graph` structs are byte-for-byte the same as before. Rebuild any `fixed-degree` index.

The same format is not self-describing, so the CLI search binaries must be given the `--graph-type` (and `--distance`) an index was built with. Passing the wrong one fails inside the decoder. Files written by the Python `Index.save` start with a header that records how the index was built, so `kannolo.load` needs no such arguments.

## See Also

- [Binaries.md](Binaries.md): full CLI flag reference for `hnsw_build`/`hnsw_search`
- [PythonUsage.md](PythonUsage.md): full Python API reference
