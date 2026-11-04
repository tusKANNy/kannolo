<h1>
    <img height="72" align="center" alt="" src="https://silviom97.github.io/assets/img/logos/kannolo.webp" />
    &nbsp;&nbsp;&nbsp;kANNolo
</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2611.03927"><img src="https://badgen.net/static/arXiv/2611.03927/red" /></a>
    <a href="https://arxiv.org/abs/2501.06121"><img src="https://badgen.net/static/paper/ECIR 2025/yellow" /></a>
</p>

<p align="center">
    <!-- crates.io is stale (0.3.8) until the next crates.io release; restore these then.
    <a href="https://crates.io/crates/kannolo"><img src="https://badgen.infra.medigy.com/crates/v/kannolo" /></a>
    <a href="https://crates.io/crates/kannolo"><img src="https://badgen.infra.medigy.com/crates/d/kannolo" /></a>
    -->
    <a href="https://pypi.org/project/kannolo/"><img src="https://badgen.net/pypi/v/kannolo" /></a>
    <a href="LICENSE"><img src="https://badgen.net/static/license/MIT/blue" /></a>
</p>

**Fast, compact approximate nearest neighbor indexes for dense, sparse and multi-vector embeddings, in Rust 🦀 with Python bindings.**

kANNolo provides **HNSW**, **IVF** and **two-stage (retrieve + rerank)** indexes. Every index is
generic over how the vectors are stored, so the same index serves `f32`, `f16`, Product
Quantization, RaBitQ, learned-sparse and compressed-sparse data. The compiler specializes each
combination, so there is no dynamic dispatch in the search loop. The storage, compression and
distance layer is a separate library, [Vectorium](https://github.com/TusKANNy/vectorium).

```bash
pip install kannolo
```

## Why kANNolo

- **Faster and smaller than the reference implementations.** On dense data:
  - Our HNSW is **1.6–2.8× faster than [hnswlib](https://github.com/nmslib/hnswlib)**, with an **index 2.0–2.2× smaller** and comparable build times, on four collections of 0.8M to 21M vectors.
  - On the same collections, our two-stage HNSW is **1.8–3.2× faster than hnswlib**, with an **index 1.7–2.0× smaller**.
  - Our IVF indexes answer queries **1.2–4.1× faster than [FAISS](https://github.com/facebookresearch/faiss)** at the same accuracy, on two collections.

  On learned sparse data:
  - Our HNSW is **3.2–20.3× faster than [PyANNs](https://github.com/hhy3/pyanns)** (winner of the Sparse Track of the NeurIPS 2023 Big-ANN challenge), with **3.1–3.7× less memory**.
  - Our two-stage HNSW is **5.2–47.9× faster than PyANNs**, with **2.9–3.3× less memory**.

  kANNolo's sparse indexes are the most compact of all the state-of-the-art systems we tested, and the fastest graph indexes.
- **Dense, sparse and multi-vector through one API.** Search any of the three representations, and
  combine them freely into two-stage pipelines with `RerankIndex`: for example, a sparse first stage
  reranked with multi-vector MaxSim.
- **Easy to use.** Python bindings, a Rust library, command-line tools, documentation, examples and
  notebooks. If something is unclear, open an issue.

<sub>Numbers from our paper [*Modular and Efficient Approximate Nearest Neighbors Search with Vectorium and kANNolo*](https://arxiv.org/abs/2611.03927) (kANNolo 0.15.1, Vectorium 0.11.1), at 90–99% Accuracy@10. See [Reproducing our papers](#reproducing-our-papers).</sub>

**When to use something else.** On learned sparse retrieval, [Seismic](https://github.com/TusKANNy/seismic),
an inverted index also built on Vectorium, is faster than any graph index, ours included. For
multi-vector data, kANNolo works best as the reranking step of a two-stage pipeline with a
single-vector first stage. For a dedicated multi-vector engine, see [TACHIOM](https://github.com/TusKANNy/tachiom).
kANNolo's indexes are in-memory and static, built once over a fixed collection: if you need
disk-based indexes, or insertions and
deletions after construction, look elsewhere for now.

## Quick start (Python)

Dense, two-stage: an HNSW over 1-bit RaBitQ codes retrieves 100 candidates, and the `f16` vectors rescore them.

```python
import numpy as np
from kannolo import DenseRerankHNSW

data = np.random.randn(100_000, 768).astype(np.float32)
index = DenseRerankHNSW.build_from_array(
    data.flatten(), dim=768, metric="dotproduct",
    m=32, ef_construction=200,
    encoder="rabitq",            # or "pq", "rabitq-ext"
)

query = data[42]
dists, ids = index.search(query, k=10, k_candidates=100, ef_search=100)
index.save("dense.idx")
```

Sparse: the first stage keeps the components that carry the top 60% of each vector's L1 mass, and
the pruned-away residual restores the exact score when a candidate is rescored.

```python
from kannolo import SparseRerankHNSW

index = SparseRerankHNSW.build_from_file(
    "documents.bin",             # Seismic binary format
    prune_percentage=60.0, residual_rerank=True,
    m=32, ef_construction=500,
)
dists, ids = index.search(query_components, query_values, k=10, k_candidates=100, ef_search=100)
```

Single-stage indexes (`DensePlainHNSW`, `DensePQHNSW`, `SparsePlainHNSW`, …), flat indexes,
batch search, graph compression and early termination are covered in
[`docs/PythonUsage.md`](docs/PythonUsage.md).

## What's inside

| | |
|---|---|
| **Indexes** | HNSW · IVF (flat, HNSW or IVF centroid index; plain or residual clusters) · `RerankIndex` (any first stage + any rescoring dataset) · exact `FlatIndex` |
| **Representations** (from Vectorium) | dense · sparse · multi-vector (as a rescoring stage) |
| **Encodings** (from Vectorium) | `f32`, `f16`, `bf16`, 8/16-bit fixed point · Product Quantization · RaBitQ (1 bit) and extended RaBitQ (2/4/8 bits) · blocked sparse · DotPacking8 compressed sparse · multi-vector PQ and two-level PQ |
| **Metrics** | inner product · squared Euclidean · MaxSim |
| **Graph storage** | `standard` · `permuted` (cache-friendly node order) · `compressed` (StreamVByte + Elias–Fano; roughly halves the edges memory) |
| **Interfaces** | Rust library · Python bindings · command-line build/search tools |

## Rust

```toml
[dependencies]
kannolo   = { git = "https://github.com/TusKANNy/kannolo.git",   tag = "v0.15.1" }
vectorium = { git = "https://github.com/TusKANNy/vectorium.git", tag = "v0.11.1" }
```

```rust
use kannolo::rerank::{DenseEncoder, DenseRerankConfig, DenseRerankIndex, RerankSearchParams};
use vectorium::distances::DotProduct;
use vectorium::vector::DenseVectorView;
use vectorium::Index;

// RaBitQ-ext 4-bit first stage, reranked on 8-bit RaBitQ-ext codes.
let (dim, data): (usize, Vec<f32>) = (768, vec![0.0; 768 * 1000]);
let mut cfg = DenseRerankConfig::new(DenseEncoder::RabitqExt { total_bits: 4 });
cfg.rerank_bits = 8;
let index = DenseRerankIndex::<DotProduct>::build(&data, dim, &cfg).unwrap();
let top10 = index.search(DenseVectorView::new(&data[..dim]), 10, &RerankSearchParams::new(100, 100));
```

The lower-level API composes the parts yourself: build an `HNSW` over any Vectorium dataset, swap
its dataset for a compressed one, and wrap it in a `RerankIndex`:

```rust
type Rescoring = DenseDataset<PlainDenseQuantizer<f16, DotProduct>>;
type Codes = DenseDataset<RabitqQuantizer<DotProduct>>;

let graph: HNSW<Rescoring, Graph> = HNSW::build_index(exact.clone(), &build_config);
let first: HNSW<Codes, Graph> = graph
    .permute_and_encode::<PlainNeighbors>()               // cache-friendly node order
    .convert_dataset_into_ref(RabitqConfig::default());   // vectors -> 1-bit codes
let index = RerankIndex::new(first, exact);               // rescore on f16
```

Replacing `Codes` with a PQ or extended-RaBitQ dataset gives another index with no other change.
The module docs of `kannolo::rerank` and `kannolo::rerank_index` have complete examples.

**Always build with `RUSTFLAGS="-C target-cpu=native"`.** The distance kernels are plain Rust that the
compiler vectorizes for your CPU, and without this flag they fall back to baseline SIMD.

| Feature | What it enables | Default |
|---------|-----------------|:-------:|
| `cli` | Binaries: `hnsw_build`, `hnsw_search`, `hnsw_rerank_{build,search}_{dense,sparse}`, `ivf_build`, `ivf_search` | No |
| `multivec` | Multi-vector reranking (`SparseMultivecRerankIndex`, `SparseMultivecTwoLevelsPQRerankIndex`); with `cli`, also `hnsw_rerank_search` | No |
| `python` | PyO3 bindings, enabled automatically by maturin | No |

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release                          # library only
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli           # + binaries
RUSTFLAGS="-C target-cpu=native" cargo build --release --features "cli,multivec" # + multi-vector
```

Binaries land in `target/release/`. [`docs/Binaries.md`](docs/Binaries.md) documents every flag.

## Installing the Python package

`pip install kannolo` installs a prebuilt wheel with every index: dense, sparse, two-stage and
sparse + multi-vector reranking. If no wheel matches your platform, pip compiles from source.

**For maximum performance, build from source for your CPU** (requires Rust nightly via
[`rustup`](https://rustup.rs); the repository pins the toolchain):

```bash
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: kannolo
```

**Development build** (editable install from a clone):

```bash
git clone https://github.com/TusKANNy/kannolo.git && cd kannolo
pip install maturin
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

## Documentation

- [`docs/PythonUsage.md`](docs/PythonUsage.md): the Python API
- [`docs/RustUsage.md`](docs/RustUsage.md) and [`docs/Binaries.md`](docs/Binaries.md): the command-line tools
- [`docs/GraphCompression.md`](docs/GraphCompression.md): graph layouts
- [`docs/MultiVectorUsage.md`](docs/MultiVectorUsage.md): sparse + multi-vector pipelines
- [`docs/RunExperiments.md`](docs/RunExperiments.md): the TOML-driven experiment runner

**Data formats:** dense collections are `.npy` (`float32`), sparse collections use the Seismic binary
format. [`scripts/`](scripts/) converts between the two.

## Reproducing our papers

kANNolo keeps evolving, so this repository does not track the exact setup of any paper. The
versions, configurations and instructions that reproduce the results of our papers are in
[**papers_reproducibility**](https://github.com/tusKANNy/papers_reproducibility).


## 📚 Bibliography

1. Silvio Martinico, Franco Maria Nardini, Cosimo Rulli and Rossano Venturini. "*Modular and Efficient Approximate Nearest Neighbors Search with Vectorium and kANNolo*." arXiv:2611.03927. 2026. Submitted to *Information Systems*.
2. Leonardo Delfino, Domenico Erriquez, Silvio Martinico, Franco Maria Nardini, Cosimo Rulli and Rossano Venturini. "*kANNolo: Sweet and Smooth Approximate k-Nearest Neighbors Search*." Proc. ECIR. 2025.

### Citation
If you use kANNolo for your work (whether be it a research paper, dissertation, article, poster, presentation, or documentation), please cite our papers:

arXiv 2026 (submitted to *Information Systems*):

```bibtex
@misc{martinico2026modularefficientapproximatenearest,
      title={Modular and Efficient Approximate Nearest Neighbors Search with Vectorium and kANNolo},
      author={Silvio Martinico and Franco Maria Nardini and Cosimo Rulli and Rossano Venturini},
      year={2026},
      eprint={2611.03927},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2611.03927},
}
```

ECIR 2025:

```bibtex
@InProceedings{10.1007/978-3-031-88717-8_29,
author =    "Leonardo Delfino and
             Domenico Erriquez and
             Silvio Martinico and
             Franco Maria Nardini and
             Cosimo Rulli and
             Rossano Venturini",
title =     "kANNolo: Sweet and Smooth Approximate k-Nearest Neighbors Search",
booktitle = "Advances in Information Retrieval",
year =      "2025",
publisher = "Springer Nature Switzerland",
pages =     "400--406",
isbn =      "978-3-031-88717-8"
}
```
