## Replicate Results

We provide a quick way to replicate the results of our paper. 

Use the [`scripts/run_experiments.py`](../scripts/run_experiments.py) script to quickly reproduce a result from the paper. 
This script is configurable via TOML files, which specify the parameters to build the index and execute queries on it.  
The script measures average query time (in microseconds), recall with respect to the true closest vectors of the query (accuracy@k), MRR or other metrics with respect to judged qrels if specified, and index space usage (bytes).

The runner drives `hnsw_build`/`hnsw_search`, the IVF binaries (`ivf_build`/`ivf_search`, selected by naming them in `build-command`/`query-command`), and `hnsw_rerank_search` for two-stage reranking. What is *not* supported is `dataset-type = "multivector"`: that path was removed and the runner raises an error if a config still uses it.

TOML files to reproduce the experiments of our paper can be found in [`experiments/ecir2025`](../experiments/ecir2025), and the graph-compression experiments in [`experiments/compressed_graph`](../experiments/compressed_graph).

Datasets can be found at [`Hugging Face`](https://huggingface.co/collections/tuskanny/kannolo-datasets-67f2527781f4f7a1b4c9fe54).

As an example, let's now run the experiments using the TOML file [`experiments/ecir2025/dense_sift1m.toml`](experiments/ecir2025/dense_sift1m.toml), which replicates the results of kANNolo on the SIFT1M dataset.

### <a name="bin_data">Setting up for the Experiment</a>
Let's start by creating a working directory for the data and indexes.

```bash
mkdir -p ~/knn_datasets/dense_datasets/sift1M
mkdir -p ~/knn_indexes/dense_datasets/sift1M
```

We need to download datasets, queries, ground truth (and, eventually, qrels and query IDs) as follows. Here, we are downloading SIFT1M vectors.  

```bash
cd ~/knn_datasets/dense_datasets/sift1M
wget https://huggingface.co/datasets/tuskanny/kannolo-sift1M/resolve/main/dataset.npy
wget https://huggingface.co/datasets/tuskanny/kannolo-sift1M/resolve/main/groundtruth.npy
wget https://huggingface.co/datasets/tuskanny/kannolo-sift1M/resolve/main/queries.npy

```


### Running the Experiment
We are now ready to run the experiment.

First, clone the kANNolo Git repository and compile kANNolo:

```bash
cd ~
git clone git@github.com:TusKANNy/kannolo.git
cd kannolo
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

If needed, install Rust on your machine with the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Now we can run the experiment with the following command:

```bash
python3 scripts/run_experiments.py --exp experiments/ecir2025/dense_sift1m.toml
```

Please install the required Python's libraries with the following command:
```bash
pip install -r scripts/requirements.txt
```

The script will build an index using the unified binary parameters specified at the top level of the TOML file (`build-command`, `dataset-type`, `value-type`, optional sparse `component-type`, `encoder`, `graph-type`) and the traditional indexing parameters in the `[indexing_parameters]` section (`m`, `ef-construction`, `metric`).  
The index is saved in the directory `~/knn_indexes/dense_datasets/sift1M`.  
You can change directory names by modifying the `[folder]` section in the TOML file.

Next, the script will query the same index with different parameters, as specified in the `[query]` section.  
These parameters provide different trade-offs between query time and accuracy.

**Important**: if your machine is NUMA, the NUMA setting in the TOML file should be UNcommented and should be configured according to your hardware for better performance. 

## TOML Configuration Structure

The TOML configuration files have been updated to work with the unified binaries. Here's the structure:

### Top-level Parameters
- `build-command`: Path to the unified build binary (e.g., `"./target/release/hnsw_build"`)
- `query-command`: Path to the unified search binary (e.g., `"./target/release/hnsw_search"`)
- `dataset-type`: Type of vectors - `"dense"` or `"sparse"` (multivector is no longer supported)
- `value-type`: Value type - `"f32"` or `"f16"` for dense; for sparse also `"fixedu8"`, `"fixedu16"`, `"scalaru8"`, `"scalaru16"`, the blocked and the DotPacking8 types (e.g. `"dotpacking8-fixedu8"`, which replaces the removed `"dotvbyte"` encoder). See [Binaries.md](Binaries.md#sparse-value-types). Ignored for `encoder = "pq"`.
- `component-type`: Sparse-only component type - `"u16"` or `"u32"` (blocked and most DotPacking8 value types require `"u16"`)
- `encoder`: Encoder type - `"plain"`, `"pq"`, `"rabitq"` or `"rabitq-ext"` (all but `plain` are dense-only)
- `graph-type`: Graph type - `"standard"`, `"fixed-degree"`, `"permuted"`, or `"streamvbyte"`. `"permuted"` reorders the graph for faster queries at the same size; `"streamvbyte"` also compresses it, requires `m <= 128`, and roughly halves the graph portion of the index. All produce identical results. See [GraphCompression.md](GraphCompression.md)

`graph-type` may also be set **per query subsection**, overriding the top-level value. Because
graph compression is applied at build time and must match at search time, the runner builds one
index per distinct graph type used in the file and points each subsection at the matching index.
This lets a single experiment compare compressed and uncompressed graphs side by side:

```toml
graph-type = "standard"   # default for subsections that do not override it

[query]
    [query.recall_90_standard]
    ef-search = 11
    graph-type = "standard"

    [query.recall_90_streamvbyte]
    ef-search = 11
    graph-type = "streamvbyte"
```

See `experiments/compressed_graph/dense_sift1m.toml` for a complete example.

### Sections
- `[indexing_parameters]`: Traditional HNSW parameters (`m`, `ef-construction`, `metric`)
- `[pq_parameters]`: PQ-specific parameters (`pq-subspaces`, `nbits`, `sample-size`) when using PQ encoder. Supported `pq-subspaces` values are `4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256`, and the value must divide the vector dimensionality. `nbits` and `sample-size` are accepted for compatibility but ignored by the current vectorium PQ implementation.
- `[folder]`: Directory paths for data, indexes, and experiments
- `[filename]`: Filenames for dataset, queries, groundtruth, etc.
- `[settings]`: Runtime settings (k, NUMA, build flag, evaluation metric)
- `[query]`: Different ef-search values for query experiments

### Example TOML Structure

Here's an example of the complete TOML structure for a dense PQ experiment:

```toml
name = "example_hnsw_pq"
title = "Example HNSW PQ Experiment"
description = "Example experiment with Product Quantization"
dataset = "Example Dataset"
build-command = "./target/release/hnsw_build"
query-command = "./target/release/hnsw_search"
dataset-type = "dense"
value-type = "f32"
encoder = "pq"
graph-type = "standard"

[settings]
k = 10
num-runs = 1
NUMA = "numactl --physcpubind='0-15' --localalloc"
build = true
metric = ""

[folder]
data = "~/knn_datasets/dense_datasets/example"
index = "~/knn_indexes/dense_datasets/example"
experiment = "."

[filename]
dataset = "dataset.npy"
queries = "queries.npy"
groundtruth = "groundtruth.npy"
index = "example_index"

[indexing_parameters]
m = 16
ef-construction = 150
metric = "dotproduct"

[pq_parameters]  # Only needed when encoder = "pq"
pq-subspaces = 64
nbits = 8
sample-size = 100000

[query]
    [query.efs_40]
    ef-search = 40
    [query.efs_80]
    ef-search = 80
``` 

### Getting the Results
The script creates a folder named `<name>_<timestamp>`, where `<name>` is the top-level `name` field of the TOML and `<timestamp>` is `YYYY-MM-DD_HH:MM:SS`. This ensures that each run creates a unique directory. It is written under `[folder].experiment`, which defaults to the repository root in the shipped configs.

These directories are untracked scratch output and are **not** in `.gitignore`, so `git status` will be full of them. Never use `git add -A` or `git commit -a` in this repository; stage files explicitly.

Inside the folder, you can find the data collected during the experiment.

The most important file is `report.tsv`, with one row per query subsection and, in column order:
*query time*, *bits/edge*, *accuracy*, the optional `metric` column, *memory usage* and *building
time*. The building time reported is that
of the index the subsection actually searched, so subsections using different graph types show
their own build cost. Build logs are written to one `building_<graph-type>.output` file per index.
