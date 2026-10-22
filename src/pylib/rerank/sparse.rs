//! `SparseRerankHNSW`: the Python binding of [`crate::rerank::SparseRerankIndex`]. Its arguments
//! mirror the `hnsw_rerank_build_sparse` and `hnsw_rerank_search_sparse` binaries.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::dataset::ScoredVector;
use vectorium::distances::DotProduct;
use vectorium::readers::read_seismic_format;
use vectorium::vector::SparseVectorView;
use vectorium::{Index, PlainSparseDataset};

use crate::rerank::{SparseFirstStage, SparseRerankConfig, SparseRerankIndex};

use super::dense::{io_err, layout, search_params, value_err};
use crate::pylib::common::{
    build_sparse_dataset_from_parts, check_same_len, convert_components_to_u16, offsets_to_usize,
    push_results, sort_queries_by_component, validate_offsets,
};
use crate::pylib::hnsw::{parse_build_graph_type, parse_graph_type};

/// A sparse two-stage index built from one collection; see `docs/PythonUsage.md` for the build
/// options and the configurations they give.
#[pyclass]
pub struct SparseRerankHNSW {
    inner: SparseRerankIndex,
}

fn parse_first_stage(value_type: &str) -> PyResult<SparseFirstStage> {
    match value_type.to_lowercase().as_str() {
        "f16" => Ok(SparseFirstStage::F16),
        "fixedu8" | "u8" => Ok(SparseFirstStage::FixedU8),
        "blocked-sparse" | "blocked" => Ok(SparseFirstStage::Blocked),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown first_stage_value_type {other:?}; choose \"blocked-sparse\" (the \
             default), \"f16\" or \"fixedu8\"."
        ))),
    }
}

fn build(
    collection: PlainSparseDataset<u16, f32, DotProduct>,
    m: usize,
    ef_construction: usize,
    first_stage_value_type: &str,
    graph_type: &str,
    prune_percentage: f32,
    residual_rerank: bool,
) -> PyResult<SparseRerankHNSW> {
    let cfg = SparseRerankConfig {
        m,
        ef_construction,
        graph: layout(parse_build_graph_type(graph_type, m)?),
        first_stage: parse_first_stage(first_stage_value_type)?,
        prune_percentage,
        residual_rerank,
    };
    let inner = SparseRerankIndex::build(collection, &cfg).map_err(value_err)?;
    Ok(SparseRerankHNSW { inner })
}

#[pymethods]
impl SparseRerankHNSW {
    /// Builds the index from a collection in Seismic binary format.
    ///
    /// # Arguments
    /// * `data_file` – the collection in Seismic binary format.
    /// * `prune_percentage` – in `[0, 100]`; the first stage keeps, per vector, the components
    ///   carrying the top `prune_percentage`% of its L1 mass. 100 means no pruning.
    /// * `m`, `ef_construction` – HNSW build parameters for the first stage.
    /// * `first_stage_value_type` – `"blocked-sparse"` (default; both stages stored blocked, the
    ///   graph built over `f16` values), `"f16"` or `"fixedu8"`.
    /// * `graph_type` – `"standard"` (default), `"permuted"` or `"compressed"` (the latter needs an
    ///   `"f16"` or `"fixedu8"` first stage).
    /// * `residual_rerank` – store only the pruned-away components in the rerank stage and sum
    ///   the two scores. Needs `prune_percentage < 100`. Default: False.
    ///
    /// `prune_percentage=100` with a `"blocked-sparse"` or `"f16"` first stage is rejected, since
    /// the rerank stage would be an identical copy.
    #[staticmethod]
    #[pyo3(signature = (data_file, prune_percentage, m=32, ef_construction=200, first_stage_value_type="blocked-sparse".to_string(), graph_type="standard".to_string(), residual_rerank=false))]
    pub fn build_from_file(
        data_file: &str,
        prune_percentage: f32,
        m: usize,
        ef_construction: usize,
        first_stage_value_type: String,
        graph_type: String,
        residual_rerank: bool,
    ) -> PyResult<Self> {
        let collection: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(data_file)
            .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading dataset: {e:?}"))
        })?;
        build(
            collection,
            m,
            ef_construction,
            &first_stage_value_type,
            &graph_type,
            prune_percentage,
            residual_rerank,
        )
    }

    /// Builds the index from a collection given as flat arrays.
    ///
    /// `components` and `values` hold every vector's non-zeroes back to back, and `offsets` holds
    /// `n + 1` entries delimiting the `n` vectors. The other arguments are as for
    /// [`build_from_file`](Self::build_from_file).
    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, prune_percentage, m=32, ef_construction=200, first_stage_value_type="blocked-sparse".to_string(), graph_type="standard".to_string(), residual_rerank=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        prune_percentage: f32,
        m: usize,
        ef_construction: usize,
        first_stage_value_type: String,
        graph_type: String,
        residual_rerank: bool,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let offsets_vec = offsets_to_usize(offsets.as_slice()?)?;
        let dim = components_vec
            .iter()
            .max()
            .map(|&x| (x as usize) + 1)
            .unwrap_or(0);
        let collection = build_sparse_dataset_from_parts::<f32, DotProduct>(
            components_vec,
            values.as_slice()?.to_vec(),
            offsets_vec,
            dim,
        )?;
        build(
            collection,
            m,
            ef_construction,
            &first_stage_value_type,
            &graph_type,
            prune_percentage,
            residual_rerank,
        )
    }

    /// Bytes held by the first-stage index and the rerank dataset.
    pub fn space_usage_bytes(&self) -> usize {
        self.inner.space_usage_bytes()
    }

    /// Writes the first-stage graph and the rerank dataset to one file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(io_err)
    }

    /// Loads an index written by [`Self::save`].
    ///
    /// Index files carry no header, so `first_stage_value_type`, `graph_type` and
    /// `residual_rerank` must match the build. `"standard"` and `"permuted"` load as the same type.
    #[staticmethod]
    #[pyo3(signature = (path, first_stage_value_type="blocked-sparse".to_string(), graph_type="standard".to_string(), residual_rerank=false))]
    pub fn load(
        path: &str,
        first_stage_value_type: String,
        graph_type: String,
        residual_rerank: bool,
    ) -> PyResult<Self> {
        let inner = SparseRerankIndex::load(
            path,
            parse_first_stage(&first_stage_value_type)?,
            layout(parse_graph_type(&graph_type)?),
            residual_rerank,
        )
        .map_err(io_err)?;
        Ok(SparseRerankHNSW { inner })
    }

    /// Two-stage search for a single sparse query. The same query is used for both stages.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of sparse query component indices.
    /// * `query_values` – 1-D float32 array of sparse query values.
    /// * `k_candidates` – first-stage candidates to rerank. Default: 100.
    /// * `k` – results returned. Default: 10.
    /// * `ef_search` – HNSW candidate list size. Default: 100.
    /// * `alpha` – candidate pruning on the first-stage score (0-1). Default: None.
    /// * `beta` – early exit during reranking. Default: None.
    /// * `early_exit_threshold` – lambda for distance-adaptive early termination. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length `k`, padded with `inf`/`-1`.
    #[pyo3(signature = (query_components, query_values, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        k_candidates: usize,
        k: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let n = comp_vec.len();
        let (comp_vec, values) =
            sort_queries_by_component(comp_vec, query_values.as_slice()?, &[0, n])?;
        let params = search_params(k_candidates, ef_search, alpha, beta, early_exit_threshold);
        let query = SparseVectorView::new(&comp_vec, &values);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        push_results(
            self.inner.search(query, k, &params),
            k,
            &mut distances,
            &mut ids,
        );

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch two-stage search, optionally in parallel.
    ///
    /// `num_threads` is `0` for rayon's default pool, `1` for a serial loop without rayon, or an
    /// explicit thread count.
    ///
    /// `sparse_offsets` delimits the queries: for `N` queries pass `N + 1` offsets starting at
    /// `0` and ending at the total number of non-zeroes.
    #[pyo3(signature = (query_components, query_values, sparse_offsets, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None, num_threads=0))]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        sparse_offsets: PyReadonlyArray1<i64>,
        k_candidates: usize,
        k: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        check_same_len(comp_vec.len(), values_slice.len())?;
        let offsets = offsets_to_usize(sparse_offsets.as_slice()?)?;
        validate_offsets(&offsets, values_slice.len())?;
        let (comp_vec, values) = sort_queries_by_component(comp_vec, values_slice, &offsets)?;
        let values_slice = values.as_slice();
        let params = search_params(k_candidates, ef_search, alpha, beta, early_exit_threshold);
        let num_queries = offsets.len() - 1;

        let search_one = |q_idx: usize| -> Vec<ScoredVector<DotProduct>> {
            let (start, end) = (offsets[q_idx], offsets[q_idx + 1]);
            let query = SparseVectorView::new(&comp_vec[start..end], &values_slice[start..end]);
            self.inner.search(query, k, &params)
        };

        let results: Vec<Vec<ScoredVector<DotProduct>>> = py.detach(|| match num_threads {
            1 => (0..num_queries).map(search_one).collect(),
            0 => (0..num_queries).into_par_iter().map(search_one).collect(),
            n => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("failed to build rayon thread pool")
                .install(|| (0..num_queries).into_par_iter().map(search_one).collect()),
        });

        let mut all_distances = Vec::with_capacity(num_queries * k);
        let mut all_ids = Vec::with_capacity(num_queries * k);
        for r in results {
            push_results(r, k, &mut all_distances, &mut all_ids);
        }

        let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
        let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }

    /// Number of vectors in the index.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }
}
