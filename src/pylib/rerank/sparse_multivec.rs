//! `SparseMultivecRerankIndex`: sparse HNSW base index reranked with a plain
//! multivector dataset.

use std::f32;

use crate::graph::Graph;
use crate::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use half::f16;
use vectorium::IndexSerializer;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::rerank_index::RerankIndex;
use vectorium::PlainSparseDataset;
use vectorium::distances::DotProduct;
use vectorium::vector::DenseMultiVectorView;
use vectorium::vector::SparseVectorView;
use vectorium::{MultiVectorDataset, PlainMultiVecQuantizer};

use crate::pylib::common::{
    convert_components_to_u16, load_index_err, push_results, save_index_err,
};

fn load_multivec_dataset_plain(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    let documents_path = Path::new(data_folder).join("documents.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");

    let documents_file = File::open(&documents_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening documents file at {:?}: {}",
            documents_path, e
        ))
    })?;
    let documents_u16: Array2<u16> =
        Array2::read_npy(BufReader::new(documents_file)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error reading documents array: {}",
                e
            ))
        })?;

    let (_n_tokens, token_dim) = documents_u16.dim();
    let documents_raw = documents_u16.into_raw_vec_and_offset().0;
    let mut documents_flat: Vec<f32> = Vec::with_capacity(documents_raw.len());
    for u16_val in documents_raw {
        let f16_val = f16::from_bits(u16_val);
        documents_flat.push(f32::from(f16_val));
    }

    let doclens_file = File::open(&doclens_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening doclens file at {:?}: {}",
            doclens_path, e
        ))
    })?;
    let doclens_array: ndarray::Array1<i32> =
        ndarray::Array1::read_npy(BufReader::new(doclens_file)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error reading doclens array: {}",
                e
            ))
        })?;

    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Build offsets array from doclens
    let mut offsets = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * token_dim);
    }

    let encoder = PlainMultiVecQuantizer::<f32>::new(token_dim);
    Ok(MultiVectorDataset::from_raw(
        documents_flat.into(),
        offsets.into(),
        encoder,
    ))
}

#[pyclass]
pub struct SparseMultivecRerankIndex {
    inner: RerankIndex<
        HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
        MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
    >,
}

#[pymethods]
impl SparseMultivecRerankIndex {
    /// Build a rerank index from a pre-built sparse HNSW index and multivector data folder.
    ///
    /// # Arguments
    /// * `sparse_index_path` – Path to the pre-built sparse HNSW index file.
    /// * `multivec_data_folder` – Path to folder containing multivector data files (plain quantizer).
    ///
    /// # Multivector Data Folder Structure (Plain Quantizer)
    /// The folder must contain the following files:
    /// * `documents.npy` – Token embeddings (shape: [n_tokens, token_dim], dtype: uint16,
    ///   reinterpreted as f16). Note this is 2-D and token-major: documents are delimited by
    ///   `doclens`, not by an array dimension.
    /// * `doclens.npy` – Document lengths (shape: [n_documents], dtype: int32)
    ///
    #[staticmethod]
    #[pyo3(signature = (sparse_index_path, multivec_data_folder))]
    pub fn build_from_file(sparse_index_path: &str, multivec_data_folder: &str) -> PyResult<Self> {
        let sparse_index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> =
            <HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> as IndexSerializer>::load_index(
                sparse_index_path,
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Error loading sparse index: {:?}",
                    e
                ))
            })?;

        let multivec_dataset = load_multivec_dataset_plain(multivec_data_folder)?;

        Ok(SparseMultivecRerankIndex {
            inner: RerankIndex::new(sparse_index, multivec_dataset),
        })
    }

    /// Saves the whole two-stage index — first-stage graph and rerank dataset — to one file.
    ///
    /// `build_from_file` reconstructs the index from a saved first-stage index plus the original
    /// multivector folder; this instead round-trips the index as it stands, so the rerank
    /// dataset does not have to be re-read and re-encoded.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save_index(path).map_err(save_index_err)
    }

    /// Loads an index written by [`Self::save`].
    ///
    /// Takes no build arguments: this class has exactly one concrete instantiation.
    #[staticmethod]
    #[pyo3(signature = (path))]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = RerankIndex::load_index(path).map_err(load_index_err)?;

        Ok(SparseMultivecRerankIndex { inner })
    }

    /// Search with reranking using plain multivector encoding (single query).
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of sparse query component indices.
    /// * `query_values` – 1-D float32 array of sparse query values.
    /// * `multivec_query` – 1-D float32 array of the multivector query (n_tokens × token_dim).
    /// * `n_tokens` – Number of tokens in the multivector query.
    /// * `token_dim` – Dimension of each token.
    /// * `k_candidates` – Number of candidates to retrieve in first stage. Default: 100.
    /// * `k` – Number of final results to return. Default: 10.
    /// * `ef_search` – Candidate list size for HNSW search. Default: 100.
    /// * `alpha` – Alpha parameter for candidate pruning (0-1). Default: None.
    /// * `beta` – Beta parameter for early exit. Default: None.
    /// * `early_exit_threshold` – Lambda for early termination. Default: None.
    /// * `residuals` – Score candidates by the sum of first-stage and rerank scores instead of the
    ///   rerank score alone. Only meaningful when the two scores are summable (e.g. the rerank
    ///   dataset holds the residual part of a decomposed representation). Default: False.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query_components, query_values, multivec_query, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None, residuals=false))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        multivec_query: PyReadonlyArray1<f32>,
        n_tokens: usize,
        token_dim: usize,
        k_candidates: usize,
        k: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
        residuals: bool,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let query_values_slice = query_values.as_slice()?;
        let multivec_query_slice = multivec_query.as_slice()?;

        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let sparse_query = SparseVectorView::new(&comp_vec, query_values_slice);
        let multivec_query_view = DenseMultiVectorView::new(multivec_query_slice, token_dim);
        let _ = n_tokens; // token count is implicit from slice length / token_dim

        let results = self.inner.search(
            sparse_query,
            multivec_query_view,
            k_candidates,
            k,
            &search_config,
            &(),
            alpha,
            beta,
            residuals,
        );

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        push_results(results, k, &mut distances, &mut ids);

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch search with reranking using plain multivector encoding, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of sparse query component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of sparse query values (concatenated for batch).
    /// * `sparse_offsets` – 1-D int64 array defining sparse query boundaries. For N queries, pass [0, n1, n1+n2, ..., total].
    /// * `multivec_queries` – 1-D float32 array of all multivector queries concatenated (total_queries × n_tokens × token_dim).
    /// * `n_tokens` – Number of tokens per multivector query (fixed).
    /// * `token_dim` – Dimension of each token in the multivector queries.
    /// * `k_candidates` – Number of candidates to retrieve in first stage. Default: 100.
    /// * `k` – Number of final results to return per query. Default: 10.
    /// * `ef_search` – Candidate list size for HNSW search. Default: 100.
    /// * `alpha` – Alpha parameter for candidate pruning (0-1). Default: None.
    /// * `beta` – Beta parameter for early exit. Default: None.
    /// * `early_exit_threshold` – Lambda for early termination. Default: None.
    /// * `num_threads` – Threading model (see above). Default: 0 (all cores).
    /// * `residuals` – Score candidates by the sum of first-stage and rerank scores instead of the
    ///   rerank score alone. Only meaningful when the two scores are summable (e.g. the rerank
    ///   dataset holds the residual part of a decomposed representation). Default: False.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, sparse_offsets, multivec_queries, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None, num_threads=0, residuals=false))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        sparse_offsets: PyReadonlyArray1<i64>,
        multivec_queries: PyReadonlyArray1<f32>,
        n_tokens: usize,
        token_dim: usize,
        k_candidates: usize,
        k: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
        residuals: bool,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let query_values_slice = query_values.as_slice()?;
        let sparse_offsets_slice = sparse_offsets.as_slice()?;
        let multivec_queries_slice = multivec_queries.as_slice()?;

        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let num_queries = sparse_offsets_slice.len() - 1;
        let multivec_query_size = n_tokens * token_dim;

        let search_one = |q_idx: usize| -> (Vec<f32>, Vec<i64>) {
            let sparse_start = sparse_offsets_slice[q_idx] as usize;
            let sparse_end = sparse_offsets_slice[q_idx + 1] as usize;
            let sparse_query = SparseVectorView::new(
                &comp_vec[sparse_start..sparse_end],
                &query_values_slice[sparse_start..sparse_end],
            );
            let multivec_start = q_idx * multivec_query_size;
            let multivec_query_view = DenseMultiVectorView::new(
                &multivec_queries_slice[multivec_start..multivec_start + multivec_query_size],
                token_dim,
            );
            let results = self.inner.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            );
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            push_results(results, k, &mut distances, &mut ids);
            (distances, ids)
        };

        let results: Vec<(Vec<f32>, Vec<i64>)> = py.detach(|| match num_threads {
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
        for (d, i) in results {
            all_distances.extend(d);
            all_ids.extend(i);
        }

        let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
        let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }
}
