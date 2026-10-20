//! `SparseMultivecTwoLevelsPQRerankIndex`: sparse HNSW base index reranked with a
//! two-levels PQ multivector dataset.

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
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::vector::DenseMultiVectorView;
use vectorium::vector::DenseVectorView;
use vectorium::vector::SparseVectorView;
use vectorium::{
    DatasetGrowable, DenseDataset, MultiVecTwoLevelProductQuantizer, MultiVectorDataset,
    PlainDenseDatasetGrowable, PlainDenseQuantizer,
};

use crate::pylib::common::{
    convert_components_to_u16, load_index_err, push_results, save_index_err,
};

/// The rerank dataset for one PQ subspace count: tokens stay two-level PQ encoded, exactly as
/// `hnsw_rerank_search --multivector-quantizer two-levels` keeps them.
type PqMultivecDataset<const M: usize> =
    MultiVectorDataset<MultiVecTwoLevelProductQuantizer<M, f16>>;

/// A two-stage index over one PQ subspace count.
type PqRerankIndex<const M: usize> =
    RerankIndex<HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>, PqMultivecDataset<M>>;

const KSUB: usize = 256;
/// Bytes of coarse centroid id per encoded token.
const COARSE_ID_BYTES: usize = 4;
/// Bytes of residual norm per encoded token, when `residual_norms.npy` is present.
const NORM_BYTES: usize = 2;

fn io_err(msg: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(msg)
}

fn value_err(msg: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
}

/// Load the two-level PQ multivector dataset *in its encoded form*.
///
/// Each token stays `COARSE_ID_BYTES + M` bytes (plus `NORM_BYTES` when residual norms are
/// present) — the same blocked payload the CLI builds — rather than being reconstructed into a
/// `token_dim`-wide f32 vector. `MultiVecTwoLevelPQQueryEvaluator` scores straight off those
/// codes via ADC, so reconstructing was only ever a memory cost.
fn load_multivec_dataset_pq_generic<const M: usize>(
    data_folder: &str,
) -> PyResult<PqMultivecDataset<M>> {
    use ndarray::Array1;
    use ndarray_npy::ReadNpyExt;
    use std::path::Path;

    let coarse_path = Path::new(data_folder).join("centroids.npy");
    let pq_centroids_path = Path::new(data_folder).join("pq_centroids.npy");
    let residuals_path = Path::new(data_folder).join("residuals.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");
    let assignment_path = Path::new(data_folder).join("index_assignment.npy");
    let norms_path = Path::new(data_folder).join("residual_norms.npy");

    // Load coarse centroids (n_centroids, dim) to determine token_dim
    let coarse_file = std::fs::File::open(&coarse_path).map_err(|e| {
        io_err(format!(
            "Error opening centroids.npy at {:?}: {}",
            coarse_path, e
        ))
    })?;
    let coarse_reader = std::io::BufReader::new(coarse_file);
    let coarse_array: ndarray::Array2<f32> = ndarray::Array2::read_npy(coarse_reader)
        .map_err(|e| io_err(format!("Error reading centroids.npy: {}", e)))?;
    let (n_coarse, token_dim) = coarse_array.dim();

    if token_dim % M != 0 {
        return Err(value_err(format!(
            "token_dim {} is not divisible by M={} for two-level PQ",
            token_dim, M
        )));
    }
    let dsub = token_dim / M;

    let mut coarse_growable =
        PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
            PlainDenseQuantizer::new(token_dim),
            n_coarse,
        );
    for row in coarse_array.rows() {
        // read_npy yields a C-order array, so each row is contiguous.
        coarse_growable.push(DenseVectorView::new(row.as_slice().unwrap()));
    }
    let coarse_centroids: DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>> =
        coarse_growable.into();

    // Load PQ centroids
    let pq_file = std::fs::File::open(&pq_centroids_path).map_err(|e| {
        io_err(format!(
            "Error opening pq_centroids.npy at {:?}: {}",
            pq_centroids_path, e
        ))
    })?;
    let pq_reader = std::io::BufReader::new(pq_file);
    let pq_array: Array1<f32> = Array1::read_npy(pq_reader)
        .map_err(|e| io_err(format!("Error reading pq_centroids.npy: {}", e)))?;
    let pq_flat = pq_array.to_vec();

    let expected_pq_len = M * KSUB * dsub;
    if pq_flat.len() != expected_pq_len {
        return Err(value_err(format!(
            "pq_centroids.npy size mismatch: got {}, expected {}",
            pq_flat.len(),
            expected_pq_len
        )));
    }

    let mut pq_centroids: Vec<DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>> =
        Vec::with_capacity(M);
    for m in 0..M {
        let mut subspace =
            PlainDenseDatasetGrowable::with_capacity(PlainDenseQuantizer::new(dsub), KSUB);
        for code in 0..KSUB {
            let offset = m * KSUB * dsub + code * dsub;
            subspace.push(DenseVectorView::new(&pq_flat[offset..offset + dsub]));
        }
        pq_centroids.push(subspace.into());
    }

    // Load doclens
    let doclens_file = std::fs::File::open(&doclens_path).map_err(|e| {
        io_err(format!(
            "Error opening doclens.npy at {:?}: {}",
            doclens_path, e
        ))
    })?;
    let doclens_reader = std::io::BufReader::new(doclens_file);
    let doclens_array: Array1<i32> = Array1::read_npy(doclens_reader)
        .map_err(|e| io_err(format!("Error reading doclens.npy: {}", e)))?;
    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Load residuals (PQ codes)
    let residuals_file = std::fs::File::open(&residuals_path).map_err(|e| {
        io_err(format!(
            "Error opening residuals.npy at {:?}: {}",
            residuals_path, e
        ))
    })?;
    let residuals_reader = std::io::BufReader::new(residuals_file);
    let residuals_array: ndarray::Array2<u8> = ndarray::Array2::read_npy(residuals_reader)
        .map_err(|e| io_err(format!("Error reading residuals.npy: {}", e)))?;
    let (n_tokens, m_check) = residuals_array.dim();
    if m_check != M {
        return Err(value_err(format!(
            "residuals.npy has {} subspaces, expected {}",
            m_check, M
        )));
    }

    // Load index assignments
    let assignment_file = std::fs::File::open(&assignment_path).map_err(|e| {
        io_err(format!(
            "Error opening index_assignment.npy at {:?}: {}",
            assignment_path, e
        ))
    })?;
    let assignment_reader = std::io::BufReader::new(assignment_file);
    let assignment_array: Array1<u64> = Array1::read_npy(assignment_reader)
        .map_err(|e| io_err(format!("Error reading index_assignment.npy: {}", e)))?;
    if assignment_array.len() != n_tokens {
        return Err(value_err(format!(
            "assignment_array length {} != n_tokens {}",
            assignment_array.len(),
            n_tokens
        )));
    }

    // Optional residual norms: their presence decides the encoded token layout, so it has to
    // match what the folder was written with.
    let norms_array: Option<Array1<f32>> = if norms_path.exists() {
        let norms_file = std::fs::File::open(&norms_path).map_err(|e| {
            io_err(format!(
                "Error opening residual_norms.npy at {:?}: {}",
                norms_path, e
            ))
        })?;
        let norms_reader = std::io::BufReader::new(norms_file);
        let norms: Array1<f32> = Array1::read_npy(norms_reader)
            .map_err(|e| io_err(format!("Error reading residual_norms.npy: {}", e)))?;
        if norms.len() != n_tokens {
            return Err(value_err(format!(
                "residual_norms.npy length mismatch: got {}, expected {}",
                norms.len(),
                n_tokens
            )));
        }
        Some(norms)
    } else {
        None
    };
    let with_norms = norms_array.is_some();

    let quantizer = MultiVecTwoLevelProductQuantizer::<M, f16>::from_pretrained(
        token_dim,
        coarse_centroids,
        pq_centroids,
        with_norms,
    );

    // Encoded blocked payload per document:
    // [coarse_ids: 4*n][pq_codes: M*n][norms: 2*n (f16) if enabled]
    let bytes_per_token = COARSE_ID_BYTES + M + if with_norms { NORM_BYTES } else { 0 };
    let mut encoded_data: Vec<u8> = Vec::with_capacity(n_tokens * bytes_per_token);

    let mut token_offset = 0usize;
    for &doclen in &doclens {
        if token_offset + doclen > n_tokens {
            return Err(value_err(format!(
                "doclens.npy spans {} tokens, more than the {} in residuals.npy",
                token_offset + doclen,
                n_tokens
            )));
        }
        for i in 0..doclen {
            let coarse_idx = assignment_array[token_offset + i];
            if coarse_idx as usize >= n_coarse {
                return Err(value_err(format!(
                    "coarse_idx {} >= n_coarse {}",
                    coarse_idx, n_coarse
                )));
            }
            encoded_data.extend((coarse_idx as u32).to_le_bytes());
        }
        for i in 0..doclen {
            for subspace_idx in 0..M {
                encoded_data.push(residuals_array[[token_offset + i, subspace_idx]]);
            }
        }
        if let Some(ref norms) = norms_array {
            for i in 0..doclen {
                encoded_data
                    .extend_from_slice(&f16::from_f32(norms[token_offset + i]).to_le_bytes());
            }
        }
        token_offset += doclen;
    }

    if token_offset != n_tokens {
        return Err(value_err(format!(
            "doclens.npy covers {} tokens, expected {}",
            token_offset, n_tokens
        )));
    }

    let mut offsets = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * bytes_per_token);
    }

    Ok(MultiVectorDataset::from_raw(
        encoded_data.into_boxed_slice(),
        offsets.into(),
        quantizer,
    ))
}

// Enum to handle different PQ subspace counts
enum SparseMultivecTwoLevelsPQRerankIndexEnum {
    M8(PqRerankIndex<8>),
    M16(PqRerankIndex<16>),
    M32(PqRerankIndex<32>),
    M64(PqRerankIndex<64>),
}

#[pyclass]
pub struct SparseMultivecTwoLevelsPQRerankIndex {
    inner: SparseMultivecTwoLevelsPQRerankIndexEnum,
}

#[pymethods]
impl SparseMultivecTwoLevelsPQRerankIndex {
    /// Build a rerank index from a pre-built sparse HNSW index and multivector data folder with two-level PQ encoding.
    ///
    /// # Arguments
    /// * `sparse_index_path` – Path to the pre-built sparse HNSW index file.
    /// * `multivec_data_folder` – Path to folder containing multivector data files (two-level PQ quantizer).
    /// * `pq_subspaces` – Number of PQ subspaces (M). Supported values: 8, 16, 32, 64.
    ///
    /// # Multivector Data Folder Structure (Two-Level PQ Quantizer)
    /// The folder must contain the following files:
    /// * `doclens.npy` – Document lengths (shape: [n_documents], dtype: int32)
    /// * `centroids.npy` – Coarse centroids from first-level quantization (shape:
    ///   [n_coarse_centroids, token_dim], dtype: float32)
    /// * `index_assignment.npy` – Coarse centroid index per token (shape: [n_tokens], dtype:
    ///   uint64)
    /// * `residuals.npy` – PQ codes (shape: [n_tokens, M], dtype: uint8)
    /// * `pq_centroids.npy` – Flattened PQ centroids (shape: [M * 256 * dsub], dtype: float32,
    ///   where `dsub = token_dim / M`)
    ///
    /// Optionally:
    /// * `residual_norms.npy` – Per-token residual norm (shape: [n_tokens], dtype: float32),
    ///   for folders whose PQ codes were fitted to unit-normalised residuals. Its presence
    ///   changes the encoded token layout, so it is picked up automatically, exactly as
    ///   `hnsw_rerank_search --multivector-quantizer two-levels` does.
    ///
    /// The tokens stay PQ-encoded in memory — `COARSE_ID_BYTES + M` bytes each, plus
    /// `NORM_BYTES` with residual norms — and are scored by ADC. They are never reconstructed
    /// into `token_dim`-wide f32 vectors.
    ///
    #[staticmethod]
    #[pyo3(signature = (sparse_index_path, multivec_data_folder, pq_subspaces))]
    pub fn build_from_file(
        sparse_index_path: &str,
        multivec_data_folder: &str,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
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

        let inner = match pq_subspaces {
            8 => SparseMultivecTwoLevelsPQRerankIndexEnum::M8(RerankIndex::new(
                sparse_index,
                load_multivec_dataset_pq_generic::<8>(multivec_data_folder)?,
            )),
            16 => SparseMultivecTwoLevelsPQRerankIndexEnum::M16(RerankIndex::new(
                sparse_index,
                load_multivec_dataset_pq_generic::<16>(multivec_data_folder)?,
            )),
            32 => SparseMultivecTwoLevelsPQRerankIndexEnum::M32(RerankIndex::new(
                sparse_index,
                load_multivec_dataset_pq_generic::<32>(multivec_data_folder)?,
            )),
            64 => SparseMultivecTwoLevelsPQRerankIndexEnum::M64(RerankIndex::new(
                sparse_index,
                load_multivec_dataset_pq_generic::<64>(multivec_data_folder)?,
            )),
            _ => {
                return Err(value_err(format!(
                    "Unsupported pq_subspaces value: {}. Supported: 8, 16, 32, 64",
                    pq_subspaces
                )));
            }
        };

        Ok(SparseMultivecTwoLevelsPQRerankIndex { inner })
    }

    /// Saves the whole two-stage index — first-stage graph and rerank dataset — to one file.
    ///
    /// `build_from_file` reconstructs the index from a saved first-stage index plus the original
    /// multivector folder; this round-trips the index as it stands, so the PQ-encoded rerank
    /// dataset does not have to be re-read.
    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            SparseMultivecTwoLevelsPQRerankIndexEnum::M8(index) => index.save_index(path),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M16(index) => index.save_index(path),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M32(index) => index.save_index(path),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M64(index) => index.save_index(path),
        }
        .map_err(save_index_err)
    }

    /// Loads an index written by [`Self::save`].
    ///
    /// `pq_subspaces` must match the value used at build time, mirroring `build_from_file`.
    #[staticmethod]
    #[pyo3(signature = (path, pq_subspaces))]
    pub fn load(path: &str, pq_subspaces: usize) -> PyResult<Self> {
        let inner = match pq_subspaces {
            8 => SparseMultivecTwoLevelsPQRerankIndexEnum::M8(
                RerankIndex::load_index(path).map_err(load_index_err)?,
            ),
            16 => SparseMultivecTwoLevelsPQRerankIndexEnum::M16(
                RerankIndex::load_index(path).map_err(load_index_err)?,
            ),
            32 => SparseMultivecTwoLevelsPQRerankIndexEnum::M32(
                RerankIndex::load_index(path).map_err(load_index_err)?,
            ),
            64 => SparseMultivecTwoLevelsPQRerankIndexEnum::M64(
                RerankIndex::load_index(path).map_err(load_index_err)?,
            ),
            other => {
                return Err(value_err(format!(
                    "Unsupported pq_subspaces value: {other}. Supported: 8, 16, 32, 64"
                )));
            }
        };

        Ok(SparseMultivecTwoLevelsPQRerankIndex { inner })
    }

    /// Search with reranking using two-level PQ multivector encoding (single query).
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
        let _ = n_tokens;

        let results = match &self.inner {
            SparseMultivecTwoLevelsPQRerankIndexEnum::M8(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M16(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M32(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M64(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
        };

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        push_results(results, k, &mut distances, &mut ids);

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch search with reranking using two-level PQ multivector encoding, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
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
            let results = match &self.inner {
                SparseMultivecTwoLevelsPQRerankIndexEnum::M8(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M16(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M32(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M64(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
            };
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
