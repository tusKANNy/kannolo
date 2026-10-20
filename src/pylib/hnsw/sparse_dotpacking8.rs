//! `SparseDotPacking8HNSW`: HNSW over DotPacking8-packed sparse data with fixed-point 8-bit
//! values (u16 components, dotproduct only).

use std::f32;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{
    EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration,
};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::distances::DotProduct;
use vectorium::readers::read_seismic_format;
use vectorium::vector::SparseVectorView;
use vectorium::{DotPacking8U16FixedU8Dataset, PlainSparseDataset};

use super::{GraphTypeKind, load_index_err, parse_build_graph_type, parse_graph_type};
use crate::pylib::common::{
    build_sparse_dataset_from_parts, convert_components_to_u16, push_results,
};

enum SparseDotPacking8HNSWEnum {
    Plain(HNSW<DotPacking8U16FixedU8Dataset, Graph>),
    StreamVByte(HNSW<DotPacking8U16FixedU8Dataset, GenericGraph<StreamVByteNeighbors>>),
}

/// Applies `gt` to a plain-built sparse HNSW (permuting via EGB when requested) and
/// converts it into the DotPacking8 dataset representation.
///
/// The conversion runs before `permute_and_encode`, as in `build_permuted_and_save` in
/// `src/bin/hnsw_build.rs`: the encoder is then trained exactly as on the `standard` path and
/// the permutation copies the encoded rows verbatim.
fn build_sparse_dotpacking8_inner(
    plain_hnsw: HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph>,
    gt: GraphTypeKind,
) -> SparseDotPacking8HNSWEnum {
    match gt {
        GraphTypeKind::Standard => {
            let packed: HNSW<DotPacking8U16FixedU8Dataset, Graph> =
                plain_hnsw.convert_dataset_into(());
            SparseDotPacking8HNSWEnum::Plain(packed)
        }
        GraphTypeKind::Permuted => {
            let packed: HNSW<DotPacking8U16FixedU8Dataset, Graph> =
                plain_hnsw.convert_dataset_into(());
            SparseDotPacking8HNSWEnum::Plain(packed.permute_and_encode::<PlainNeighbors>())
        }
        GraphTypeKind::Compressed => {
            let packed: HNSW<DotPacking8U16FixedU8Dataset, Graph> =
                plain_hnsw.convert_dataset_into(());
            SparseDotPacking8HNSWEnum::StreamVByte(
                packed.permute_and_encode::<StreamVByteNeighbors>(),
            )
        }
    }
}

#[pyclass]
pub struct SparseDotPacking8HNSW {
    inner: SparseDotPacking8HNSWEnum,
}

#[pymethods]
impl SparseDotPacking8HNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        m: usize,
        ef_construction: usize,
        graph_type: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let dataset: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(data_file)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Error reading dataset: {:?}",
                    e
                ))
            })?;

        let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
        let inner = build_sparse_dotpacking8_inner(plain_hnsw, gt);

        Ok(SparseDotPacking8HNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, graph_type="standard".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        graph_type: String,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_vec = values.as_slice()?.to_vec();
        let offsets_vec = offsets
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();

        // Compute dimensionality from max component index
        let d = components_vec
            .iter()
            .max()
            .map(|&x| (x as usize) + 1)
            .unwrap_or(0);

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
            components_vec,
            values_vec,
            offsets_vec,
            d,
        )?;
        let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
        let inner = build_sparse_dotpacking8_inner(plain_hnsw, gt);

        Ok(SparseDotPacking8HNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            SparseDotPacking8HNSWEnum::Plain(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            SparseDotPacking8HNSWEnum::StreamVByte(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
        }
    }

    /// Loads a previously saved index. `graph_type` must match the value used at build
    /// time (`standard`/`permuted` both load as the same on-disk representation).
    #[staticmethod]
    #[pyo3(signature = (path, graph_type="standard".to_string()))]
    pub fn load(path: &str, graph_type: String) -> PyResult<Self> {
        let inner = match parse_graph_type(&graph_type)? {
            GraphTypeKind::Standard | GraphTypeKind::Permuted => {
                let index: HNSW<DotPacking8U16FixedU8Dataset, Graph> =
                    <HNSW<DotPacking8U16FixedU8Dataset, Graph> as IndexSerializer>::load_index(
                        path,
                    )
                    .map_err(load_index_err)?;
                SparseDotPacking8HNSWEnum::Plain(index)
            }
            GraphTypeKind::Compressed => {
                let index: HNSW<
                    DotPacking8U16FixedU8Dataset,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DotPacking8U16FixedU8Dataset,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparseDotPacking8HNSWEnum::StreamVByte(index)
            }
        };

        Ok(SparseDotPacking8HNSW { inner })
    }

    /// Search for approximate nearest neighbors for a single compressed sparse query.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices for the query.
    /// * `query_values` – 1-D float32 array of component values for the query.
    /// * `k` – Number of nearest neighbors to return.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query_components, query_values, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let mut ids = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        let query_view = SparseVectorView::new(&comp_vec, values_slice);
        match &self.inner {
            SparseDotPacking8HNSWEnum::Plain(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseDotPacking8HNSWEnum::StreamVByte(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
        }

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Search a batch of compressed sparse queries, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of component values (concatenated for batch).
    /// * `offsets` – 1-D int64 array defining query boundaries, e.g. `[0, n1, n1+n2, ...]`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    /// * `num_threads` – Threading model (see above). Default: 0 (all cores).
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let offsets_slice = offsets.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let num_queries = offsets_slice.len() - 1;

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let start = offsets_slice[i] as usize;
            let end = offsets_slice[i + 1] as usize;
            let query_view =
                SparseVectorView::new(&comp_vec[start..end], &values_slice[start..end]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                SparseDotPacking8HNSWEnum::Plain(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseDotPacking8HNSWEnum::StreamVByte(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
            }
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
