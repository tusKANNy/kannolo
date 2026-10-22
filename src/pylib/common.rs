//! Helpers shared by every binding module, independent of the index structure: metric
//! parsing, numpy/`.npy` input conversion, and result packing.

use std::f32;

use half::f16;
use pyo3::prelude::*;
use vectorium::distances::Distance;
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use vectorium::readers::read_npy_f32;
use vectorium::vector::SparseVectorView;
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, Float, FromF32, PlainDenseDataset, PlainSparseDataset,
    PlainSparseDatasetGrowable, ValueType,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum MetricKind {
    Euclidean,
    DotProduct,
}

pub(super) fn parse_metric(metric: &str) -> PyResult<MetricKind> {
    let metric = metric.to_lowercase();
    match metric.as_str() {
        "euclidean" | "l2" => Ok(MetricKind::Euclidean),
        "dotproduct" | "ip" => Ok(MetricKind::DotProduct),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid metric; choose 'euclidean' or 'dotproduct'",
        )),
    }
}

pub(super) fn read_npy_dataset<D>(path: &str) -> PyResult<PlainDenseDataset<f32, D>>
where
    D: ScalarDenseSupportedDistance,
{
    read_npy_f32::<D>(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading .npy file: {e:?}"))
    })
}

pub(super) fn read_npy_dataset_f16<D>(path: &str) -> PyResult<PlainDenseDataset<f16, D>>
where
    D: ScalarDenseSupportedDistance + std::fmt::Debug,
{
    let dataset_f32 = read_npy_f32::<D>(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading .npy file: {e:?}"))
    })?;

    let dim = dataset_f32.input_dim();
    let n_vecs = dataset_f32.len();
    let data_f32: Vec<f32> = dataset_f32
        .iter()
        .flat_map(|v| v.values().iter().copied())
        .collect();
    let data_f16: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let encoder = PlainDenseQuantizer::<f16, D>::new(dim);
    Ok(DenseDataset::from_raw(
        data_f16.into_boxed_slice(),
        n_vecs,
        encoder,
    ))
}

pub(super) fn convert_components_to_u16(components: &[i32]) -> PyResult<Vec<u16>> {
    let mut out = Vec::with_capacity(components.len());
    for &c in components {
        if c < 0 || c > u16::MAX as i32 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Component out of range for u16",
            ));
        }
        out.push(c as u16);
    }
    Ok(out)
}

/// Sorts each query `[bounds[i], bounds[i + 1])` of a flattened batch by component, moving its
/// values along. The blocked sparse kernel walks a short query in component order, and numpy
/// input need not be sorted.
pub(super) fn sort_queries_by_component(
    mut components: Vec<u16>,
    values: &[f32],
    bounds: &[usize],
) -> PyResult<(Vec<u16>, Vec<f32>)> {
    check_same_len(components.len(), values.len())?;
    let mut values = values.to_vec();
    for w in bounds.windows(2) {
        let (c, v) = (&mut components[w[0]..w[1]], &mut values[w[0]..w[1]]);
        if c.is_sorted() {
            continue;
        }
        let mut pairs: Vec<(u16, f32)> = c.iter().copied().zip(v.iter().copied()).collect();
        pairs.sort_by_key(|p| p.0);
        for (i, (ci, vi)) in pairs.into_iter().enumerate() {
            c[i] = ci;
            v[i] = vi;
        }
    }
    Ok((components, values))
}

/// Rejects sparse components and values of different lengths.
pub(super) fn check_same_len(components: usize, values: usize) -> PyResult<()> {
    if components != values {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "components and values must have the same length, got {components} and {values}"
        )));
    }
    Ok(())
}

/// Converts `i64` offsets to `usize`, rejecting negative ones.
pub(super) fn offsets_to_usize(offsets: &[i64]) -> PyResult<Vec<usize>> {
    offsets
        .iter()
        .map(|&x| {
            usize::try_from(x).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Offsets must be non-negative")
            })
        })
        .collect()
}

pub(super) fn validate_offsets(offsets: &[usize], values_len: usize) -> PyResult<()> {
    if offsets.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Offsets must be non-empty",
        ));
    }
    if offsets[0] != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Offsets must start at 0",
        ));
    }
    if let Some(&last) = offsets.last()
        && last != values_len
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Offsets last element must equal number of values",
        ));
    }
    for w in offsets.windows(2) {
        if w[0] > w[1] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Offsets must be non-decreasing",
            ));
        }
    }
    Ok(())
}

pub(super) fn build_sparse_dataset_from_parts<V, D>(
    components: Vec<u16>,
    values: Vec<V>,
    offsets: Vec<usize>,
    dim: usize,
) -> PyResult<PlainSparseDataset<u16, V, D>>
where
    V: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    check_same_len(components.len(), values.len())?;
    validate_offsets(&offsets, values.len())?;

    let encoder = PlainSparseQuantizer::<u16, V, D>::new(dim, dim);
    let mut dataset: PlainSparseDatasetGrowable<u16, V, D> = DatasetGrowable::new(encoder);

    for i in 0..offsets.len() - 1 {
        let start = offsets[i];
        let end = offsets[i + 1];
        let view = SparseVectorView::new(&components[start..end], &values[start..end]);
        dataset.push(view);
    }

    Ok(dataset.into())
}

/// Error for a failed `save_index`.
pub(super) fn save_index_err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {e:?}"))
}

/// Error for a failed `load_index`.
///
/// Index files carry no header or type tag, so a file written with different build arguments
/// decodes into garbage or fails outright rather than reporting a mismatch.
pub(super) fn load_index_err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
        "Error loading index: {e:?}. Index files are not self-describing: check that the \
         arguments given here match the ones used when the index was built, and that the file \
         was written by this version of kannolo."
    ))
}

/// Pads `(score, id)` results to `k` entries with `inf` / `-1` and appends them to the buffers.
pub(super) fn push_pairs(
    results: Vec<(f32, usize)>,
    k: usize,
    distances: &mut Vec<f32>,
    ids: &mut Vec<i64>,
) {
    let found = results.len().min(k);
    for (d, id) in results.into_iter().take(k) {
        distances.push(d);
        ids.push(id as i64);
    }
    for _ in found..k {
        distances.push(f32::INFINITY);
        ids.push(-1);
    }
}

pub(super) fn push_results<D: Distance>(
    results: Vec<vectorium::dataset::ScoredVector<D>>,
    k: usize,
    distances: &mut Vec<f32>,
    ids: &mut Vec<i64>,
) {
    let mut found = 0;
    for scored in results.into_iter().take(k) {
        distances.push(scored.distance.distance());
        ids.push(scored.vector as i64);
        found += 1;
    }

    for _ in found..k {
        distances.push(f32::INFINITY);
        ids.push(-1);
    }
}
