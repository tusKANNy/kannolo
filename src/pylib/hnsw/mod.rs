//! HNSW index bindings: one module per encoder/storage combination exposed to Python,
//! plus the `graph_type` vocabulary shared by all of them.

mod dense_plain;
mod dense_pq;
mod sparse_dotpacking8;
mod sparse_fixed_u16;
mod sparse_fixed_u8;
mod sparse_plain;

pub use dense_plain::DensePlainHNSW;
pub use dense_pq::DensePQHNSW;
pub use sparse_dotpacking8::SparseDotPacking8HNSW;
pub use sparse_fixed_u8::SparseFixedU8HNSW;
pub use sparse_fixed_u16::SparseFixedU16HNSW;
pub use sparse_plain::SparsePlainHNSW;

use crate::graph::neighbors::MAX_NEIGHBORS_PER_NODE;

use pyo3::prelude::*;

/// The HNSW graph storage/ordering strategy for build/load.
///
/// `Standard` and `Permuted` load as the same Rust type (`Graph<PlainNeighbors>`) — the
/// EGB permutation, when applied, is baked into the serialized index data itself
/// (`original_ids`), not reflected in the graph's storage type.
///
/// `Compressed` is the Python-facing name for what the CLI calls `streamvbyte`: the variant
/// names the user-visible behaviour, while the `StreamVByteNeighbors` type it selects names
/// the storage that implements it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum GraphTypeKind {
    Standard,
    Permuted,
    Compressed,
}

pub(super) fn parse_graph_type(graph_type: &str) -> PyResult<GraphTypeKind> {
    match graph_type.to_lowercase().as_str() {
        "standard" => Ok(GraphTypeKind::Standard),
        "permuted" => Ok(GraphTypeKind::Permuted),
        "compressed" => Ok(GraphTypeKind::Compressed),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown graph_type {other:?}; choose \"standard\" (the default), \"compressed\" \
             (smaller index and faster queries), or \"permuted\" (faster queries only). All \
             three return the same search results."
        ))),
    }
}

/// Parses `graph_type` for a *build* call, rejecting parameter combinations the chosen
/// representation cannot express.
///
/// Checked here rather than in `StreamVByteNeighbors::from`, which only runs once the whole
/// index has already been built — and which would abort the interpreter rather than raise,
/// since the release profile sets `panic = "abort"`.
pub(super) fn parse_build_graph_type(graph_type: &str, m: usize) -> PyResult<GraphTypeKind> {
    let gt = parse_graph_type(graph_type)?;

    if gt == GraphTypeKind::Compressed && 2 * m > MAX_NEIGHBORS_PER_NODE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "graph_type=\"compressed\" stores at most {MAX_NEIGHBORS_PER_NODE} neighbors per node, \
             but m={m} gives 2 * m = {} on the ground level; use m <= {} or graph_type=\"standard\"",
            2 * m,
            MAX_NEIGHBORS_PER_NODE / 2
        )));
    }

    Ok(gt)
}

/// Wraps a deserialization failure with the cause a caller is most likely to have hit.
///
/// Index files are plain bincode with no header or type tag, so loading one with the wrong
/// `metric` or `graph_type` fails somewhere inside the decoder with a message that says
/// nothing about the actual mistake.
fn load_index_err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
        "Error loading index: {e:?}. Index files are not self-describing: check that `metric` and \
         `graph_type` match the values used when the index was built, and that the file was \
         written by this version of kannolo."
    ))
}
