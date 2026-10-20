#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

#[cfg(feature = "python")]
pub mod pylib;
#[cfg(feature = "python")]
use crate::pylib::DenseFlatIndex;
#[cfg(feature = "python")]
use crate::pylib::DensePQHNSW as DensePQIndexPy;
#[cfg(feature = "python")]
use crate::pylib::DensePlainHNSW as DensePlainIndexPy;
#[cfg(feature = "python")]
use crate::pylib::DenseRerankHNSW;
#[cfg(feature = "python")]
use crate::pylib::SparseDotPacking8HNSW as SparseDotPacking8IndexPy;
#[cfg(feature = "python")]
use crate::pylib::SparseFixedU8HNSW as SparseFixedU8IndexPy;
#[cfg(feature = "python")]
use crate::pylib::SparseFixedU16HNSW as SparseFixedU16IndexPy;
#[cfg(feature = "python")]
use crate::pylib::SparseFlatIndex;
#[cfg(all(feature = "python", feature = "multivec"))]
use crate::pylib::SparseMultivecRerankIndex;
#[cfg(all(feature = "python", feature = "multivec"))]
use crate::pylib::SparseMultivecTwoLevelsPQRerankIndex;
#[cfg(feature = "python")]
use crate::pylib::SparsePlainHNSW as SparsePlainIndexPy;
#[cfg(feature = "python")]
use crate::pylib::SparseRerankHNSW;
#[cfg(feature = "python")]
use pyo3::prelude::PyModule;
#[cfg(feature = "python")]
use pyo3::{Bound, PyResult, pymodule};

pub mod graph;
pub mod utils;
pub mod visited_set;

pub mod indexes;
pub use indexes::{hnsw, hnsw_utils, ivf, rerank, rerank_index};

#[cfg(feature = "python")]
#[pymodule]
pub fn kannolo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DensePlainIndexPy>()?;
    m.add_class::<SparsePlainIndexPy>()?;
    m.add_class::<SparseDotPacking8IndexPy>()?;
    m.add_class::<SparseFixedU8IndexPy>()?;
    m.add_class::<SparseFixedU16IndexPy>()?;
    m.add_class::<DensePQIndexPy>()?;
    m.add_class::<DenseFlatIndex>()?;
    m.add_class::<SparseFlatIndex>()?;
    m.add_class::<DenseRerankHNSW>()?;
    m.add_class::<SparseRerankHNSW>()?;
    #[cfg(feature = "multivec")]
    m.add_class::<SparseMultivecRerankIndex>()?;
    #[cfg(feature = "multivec")]
    m.add_class::<SparseMultivecTwoLevelsPQRerankIndex>()?;
    Ok(())
}
