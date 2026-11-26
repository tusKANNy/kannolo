#[cfg(feature = "python")]
use pyo3::types::PyModuleMethods;

#[cfg(feature = "python")]
pub mod pylib;
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
#[pyo3(name = "_kannolo")]
pub fn kannolo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::pylib::ComposedIndex>()?;
    m.add_function(pyo3::wrap_pyfunction!(crate::pylib::validate_spec, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::pylib::compiled_features, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(crate::pylib::build_profile, m)?)?;
    Ok(())
}
