//! Python bindings.
//!
//! Two clippy lints are allowed module-wide rather than at ~35 individual sites:
//!
//! * `too_many_arguments` — every `#[pyfunction]`/`#[pymethods]` signature here mirrors the
//!   Python-facing keyword arguments one-for-one. Bundling them into a struct would change
//!   the Python API to satisfy a Rust lint.
//! * `type_complexity` — the binding layer names fully monomorphized index types
//!   (`HNSW<DenseDataset<ProductQuantizer<M, D>>, GenericGraph<Ndst>>` and friends) in enum
//!   variants covering every encoder/graph combination. The types are long because they are
//!   explicit, which is the point of the enum dispatch.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod common;
mod flat;
mod hnsw;
mod rerank;

pub use flat::{DenseFlatIndex, SparseFlatIndex};
pub use hnsw::{
    DensePQHNSW, DensePlainHNSW, SparseDotPacking8HNSW, SparseFixedU8HNSW, SparseFixedU16HNSW,
    SparsePlainHNSW,
};
pub use rerank::{DenseRerankHNSW, SparseRerankHNSW};
#[cfg(feature = "multivec")]
pub use rerank::{SparseMultivecRerankIndex, SparseMultivecTwoLevelsPQRerankIndex};
