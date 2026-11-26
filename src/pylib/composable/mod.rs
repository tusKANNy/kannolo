//! The composable index: one Python class for every combination of representation, metric,
//! encoder, index structure and rescoring stages, chosen at run time.
//!
//! - [`config`]: the index configuration sent from Python as JSON, and its validation.
//! - [`build_load`]: from a configuration to the concrete Rust index types, built or loaded.
//! - [`any_index`]: one common interface for all those types, and the two-stage index.
//! - [`queries`]: queries from NumPy, converted for each representation.
//! - [`filter`]: filtered search (ACORN on HNSW).
//! - [`python_index`]: the Python class and the saved-file format.

mod any_index;
mod build_load;
mod config;
mod filter;
mod python_index;
mod queries;

pub use python_index::{ComposedIndex, build_profile, compiled_features, validate_spec};
