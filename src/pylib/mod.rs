//! Python bindings: the composable index (`kannolo.build`), see [`composable`].
//!
//! `too_many_arguments` is allowed module-wide. The `#[pymethods]` signatures mirror the
//! Python-facing keyword arguments one-for-one, and bundling them into a struct would change the
//! Python API to satisfy a Rust lint.
#![allow(clippy::too_many_arguments)]

mod composable;
pub use composable::{ComposedIndex, build_profile, compiled_features, validate_spec};
