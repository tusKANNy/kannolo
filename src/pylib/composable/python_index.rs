//! The Python class of the composable index, `_ComposedIndex`: one class for every combination
//! of representation, metric, encoder, index structure and rescoring stages, chosen at run time
//! from an [`IndexConfig`].
//!
//! The user-facing API lives in `python/kannolo/composable.py`; this class is its backend and
//! is not meant to be used directly.
//!
//! # File format
//!
//! ```text
//! b"KANNOLO\0" | u32 LE format version | u32 LE spec length | spec (JSON) | payloads
//! ```
//!
//! The payloads are the first stage, then each rescoring dataset, each in the format that
//! `vectorium::IndexSerializer` writes. The header describes the file's contents, so `load` needs
//! no arguments besides the path.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::any_index::{AnyIndex, RerankArgs, SearchArgs};
use super::build_load::{Input, OwnedInput};
use super::config::IndexConfig;
use super::filter::Filter;
use super::queries::{MultiVecBatch, Queries, SparseBatch};

const MAGIC: &[u8; 8] = b"KANNOLO\0";
const FORMAT_VERSION: u32 = 1;

fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn offsets_to_usize(offsets: &[i64]) -> PyResult<Vec<usize>> {
    offsets
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| value_err("offsets must be non-negative")))
        .collect()
}

fn parse_spec(spec_json: &str) -> PyResult<IndexConfig> {
    IndexConfig::from_json(spec_json).map_err(value_err)
}

/// Checks a spec without building anything. Raises `ValueError` with the reason if the
/// combination is not available.
#[pyfunction]
#[pyo3(name = "_validate_spec")]
pub fn validate_spec(spec_json: &str) -> PyResult<()> {
    parse_spec(spec_json).map(|_| ())
}

/// Data handed over from Python, kept alive (and borrowed from) while an index is built.
///
/// One descriptor per stage, a tuple whose first item names its kind:
/// `("dense", values, dim)`, `("npy", path)`, `("sparse", components, values, offsets)`,
/// `("bin", path)`, `("multivec", values, token_offsets, dim)`, or `None`.
enum PythonInput<'py> {
    None,
    Dense(PyReadonlyArray1<'py, f32>, usize),
    Sparse(
        PyReadonlyArray1<'py, u32>,
        PyReadonlyArray1<'py, f32>,
        Vec<usize>,
    ),
    MultiVec(PyReadonlyArray1<'py, f32>, Vec<usize>, usize),
    Owned(OwnedInput),
}

impl<'py> PythonInput<'py> {
    fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if obj.is_none() {
            return Ok(PythonInput::None);
        }
        let t = obj.cast::<PyTuple>()?;
        let kind: String = t.get_item(0)?.extract()?;
        Ok(match kind.as_str() {
            "dense" => PythonInput::Dense(t.get_item(1)?.extract()?, t.get_item(2)?.extract()?),
            "npy" => PythonInput::Owned(super::build_load::read_npy(
                &t.get_item(1)?.extract::<String>()?,
            )?),
            "sparse" => {
                let offsets: PyReadonlyArray1<'py, i64> = t.get_item(3)?.extract()?;
                PythonInput::Sparse(
                    t.get_item(1)?.extract()?,
                    t.get_item(2)?.extract()?,
                    offsets_to_usize(offsets.as_slice()?)?,
                )
            }
            "bin" => PythonInput::Owned(super::build_load::read_seismic(
                &t.get_item(1)?.extract::<String>()?,
            )?),
            "multivec" => {
                let offsets: PyReadonlyArray1<'py, i64> = t.get_item(2)?.extract()?;
                PythonInput::MultiVec(
                    t.get_item(1)?.extract()?,
                    offsets_to_usize(offsets.as_slice()?)?,
                    t.get_item(3)?.extract()?,
                )
            }
            other => return Err(value_err(format!("unknown data kind {other:?}"))),
        })
    }

    fn input(&self) -> PyResult<Option<Input<'_>>> {
        Ok(match self {
            PythonInput::None => None,
            PythonInput::Dense(values, dim) => Some(Input::Dense {
                values: values.as_slice()?,
                dim: *dim,
            }),
            PythonInput::Sparse(components, values, offsets) => Some(Input::Sparse {
                components: components.as_slice()?,
                values: values.as_slice()?,
                offsets,
            }),
            PythonInput::MultiVec(values, token_offsets, dim) => Some(Input::MultiVec {
                values: values.as_slice()?,
                token_offsets,
                dim: *dim,
            }),
            PythonInput::Owned(owned) => Some(owned.view()),
        })
    }
}

/// The parts of the composable API this build compiled: a subset of "dense", "sparse",
/// "multivec" and "ivf" (the Cargo features of the same names).
#[pyfunction]
#[pyo3(name = "_compiled_features")]
pub fn compiled_features() -> Vec<&'static str> {
    super::config::compiled_features()
}

/// One rescoring stage's search arguments from Python: `(k_candidates, alpha, beta, query_bits)`.
type RerankStageArgs = (usize, Option<f32>, Option<usize>, Option<u32>);

/// Search results for Python: `(n_queries, distances, ids)`, flattened to `n_queries * k`.
type SearchOutput = (usize, Py<PyArray1<f32>>, Py<PyArray1<i64>>);

/// The Cargo profile this library was built with: `"release"` (what the published wheels use,
/// `codegen-units = 1`) or `"dev-python"` (faster to compile, slower search).
#[pyfunction]
#[pyo3(name = "_build_profile")]
pub fn build_profile() -> &'static str {
    if cfg!(kannolo_dev_profile) {
        "dev-python"
    } else {
        "release"
    }
}

/// Backend of `kannolo.Index`.
#[pyclass(name = "_ComposedIndex", module = "kannolo._kannolo")]
pub struct ComposedIndex {
    /// Shared: `compose` reuses it as the first stage of new indexes without copying it.
    inner: Arc<dyn AnyIndex>,
    spec: IndexConfig,
}

#[pymethods]
impl ComposedIndex {
    /// `data` holds one descriptor per stage (see `PythonInput`): the first stage's data, then each
    /// rescoring stage's. `None` means "the first stage's data".
    #[staticmethod]
    fn build(py: Python<'_>, spec_json: &str, data: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let spec = parse_spec(spec_json)?;
        if data.len() != 1 + spec.rerank.len() {
            return Err(value_err(format!(
                "{} data descriptors for {} stages",
                data.len(),
                1 + spec.rerank.len()
            )));
        }
        let held: Vec<PythonInput<'_>> = data
            .iter()
            .map(PythonInput::extract)
            .collect::<PyResult<_>>()?;
        let inputs: Vec<Option<Input<'_>>> = held
            .iter()
            .map(PythonInput::input)
            .collect::<PyResult<_>>()?;
        let inner = py.detach(|| super::build_load::build(&spec, &inputs))?;
        Ok(Self { inner, spec })
    }

    /// A new index whose first stage is `first`, shared without copying. Its spec is `spec_json`:
    /// `first`'s spec plus more rescoring stages. Each new stage is built from its descriptor in
    /// `data` (see `PythonInput`).
    #[staticmethod]
    fn compose(
        py: Python<'_>,
        first: PyRef<'_, Self>,
        spec_json: &str,
        data: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let spec = parse_spec(spec_json)?;
        let from_level = first.spec.rerank.len();
        if spec.first != first.spec.first
            || spec.rerank.len() <= from_level
            || spec.rerank[..from_level] != first.spec.rerank[..]
        {
            return Err(value_err(
                "the composed spec must be the index's own spec plus rescoring stages",
            ));
        }
        let held: Vec<PythonInput<'_>> = data
            .iter()
            .map(PythonInput::extract)
            .collect::<PyResult<_>>()?;
        let inputs: Vec<Input<'_>> = held
            .iter()
            .map(|h| {
                h.input()?
                    .ok_or_else(|| value_err("every composed rescoring stage needs its data"))
            })
            .collect::<PyResult<_>>()?;
        let shared = Arc::clone(&first.inner);
        let inner = py.detach(|| super::build_load::extend(shared, &spec, from_level, &inputs))?;
        Ok(Self { inner, spec })
    }

    /// Builds (or rebuilds) the ACORN-γ neighbor lists of the HNSW first stage, used by
    /// filtered search. They live in memory only: `save` does not write them.
    fn build_acorn_gamma(&self, py: Python<'_>, gamma: usize) -> PyResult<()> {
        py.detach(|| self.inner.build_acorn_gamma(gamma))
    }

    #[getter]
    fn has_acorn_gamma(&self) -> bool {
        self.inner.has_acorn_gamma()
    }

    /// `queries` maps a representation ("dense", "sparse", "multivec") to its batch:
    /// a flat float array, `(components, values, offsets)`, or `(values, token_offsets, dim)`.
    /// `rerank` holds one `(k_candidates, alpha, beta, query_bits)` per rescoring stage.
    /// `filter`, when given, is one of:
    /// - `("mask", bool array over the collection)`;
    /// - `("masks", n_queries * n bools, n)`: one mask per query, row-major;
    /// - `("per_query", ids, offsets)`: the ids each query may return, CSR-style.
    ///
    /// Returns `(n_queries, distances, ids)`, the arrays flattened to `n_queries * k`.
    #[pyo3(signature = (queries, k, ef_search, early_exit_threshold, query_bits, nprobe, rerank, num_threads, filter=None, acorn_gamma=None))]
    fn search(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyDict>,
        k: usize,
        ef_search: Option<usize>,
        early_exit_threshold: Option<f32>,
        query_bits: Option<u32>,
        nprobe: Option<usize>,
        rerank: Vec<RerankStageArgs>,
        num_threads: usize,
        filter: Option<Bound<'_, PyTuple>>,
        acorn_gamma: Option<bool>,
    ) -> PyResult<SearchOutput> {
        if rerank.len() > self.spec.rerank.len() {
            return Err(value_err(format!(
                "k_candidates, alpha, beta or rerank_query_bits were given for {} rescoring \
                 stages, but the index has {}",
                rerank.len(),
                self.spec.rerank.len()
            )));
        }
        let dense: Option<PyReadonlyArray1<'_, f32>> = match queries.get_item("dense")? {
            Some(v) => Some(v.extract()?),
            None => None,
        };
        let sparse: Option<(
            PyReadonlyArray1<'_, u32>,
            PyReadonlyArray1<'_, f32>,
            Vec<usize>,
        )> = match queries.get_item("sparse")? {
            Some(t) => {
                let (c, v, o): (
                    PyReadonlyArray1<'_, u32>,
                    PyReadonlyArray1<'_, f32>,
                    PyReadonlyArray1<'_, i64>,
                ) = t.extract()?;
                let o = offsets_to_usize(o.as_slice()?)?;
                Some((c, v, o))
            }
            None => None,
        };
        let multivec: Option<(PyReadonlyArray1<'_, f32>, Vec<usize>, usize)> =
            match queries.get_item("multivec")? {
                Some(t) => {
                    let (v, o, d): (PyReadonlyArray1<'_, f32>, PyReadonlyArray1<'_, i64>, usize) =
                        t.extract()?;
                    let o = offsets_to_usize(o.as_slice()?)?;
                    Some((v, o, d))
                }
                None => None,
            };
        let q = Queries {
            dense: dense.as_ref().map(|d| d.as_slice()).transpose()?,
            sparse: match &sparse {
                Some((c, v, o)) => Some(SparseBatch {
                    components: c.as_slice()?,
                    values: v.as_slice()?,
                    offsets: o,
                }),
                None => None,
            },
            multivec: match &multivec {
                Some((v, o, d)) => Some(MultiVecBatch {
                    values: v.as_slice()?,
                    token_offsets: o,
                    dim: *d,
                }),
                None => None,
            },
        };
        let args = SearchArgs {
            ef_search,
            early_exit_threshold,
            query_bits,
            nprobe,
            acorn_gamma,
            rerank: rerank
                .into_iter()
                .map(|(k_candidates, alpha, beta, query_bits)| RerankArgs {
                    k_candidates,
                    alpha,
                    beta,
                    query_bits,
                })
                .collect(),
        };
        if acorn_gamma.is_some() && filter.is_none() {
            return Err(value_err(
                "acorn_gamma applies to filtered search (pass filter=)",
            ));
        }
        let mask: Option<PyReadonlyArray1<'_, bool>>;
        let per_query: Option<(PyReadonlyArray1<'_, u64>, Vec<usize>)>;
        let filter = match &filter {
            None => None,
            Some(t) => {
                let kind: String = t.get_item(0)?.extract()?;
                match kind.as_str() {
                    "mask" => {
                        mask = Some(t.get_item(1)?.extract()?);
                        Some(Filter::Shared(mask.as_ref().unwrap().as_slice()?))
                    }
                    "masks" => {
                        let width: usize = t.get_item(2)?.extract()?;
                        mask = Some(t.get_item(1)?.extract()?);
                        Some(Filter::PerQueryMasks {
                            masks: mask.as_ref().unwrap().as_slice()?,
                            n: width,
                        })
                    }
                    "per_query" => {
                        let offsets: PyReadonlyArray1<'_, i64> = t.get_item(2)?.extract()?;
                        per_query = Some((
                            t.get_item(1)?.extract()?,
                            offsets_to_usize(offsets.as_slice()?)?,
                        ));
                        let (ids, offsets) = per_query.as_ref().unwrap();
                        super::queries::check_offsets(offsets, ids.len()?)?;
                        Some(Filter::PerQuery {
                            ids: ids.as_slice()?,
                            offsets,
                        })
                    }
                    other => return Err(value_err(format!("unknown filter kind {other:?}"))),
                }
            }
        };
        let r = py.detach(|| match &filter {
            None => self.inner.search(&q, k, &args, num_threads),
            Some(f) => self.inner.search_filtered(&q, k, &args, f, num_threads),
        })?;
        Ok((
            r.n_queries,
            PyArray1::from_vec(py, r.distances).unbind(),
            PyArray1::from_vec(py, r.ids).unbind(),
        ))
    }

    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let io_err = |e: String| PyIOError::new_err(format!("cannot save index to {path:?}: {e}"));
        py.detach(|| {
            let file = File::create(path).map_err(|e| io_err(e.to_string()))?;
            let mut w = BufWriter::new(file);
            let spec = self.spec.to_json();
            w.write_all(MAGIC)
                .and_then(|_| w.write_all(&FORMAT_VERSION.to_le_bytes()))
                .and_then(|_| w.write_all(&(spec.len() as u32).to_le_bytes()))
                .and_then(|_| w.write_all(spec.as_bytes()))
                .map_err(|e| io_err(e.to_string()))?;
            self.inner.write_payload(&mut w).map_err(io_err)?;
            w.flush().map_err(|e| io_err(e.to_string()))
        })
    }

    /// Loads any index written by `save`; the spec comes from the file header.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let io_err =
            |e: String| PyIOError::new_err(format!("cannot load index from {path:?}: {e}"));
        py.detach(|| {
            let file = File::open(path).map_err(|e| io_err(e.to_string()))?;
            let mut r = BufReader::new(file);
            let spec = read_header(&mut r).map_err(io_err)?;
            let spec = IndexConfig::from_json(&spec).map_err(io_err)?;
            let inner = super::build_load::load(&mut r, &spec)?;
            Ok(Self { inner, spec })
        })
    }

    /// Reads only the spec stored in an index file's header, as JSON.
    #[staticmethod]
    fn read_spec(path: &str) -> PyResult<String> {
        let file = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("cannot open {path:?}: {e}")))?;
        read_header(&mut BufReader::new(file))
            .map_err(|e| PyIOError::new_err(format!("cannot read {path:?}: {e}")))
    }

    #[getter]
    fn spec_json(&self) -> String {
        self.spec.to_json()
    }

    /// The first stage's dimensionality (token dimension for multi-vector data).
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn n_elements(&self) -> usize {
        self.inner.n_elements()
    }

    /// Total bytes held by every stage, as reported by `SpaceUsage`.
    #[getter]
    fn space_usage_bytes(&self) -> usize {
        self.inner.space_usage_bytes()
    }
}

fn read_header(r: &mut impl Read) -> Result<String, String> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err(
            "not an index written by kannolo.Index.save (files written by the CLI binaries, \
             such as hnsw_build, have no header and are read by those binaries)"
                .into(),
        );
    }
    let mut word = [0u8; 4];
    r.read_exact(&mut word).map_err(|e| e.to_string())?;
    let version = u32::from_le_bytes(word);
    if version != FORMAT_VERSION {
        return Err(format!(
            "file format version {version}, but this kannolo reads version {FORMAT_VERSION}"
        ));
    }
    r.read_exact(&mut word).map_err(|e| e.to_string())?;
    let mut spec = vec![0u8; u32::from_le_bytes(word) as usize];
    r.read_exact(&mut spec).map_err(|e| e.to_string())?;
    String::from_utf8(spec).map_err(|e| e.to_string())
}
