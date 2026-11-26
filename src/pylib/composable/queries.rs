//! Queries passed from NumPy, and their conversion to the query type of each representation
//! (dense, sparse, multi-vector).

// In a build without one of the representations, the query layout of that representation is
// unused.
#![cfg_attr(
    not(all(
        feature = "dense",
        feature = "sparse",
        feature = "multivec",
        feature = "ivf"
    )),
    allow(dead_code, unused_imports)
)]

use std::marker::PhantomData;

use num_traits::FromPrimitive;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

use vectorium::ComponentType;
#[cfg(feature = "multivec")]
use vectorium::vector::DenseMultiVectorView;
use vectorium::vector::{DenseVectorView, SparseVectorView};

fn value_err(msg: impl Into<String>) -> pyo3::PyErr {
    PyValueError::new_err(msg.into())
}

/// CSR-style sparse queries: query `i` is `components[offsets[i]..offsets[i + 1]]`.
#[derive(Clone, Copy)]
pub(crate) struct SparseBatch<'a> {
    pub components: &'a [u32],
    pub values: &'a [f32],
    pub offsets: &'a [usize],
}

/// Multi-vector queries: query `i` is the tokens `token_offsets[i]..token_offsets[i + 1]` of
/// `values`, each of `dim` floats.
#[derive(Clone, Copy)]
pub(crate) struct MultiVecBatch<'a> {
    pub values: &'a [f32],
    pub token_offsets: &'a [usize],
    pub dim: usize,
}

/// One batch of queries per representation. Each stage of an index reads the batch of its own
/// representation, so a sparse first stage and a multi-vector rescoring stage each get theirs.
#[derive(Clone, Copy, Default)]
pub(crate) struct Queries<'a> {
    /// Row-major, `n * dim` floats.
    pub dense: Option<&'a [f32]>,
    pub sparse: Option<SparseBatch<'a>>,
    pub multivec: Option<MultiVecBatch<'a>>,
}

/// How queries for one storage layout are converted and viewed.
pub(crate) trait QueryFormat: 'static {
    /// The query type the index's `search` takes.
    type View<'q>;
    /// A batch converted once (component types narrowed, components sorted, ...), then viewed
    /// query by query from every search thread.
    type ConvertedQueries<'a>: Sync;

    fn convert<'a>(queries: &Queries<'a>, dim: usize) -> PyResult<Self::ConvertedQueries<'a>>;
    fn len(converted: &Self::ConvertedQueries<'_>) -> usize;
    fn view<'p>(converted: &'p Self::ConvertedQueries<'_>, i: usize) -> Self::View<'p>;
}

/// Dense queries, `f32` views straight into the numpy buffer (no copy).
pub(crate) struct DenseFormat;

pub(crate) struct DenseConvertedQueries<'a> {
    values: &'a [f32],
    dim: usize,
}

impl QueryFormat for DenseFormat {
    type View<'q> = DenseVectorView<'q, f32>;
    type ConvertedQueries<'a> = DenseConvertedQueries<'a>;

    fn convert<'a>(queries: &Queries<'a>, dim: usize) -> PyResult<DenseConvertedQueries<'a>> {
        let Some(values) = queries.dense else {
            return Err(value_err(
                "this index needs dense queries (a 1-D or 2-D float array)",
            ));
        };
        if dim == 0 || values.len() % dim != 0 {
            return Err(value_err(format!(
                "dense query length {} is not a multiple of the index dimension {dim}",
                values.len()
            )));
        }
        Ok(DenseConvertedQueries { values, dim })
    }

    fn len(p: &DenseConvertedQueries<'_>) -> usize {
        p.values.len() / p.dim
    }

    fn view<'p>(p: &'p DenseConvertedQueries<'_>, i: usize) -> DenseVectorView<'p, f32> {
        DenseVectorView::new(&p.values[i * p.dim..(i + 1) * p.dim])
    }
}

/// Sparse queries with components of type `C`.
pub(crate) struct SparseFormat<C>(PhantomData<C>);

pub(crate) struct SparseConvertedQueries<C> {
    components: Vec<C>,
    values: Vec<f32>,
    offsets: Vec<usize>,
}

/// Checks CSR offsets against the number of stored entries.
pub(crate) fn check_offsets(offsets: &[usize], nnz: usize) -> PyResult<()> {
    if offsets.first() != Some(&0) {
        return Err(value_err("offsets must be non-empty and start at 0"));
    }
    if offsets.last() != Some(&nnz) {
        return Err(value_err(format!(
            "the last offset must equal the number of entries ({nnz}), got {}",
            offsets.last().unwrap()
        )));
    }
    if offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(value_err("offsets must be non-decreasing"));
    }
    Ok(())
}

/// Sorts the `(component, value)` pairs of one sparse vector by component and rejects repeated
/// components. The sparse datasets and kernels need strictly ascending components, and a
/// violated assertion would abort the interpreter (the release profile sets `panic = "abort"`).
pub(crate) fn sorted_pairs(components: &[u32], values: &[f32]) -> PyResult<Vec<(u32, f32)>> {
    let mut pairs: Vec<(u32, f32)> = components
        .iter()
        .copied()
        .zip(values.iter().copied())
        .collect();
    if !pairs.is_sorted_by_key(|p| p.0) {
        pairs.sort_unstable_by_key(|p| p.0);
    }
    if let Some(w) = pairs.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(value_err(format!(
            "component {} appears more than once in the same vector",
            w[0].0
        )));
    }
    Ok(pairs)
}

impl<C> QueryFormat for SparseFormat<C>
where
    C: ComponentType + FromPrimitive + Sync + 'static,
{
    type View<'q> = SparseVectorView<'q, C, f32>;
    type ConvertedQueries<'a> = SparseConvertedQueries<C>;

    fn convert<'a>(queries: &Queries<'a>, dim: usize) -> PyResult<SparseConvertedQueries<C>> {
        let Some(SparseBatch {
            components,
            values,
            offsets,
        }) = queries.sparse
        else {
            return Err(value_err(
                "this index needs sparse queries ((components, values) for one query, or \
                 (components, values, offsets) / a scipy.sparse matrix for a batch)",
            ));
        };
        if components.len() != values.len() {
            return Err(value_err(format!(
                "components and values must have the same length, got {} and {}",
                components.len(),
                values.len()
            )));
        }
        check_offsets(offsets, values.len())?;

        let mut out = SparseConvertedQueries {
            components: Vec::with_capacity(components.len()),
            values: Vec::with_capacity(values.len()),
            offsets: Vec::with_capacity(offsets.len()),
        };
        out.offsets.push(0);
        for w in offsets.windows(2) {
            for (c, v) in sorted_pairs(&components[w[0]..w[1]], &values[w[0]..w[1]])? {
                // Drop components that no indexed vector has. They cannot change any inner
                // product, and the query evaluators index a buffer of size `dim`.
                if (c as usize) < dim {
                    out.components
                        .push(C::from_u32(c).expect("c < dim fits the component type"));
                    out.values.push(v);
                }
            }
            out.offsets.push(out.components.len());
        }
        Ok(out)
    }

    fn len(p: &SparseConvertedQueries<C>) -> usize {
        p.offsets.len() - 1
    }

    fn view<'p>(p: &'p SparseConvertedQueries<C>, i: usize) -> SparseVectorView<'p, C, f32> {
        let (s, e) = (p.offsets[i], p.offsets[i + 1]);
        SparseVectorView::new(&p.components[s..e], &p.values[s..e])
    }
}

/// Multi-vector queries, `f32` token views straight into the numpy buffer.
#[cfg(feature = "multivec")]
pub(crate) struct MultiVecFormat;

#[cfg(feature = "multivec")]
pub(crate) struct MultiVecConvertedQueries<'a> {
    values: &'a [f32],
    token_offsets: &'a [usize],
    dim: usize,
}

#[cfg(feature = "multivec")]
impl QueryFormat for MultiVecFormat {
    type View<'q> = DenseMultiVectorView<'q, f32>;
    type ConvertedQueries<'a> = MultiVecConvertedQueries<'a>;

    fn convert<'a>(queries: &Queries<'a>, dim: usize) -> PyResult<MultiVecConvertedQueries<'a>> {
        let Some(MultiVecBatch {
            values,
            token_offsets,
            dim: query_dim,
        }) = queries.multivec
        else {
            return Err(value_err(
                "this index needs multi-vector queries (a 2-D (tokens, dim) array for one \
                 query, or a list of them / (tokens, lengths) for a batch)",
            ));
        };
        if query_dim != dim {
            return Err(value_err(format!(
                "query tokens have dimension {query_dim}, but the index has dimension {dim}"
            )));
        }
        check_offsets(token_offsets, values.len() / dim.max(1))?;
        if values.len() % dim != 0 {
            return Err(value_err(
                "multi-vector query values are not a whole number of tokens",
            ));
        }
        if token_offsets.windows(2).any(|w| w[0] == w[1]) {
            return Err(value_err(
                "every multi-vector query needs at least one token",
            ));
        }
        Ok(MultiVecConvertedQueries {
            values,
            token_offsets,
            dim,
        })
    }

    fn len(p: &MultiVecConvertedQueries<'_>) -> usize {
        p.token_offsets.len() - 1
    }

    fn view<'p>(p: &'p MultiVecConvertedQueries<'_>, i: usize) -> DenseMultiVectorView<'p, f32> {
        let (s, e) = (p.token_offsets[i] * p.dim, p.token_offsets[i + 1] * p.dim);
        DenseMultiVectorView::new(&p.values[s..e], p.dim)
    }
}
