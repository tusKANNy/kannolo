//! Builds and loads indexes: from an [`IndexConfig`] to the concrete Rust index types.
//!
//! An *encoding* (a type implementing [`Encoding`]) names one stored representation and how to
//! produce it from input data. The same encoding serves three roles: the dataset of an HNSW
//! graph (built on plain vectors, then encoded, as `src/bin/hnsw_build.rs` does), the dataset
//! of a Flat index, and a rescoring dataset. [`with_encoding!`] is the only table that maps a
//! spec to an encoding type.
//!
//! IVF stores its clusters in dot-product form, next to a centroid dataset. These types form a
//! separate family. [`with_ivf!`] is their table and follows `src/bin/ivf_build.rs`.
//!
//! [`build`] and [`load`] assemble a first stage and its rescoring stages through the same
//! tables, so the two always agree on the stage types.

// In a build without one of the representations (see the `dense`, `sparse`, `multivec` and
// `ivf` features), the encodings and helpers that only this representation uses are unused.
#![cfg_attr(
    not(all(
        feature = "dense",
        feature = "sparse",
        feature = "multivec",
        feature = "ivf"
    )),
    allow(dead_code, unused_imports, unused_macros)
)]

use std::io::Read;
use std::marker::PhantomData;
use std::sync::Arc;

use half::{bf16, f16};
use num_traits::FromPrimitive;
use pyo3::PyResult;
use pyo3::exceptions::{PyIOError, PyValueError};
use serde::de::DeserializeOwned;

use vectorium::core::vector_encoder::VectorEncoder;
use vectorium::dataset::ConvertFrom;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{
    PlainDenseQuantizer, ScalarDenseQuantizer, ScalarDenseSupportedDistance,
};
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::transformations::l1_prune::l1_prune_split_vector;
use vectorium::vector::{DenseVectorView, SparseVectorView};
use vectorium::{
    ComponentType, Dataset, DatasetGrowable, DenseDataset, FixedU8Q, FixedU16Q, FlatIndex, Float,
    FromF32, KMeansBuilder, PlainDenseDataset, PlainDenseDatasetGrowable, PlainSparseDataset,
    PlainSparseDatasetGrowable, RabitqConfig, RabitqExtConfig, RabitqExtQuantizer, RabitqQuantizer,
    RabitqSupportedDistance, ScalarDenseDataset, ScalarSparseDataset, SpaceUsage, ValueType,
    WithDistance,
};

#[cfg(feature = "multivec")]
use vectorium::{
    MultiVecProductQuantizer, MultiVecTwoLevelProductQuantizer, MultiVectorDataset,
    PlainMultiVecQuantizer,
};

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{HNSW, HNSWBuildConfiguration};
use crate::ivf::{AnnAssignment, IVF, ReportedMetric, encode_clusters, prepare_ivf};

use super::any_index::{
    Accepts, AnyIndex, AnyRescorer, Rescorer, SingleStage, TwoStage, read_payload,
};
use super::config::{
    CentroidIndexConfig, EncoderConfig, EncodingConfig, GraphKind, IndexConfig,
    IndexStructureConfig, Representation, RerankConfig, StageConfig, TransformConfig,
};
#[cfg(feature = "multivec")]
use super::queries::MultiVecFormat;
use super::queries::{DenseFormat, QueryFormat, SparseFormat, check_offsets, sorted_pairs};

fn value_err(msg: impl Into<String>) -> pyo3::PyErr {
    PyValueError::new_err(msg.into())
}

// ----------------------------------------------------------------------------- input data

/// The data a stage is built from, borrowed from numpy arrays or from an [`OwnedInput`].
#[derive(Clone, Copy)]
pub(crate) enum Input<'a> {
    /// Row-major, `n * dim` floats.
    Dense { values: &'a [f32], dim: usize },
    /// CSR: vector `i` is `components[offsets[i]..offsets[i + 1]]`.
    Sparse {
        components: &'a [u32],
        values: &'a [f32],
        offsets: &'a [usize],
    },
    /// Vector `i` is the tokens `token_offsets[i]..token_offsets[i + 1]`, `dim` floats each.
    MultiVec {
        values: &'a [f32],
        token_offsets: &'a [usize],
        dim: usize,
    },
}

impl Input<'_> {
    pub(crate) fn representation(&self) -> Representation {
        match self {
            Input::Dense { .. } => Representation::Dense,
            Input::Sparse { .. } => Representation::Sparse,
            Input::MultiVec { .. } => Representation::Multivector,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match *self {
            Input::Dense { values, dim } => values.len() / dim.max(1),
            Input::Sparse { offsets, .. } => offsets.len().saturating_sub(1),
            Input::MultiVec { token_offsets, .. } => token_offsets.len().saturating_sub(1),
        }
    }
}

/// Data read from a file, or derived (pruned, residual) from the input.
pub(crate) enum OwnedInput {
    Dense {
        values: Vec<f32>,
        dim: usize,
    },
    Sparse {
        components: Vec<u32>,
        values: Vec<f32>,
        offsets: Vec<usize>,
    },
}

impl OwnedInput {
    pub(crate) fn view(&self) -> Input<'_> {
        match self {
            OwnedInput::Dense { values, dim } => Input::Dense { values, dim: *dim },
            OwnedInput::Sparse {
                components,
                values,
                offsets,
            } => Input::Sparse {
                components,
                values,
                offsets,
            },
        }
    }
}

/// Reads a 2-D float32 `.npy` file.
pub(crate) fn read_npy(path: &str) -> PyResult<OwnedInput> {
    let data = read_npy_f32::<SquaredEuclideanDistance>(path)
        .map_err(|e| PyIOError::new_err(format!("cannot read {path:?}: {e:?}")))?;
    Ok(OwnedInput::Dense {
        dim: data.input_dim(),
        values: data.values().to_vec(),
    })
}

/// Reads a sparse file in the seismic binary format.
pub(crate) fn read_seismic(path: &str) -> PyResult<OwnedInput> {
    let data = read_seismic_format::<u32, f32, DotProduct>(path)
        .map_err(|e| PyIOError::new_err(format!("cannot read {path:?}: {e:?}")))?;
    let (mut components, mut values, mut offsets) = (Vec::new(), Vec::new(), vec![0]);
    for v in data.iter() {
        components.extend_from_slice(v.components());
        values.extend_from_slice(v.values());
        offsets.push(components.len());
    }
    Ok(OwnedInput::Sparse {
        components,
        values,
        offsets,
    })
}

fn dense_dataset<V, D>(input: Input<'_>) -> PyResult<PlainDenseDataset<V, D>>
where
    V: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    let Input::Dense { values, dim } = input else {
        return Err(value_err("a dense stage needs dense data"));
    };
    if dim == 0 || values.is_empty() || values.len() % dim != 0 {
        return Err(value_err(format!(
            "dense data of {} values does not split into vectors of dimension {dim}",
            values.len()
        )));
    }
    let data: Vec<V> = values.iter().map(|&v| V::from_f32_saturating(v)).collect();
    let encoder = PlainDenseQuantizer::<V, D>::new(dim);
    Ok(DenseDataset::from_raw(
        data.into_boxed_slice(),
        values.len() / dim,
        encoder,
    ))
}

fn sparse_dataset<C, V, D>(input: Input<'_>) -> PyResult<PlainSparseDataset<C, V, D>>
where
    C: ComponentType + FromPrimitive + SpaceUsage,
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    let Input::Sparse {
        components,
        values,
        offsets,
    } = input
    else {
        return Err(value_err("a sparse stage needs sparse data"));
    };
    if components.len() != values.len() {
        return Err(value_err(format!(
            "components and values must have the same length, got {} and {}",
            components.len(),
            values.len()
        )));
    }
    check_offsets(offsets, values.len())?;
    if offsets.len() < 2 {
        return Err(value_err("sparse data holds no vectors"));
    }
    let dim = components.iter().max().map_or(1, |&c| c as usize + 1);
    if C::from_usize(dim - 1).is_none() {
        return Err(value_err(format!(
            "component {} does not fit the component type; use u32 components",
            dim - 1
        )));
    }
    let encoder = PlainSparseQuantizer::<C, V, D>::new(dim, dim);
    let mut dataset: PlainSparseDatasetGrowable<C, V, D> =
        DatasetGrowable::with_capacity(encoder, offsets.len() - 1);
    let (mut cs, mut vs) = (Vec::new(), Vec::new());
    for w in offsets.windows(2) {
        cs.clear();
        vs.clear();
        for (c, v) in sorted_pairs(&components[w[0]..w[1]], &values[w[0]..w[1]])? {
            cs.push(C::from_u32(c).expect("checked against the largest component"));
            vs.push(V::from_f32_saturating(v));
        }
        dataset.push(SparseVectorView::new(&cs, &vs));
    }
    Ok(dataset.into())
}

/// Multi-vector input: values, tokens per document, token dimension.
fn multivec_parts(input: Input<'_>) -> PyResult<(&[f32], Vec<usize>, usize)> {
    let Input::MultiVec {
        values,
        token_offsets,
        dim,
    } = input
    else {
        return Err(value_err("a multi-vector stage needs multi-vector data"));
    };
    if dim == 0 || values.len() % dim != 0 {
        return Err(value_err(
            "multi-vector values are not a whole number of tokens",
        ));
    }
    check_offsets(token_offsets, values.len() / dim)?;
    if token_offsets.len() < 2 {
        return Err(value_err("multi-vector data holds no documents"));
    }
    let counts: Vec<usize> = token_offsets.windows(2).map(|w| w[1] - w[0]).collect();
    if counts.contains(&0) {
        return Err(value_err(
            "every multi-vector document needs at least one token",
        ));
    }
    Ok((values, counts, dim))
}

/// All tokens as one dense dataset, to train a multi-vector quantizer on.
fn token_dataset(values: &[f32], dim: usize) -> PlainDenseDataset<f32, SquaredEuclideanDistance> {
    let mut tokens =
        PlainDenseDatasetGrowable::with_capacity(PlainDenseQuantizer::new(dim), values.len() / dim);
    for t in values.chunks_exact(dim) {
        tokens.push(DenseVectorView::new(t));
    }
    tokens.into()
}

/// Product Quantization trains 256 centroids per subspace.
const PQ_MIN_TRAINING: usize = 256;

fn check_training_size(name: &str, n: usize, needed: usize) -> PyResult<()> {
    if n < needed {
        return Err(value_err(format!(
            "{name} trains {needed} centroids and needs at least that many training vectors, \
             got {n}"
        )));
    }
    Ok(())
}

// ----------------------------------------------------------------------------- encodings

/// One stored encoding: its query layout, its dataset type, and how to produce it.
pub(crate) trait Encoding {
    type QueryFormat: QueryFormat;
    type Final: Dataset;

    /// An HNSW graph over `input`, its dataset encoded as `Final`.
    fn hnsw(
        input: Input<'_>,
        enc: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<Self::Final, Graph>>;

    /// The encoded dataset alone, in input order (Flat index, rescoring stage).
    fn dataset(input: Input<'_>, enc: &EncodingConfig<'_>) -> PyResult<Self::Final>;
}

/// Dense, uncompressed `V` values (a cast for `f16`/`bf16`).
pub(crate) struct DensePlain<V, D>(PhantomData<(V, D)>);

impl<V, D> Encoding for DensePlain<V, D>
where
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarDenseSupportedDistance,
{
    type QueryFormat = DenseFormat;
    type Final = PlainDenseDataset<V, D>;

    fn hnsw(
        input: Input<'_>,
        _: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<Self::Final, Graph>> {
        Ok(HNSW::build_index(dense_dataset::<V, D>(input)?, config))
    }

    fn dataset(input: Input<'_>, _: &EncodingConfig<'_>) -> PyResult<Self::Final> {
        dense_dataset::<V, D>(input)
    }
}

/// A dense encoding trained on (or cast from) the plain `f32` vectors.
pub(crate) trait DenseTarget<D: ScalarDenseSupportedDistance>: Dataset + Sized {
    /// Smallest collection the encoder can be trained on.
    const MIN_TRAINING: usize = 1;
    fn encode(source: &PlainDenseDataset<f32, D>, enc: &EncoderConfig) -> Self;
    fn encode_hnsw(
        index: HNSW<PlainDenseDataset<f32, D>, Graph>,
        enc: &EncoderConfig,
    ) -> HNSW<Self, Graph>;
}

fn rabitq_config(enc: &EncoderConfig) -> RabitqConfig {
    match *enc {
        EncoderConfig::Rabitq { seed, rotate, .. } => RabitqConfig { seed, rotate },
        _ => unreachable!("a RaBitQ encoding is only dispatched for a RaBitQ spec"),
    }
}

fn rabitq_ext_config(enc: &EncoderConfig) -> RabitqExtConfig {
    match *enc {
        EncoderConfig::Rabitq { bits, seed, rotate } => RabitqExtConfig {
            total_bits: bits,
            seed,
            rotate,
            ..RabitqExtConfig::default()
        },
        _ => unreachable!("a RaBitQ encoding is only dispatched for a RaBitQ spec"),
    }
}

/// Implements [`DenseTarget`] for a dataset type converted from `&PlainDenseDataset<f32, D>`
/// with the configuration `$cfg` (a function of the encoder spec).
macro_rules! dense_target {
    ([$($gen:tt)*] $target:ty, $cfg_ty:ty, $min:expr, |$enc:ident| $cfg:expr) => {
        impl<$($gen)*> DenseTarget<D> for $target
        where
            D: ScalarDenseSupportedDistance,
            $target: Dataset,
            for<'a> $target: ConvertFrom<&'a PlainDenseDataset<f32, D>, Config = $cfg_ty>,
        {
            const MIN_TRAINING: usize = $min;
            fn encode(source: &PlainDenseDataset<f32, D>, $enc: &EncoderConfig) -> Self {
                <$target>::convert_from(source, $cfg)
            }
            fn encode_hnsw(
                index: HNSW<PlainDenseDataset<f32, D>, Graph>,
                $enc: &EncoderConfig,
            ) -> HNSW<Self, Graph> {
                index.convert_dataset_into_ref($cfg)
            }
        }
    };
}

dense_target!([const M: usize, D: ProductQuantizerDistance] DenseDataset<ProductQuantizer<M, D>>, (), PQ_MIN_TRAINING, |_enc| ());
dense_target!([D: RabitqSupportedDistance] DenseDataset<RabitqQuantizer<D>>, RabitqConfig, 1, |enc| rabitq_config(enc));
dense_target!([D: RabitqSupportedDistance] DenseDataset<RabitqExtQuantizer<D>>, RabitqExtConfig, 1, |enc| rabitq_ext_config(enc));
dense_target!([D] ScalarDenseDataset<f32, FixedU8Q, D>, (), 1, |_enc| ());
dense_target!([D] ScalarDenseDataset<f32, FixedU16Q, D>, (), 1, |_enc| ());

/// Dense, graph built on plain `f32` vectors, then the dataset encoded into `T`.
pub(crate) struct DenseEncoded<D, T>(PhantomData<(D, T)>);

fn dense_source<D: ScalarDenseSupportedDistance, T: DenseTarget<D>>(
    input: Input<'_>,
    enc: &EncodingConfig<'_>,
) -> PyResult<PlainDenseDataset<f32, D>> {
    let source = dense_dataset::<f32, D>(input)?;
    enc.validate_dim(source.input_dim()).map_err(value_err)?;
    check_training_size("this encoder", source.len(), T::MIN_TRAINING)?;
    Ok(source)
}

impl<D, T> Encoding for DenseEncoded<D, T>
where
    D: ScalarDenseSupportedDistance,
    T: DenseTarget<D>,
{
    type QueryFormat = DenseFormat;
    type Final = T;

    fn hnsw(
        input: Input<'_>,
        enc: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<T, Graph>> {
        let source = dense_source::<D, T>(input, enc)?;
        Ok(T::encode_hnsw(
            HNSW::build_index(source, config),
            enc.encoder,
        ))
    }

    fn dataset(input: Input<'_>, enc: &EncodingConfig<'_>) -> PyResult<T> {
        Ok(T::encode(&dense_source::<D, T>(input, enc)?, enc.encoder))
    }
}

/// Sparse, uncompressed `V` values with `C` components.
pub(crate) struct SparsePlain<C, V, D>(PhantomData<(C, V, D)>);

impl<C, V, D> Encoding for SparsePlain<C, V, D>
where
    C: ComponentType + FromPrimitive + SpaceUsage + Sync + 'static,
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    type QueryFormat = SparseFormat<C>;
    type Final = PlainSparseDataset<C, V, D>;

    fn hnsw(
        input: Input<'_>,
        _: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<Self::Final, Graph>> {
        Ok(HNSW::build_index(sparse_dataset::<C, V, D>(input)?, config))
    }

    fn dataset(input: Input<'_>, _: &EncodingConfig<'_>) -> PyResult<Self::Final> {
        sparse_dataset::<C, V, D>(input)
    }
}

/// Sparse, graph built on plain `V` values (`f32`, or `f16` for the blocked layout), then the
/// dataset re-encoded into `T`.
pub(crate) struct SparseEncoded<C, V, D, T>(PhantomData<(C, V, D, T)>);

impl<C, V, D, T> Encoding for SparseEncoded<C, V, D, T>
where
    C: ComponentType + FromPrimitive + SpaceUsage + Sync + 'static,
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarSparseSupportedDistance,
    T: Dataset + ConvertFrom<PlainSparseDataset<C, V, D>, Config = ()>,
{
    type QueryFormat = SparseFormat<C>;
    type Final = T;

    fn hnsw(
        input: Input<'_>,
        _: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<T, Graph>> {
        let plain: HNSW<_, Graph> = HNSW::build_index(sparse_dataset::<C, V, D>(input)?, config);
        Ok(plain.convert_dataset_into(()))
    }

    fn dataset(input: Input<'_>, _: &EncodingConfig<'_>) -> PyResult<T> {
        Ok(T::convert_from(sparse_dataset::<C, V, D>(input)?, ()))
    }
}

/// Multi-vector, uncompressed `V` token values.
#[cfg(feature = "multivec")]
pub(crate) struct MvPlain<V>(PhantomData<V>);

#[cfg(feature = "multivec")]
impl<V> Encoding for MvPlain<V>
where
    V: ValueType + Float + FromF32 + SpaceUsage + Send + Sync,
{
    type QueryFormat = MultiVecFormat;
    type Final = MultiVectorDataset<PlainMultiVecQuantizer<V>>;

    fn hnsw(
        input: Input<'_>,
        enc: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<Self::Final, Graph>> {
        Ok(HNSW::build_index(Self::dataset(input, enc)?, config))
    }

    fn dataset(input: Input<'_>, _: &EncodingConfig<'_>) -> PyResult<Self::Final> {
        let (values, counts, dim) = multivec_parts(input)?;
        let values: Vec<V> = values.iter().map(|&v| V::from_f32_saturating(v)).collect();
        Ok(MultiVectorDataset::from_flat_par(
            PlainMultiVecQuantizer::<V>::new(dim),
            &values,
            &counts,
        ))
    }
}

/// Multi-vector Product Quantization with `M` subspaces per token.
#[cfg(feature = "multivec")]
pub(crate) struct MvPq<const M: usize>;

#[cfg(feature = "multivec")]
impl<const M: usize> Encoding for MvPq<M> {
    type QueryFormat = MultiVecFormat;
    type Final = MultiVectorDataset<MultiVecProductQuantizer<M, f32>>;

    fn hnsw(
        input: Input<'_>,
        enc: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<Self::Final, Graph>> {
        Ok(HNSW::build_index(Self::dataset(input, enc)?, config))
    }

    fn dataset(input: Input<'_>, enc: &EncodingConfig<'_>) -> PyResult<Self::Final> {
        let (values, counts, dim) = multivec_parts(input)?;
        enc.validate_dim(dim).map_err(value_err)?;
        check_training_size("PQ", values.len() / dim, PQ_MIN_TRAINING)?;
        let encoder = MultiVecProductQuantizer::<M, f32>::train(&token_dataset(values, dim));
        Ok(MultiVectorDataset::from_flat_par(encoder, values, &counts))
    }
}

/// Multi-vector two-level PQ with `M` subspaces per token residual.
#[cfg(feature = "multivec")]
pub(crate) struct MvTwoLevelPq<const M: usize>;

#[cfg(feature = "multivec")]
impl<const M: usize> Encoding for MvTwoLevelPq<M> {
    type QueryFormat = MultiVecFormat;
    type Final = MultiVectorDataset<MultiVecTwoLevelProductQuantizer<M, f32>>;

    fn hnsw(
        input: Input<'_>,
        enc: &EncodingConfig<'_>,
        config: &HNSWBuildConfiguration,
    ) -> PyResult<HNSW<Self::Final, Graph>> {
        Ok(HNSW::build_index(Self::dataset(input, enc)?, config))
    }

    fn dataset(input: Input<'_>, enc: &EncodingConfig<'_>) -> PyResult<Self::Final> {
        let EncoderConfig::TwoLevelPq { n_coarse, .. } = *enc.encoder else {
            unreachable!("a two-level PQ encoding is only dispatched for a two-level PQ spec")
        };
        let (values, counts, dim) = multivec_parts(input)?;
        enc.validate_dim(dim).map_err(value_err)?;
        let n_tokens = values.len() / dim;
        check_training_size("TwoLevelPQ", n_tokens, PQ_MIN_TRAINING.max(n_coarse))?;
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f32>::train(
            &token_dataset(values, dim),
            n_coarse,
        );
        Ok(MultiVectorDataset::from_flat_par(encoder, values, &counts))
    }
}

// ----------------------------------------------------------------------------- dispatch

/// Runs `$body` with `$d` bound, as a type alias, to the distance type of `$metric`
/// (`maxsim` never reaches this macro: multi-vector encodings set their own distance).
macro_rules! with_metric {
    ($metric:expr, |$d:ident| $body:expr) => {
        match $metric {
            $crate::pylib::composable::config::Metric::L2 => {
                type $d = SquaredEuclideanDistance;
                $body
            }
            _ => {
                type $d = DotProduct;
                $body
            }
        }
    };
}

/// Runs `$body` with `$c` bound to the component type and `$d` to the distance type.
macro_rules! with_component_and_metric {
    ($enc:expr, |$c:ident, $d:ident| $body:expr) => {
        match $enc.component_type {
            Some($crate::pylib::composable::config::ComponentType::U32) => {
                type $c = u32;
                with_metric!($enc.metric, |$d| $body)
            }
            _ => {
                type $c = u16;
                with_metric!($enc.metric, |$d| $body)
            }
        }
    };
}

/// Runs `$body` with `$e` bound to `DenseEncoded<D, PQ<M>>` for the spec's metric and
/// subspace count.
macro_rules! with_pq {
    ($subspaces:expr, $metric:expr, |$e:ident| $body:expr) => {
        with_metric!($metric, |D| match $subspaces {
            8 => { type $e = DenseEncoded<D, DenseDataset<ProductQuantizer<8, D>>>; $body }
            16 => { type $e = DenseEncoded<D, DenseDataset<ProductQuantizer<16, D>>>; $body }
            32 => { type $e = DenseEncoded<D, DenseDataset<ProductQuantizer<32, D>>>; $body }
            64 => { type $e = DenseEncoded<D, DenseDataset<ProductQuantizer<64, D>>>; $body }
            128 => { type $e = DenseEncoded<D, DenseDataset<ProductQuantizer<128, D>>>; $body }
            other => Err(value_err(format!("PQ(subspaces={other}) is not compiled in"))),
        })
    };
}

/// Runs `$body` with `$e` bound to `$ty<M>` for the multi-vector subspace count.
macro_rules! with_mv_subspaces {
    ($subspaces:expr, $ty:ident, |$e:ident| $body:expr) => {
        match $subspaces {
            8 => {
                type $e = $ty<8>;
                $body
            }
            16 => {
                type $e = $ty<16>;
                $body
            }
            32 => {
                type $e = $ty<32>;
                $body
            }
            64 => {
                type $e = $ty<64>;
                $body
            }
            other => Err(value_err(format!(
                "subspaces={other} is not compiled in for multi-vector data"
            ))),
        }
    };
}

/// The encoding table: runs `$body` with `$e` bound to the [`Encoding`] type a (validated)
/// encoding spec names.
macro_rules! with_encoding {
    ($enc:expr, |$e:ident| $body:expr) => {{
        use $crate::pylib::composable::config::{
            BlockedValues as BV, ComponentType as CT, Dp8Values as DV, EncoderConfig as E,
            Representation as Rep, ValueDtype as VD,
        };
        use vectorium::{
            BlockedSparseDataset, BlockedSparseScalarU8Dataset, BlockedSparseScalarU16Dataset,
            BridgedDotPacking8U32ScalarU8Dataset, BridgedDotPacking8U32ScalarU16Dataset,
            DotPacking8U16F16Dataset, DotPacking8U16FixedU8Dataset,
            DotPacking8U16Scalar4BitDataset, DotPacking8U16ScalarU8Dataset,
            DotPacking8U16ScalarU16Dataset, ScalarU8SparseDataset, ScalarU16SparseDataset,
        };
        let enc: &EncodingConfig<'_> = $enc;
        match (enc.representation, enc.encoder) {
            // ---- dense ----
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Plain { dtype: VD::F32 }) => {
                with_metric!(enc.metric, |D| { type $e = DensePlain<f32, D>; $body })
            }
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Plain { dtype: VD::F16 }) => {
                with_metric!(enc.metric, |D| { type $e = DensePlain<f16, D>; $body })
            }
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Plain { dtype: VD::Bf16 }) => {
                with_metric!(enc.metric, |D| { type $e = DensePlain<bf16, D>; $body })
            }
            #[cfg(feature = "dense")]
            (Rep::Dense, E::FixedPoint { bits: 8 }) => with_metric!(enc.metric, |D| {
                type $e = DenseEncoded<D, ScalarDenseDataset<f32, FixedU8Q, D>>;
                $body
            }),
            #[cfg(feature = "dense")]
            (Rep::Dense, E::FixedPoint { .. }) => with_metric!(enc.metric, |D| {
                type $e = DenseEncoded<D, ScalarDenseDataset<f32, FixedU16Q, D>>;
                $body
            }),
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Pq { subspaces }) => with_pq!(*subspaces, enc.metric, |$e| $body),
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Rabitq { bits: 1, .. }) => with_metric!(enc.metric, |D| {
                type $e = DenseEncoded<D, DenseDataset<RabitqQuantizer<D>>>;
                $body
            }),
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Rabitq { .. }) => with_metric!(enc.metric, |D| {
                type $e = DenseEncoded<D, DenseDataset<RabitqExtQuantizer<D>>>;
                $body
            }),
            // ---- sparse, any component type and metric ----
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Plain { dtype: VD::F16 }) => {
                with_component_and_metric!(enc, |C, D| { type $e = SparsePlain<C, f16, D>; $body })
            }
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Plain { .. }) => {
                with_component_and_metric!(enc, |C, D| { type $e = SparsePlain<C, f32, D>; $body })
            }
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::FixedPoint { bits: 8 }) => with_component_and_metric!(enc, |C, D| {
                type $e = SparseEncoded<C, f32, D, ScalarSparseDataset<C, f32, FixedU8Q, D>>;
                $body
            }),
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::FixedPoint { .. }) => with_component_and_metric!(enc, |C, D| {
                type $e = SparseEncoded<C, f32, D, ScalarSparseDataset<C, f32, FixedU16Q, D>>;
                $body
            }),
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Scalar { bits: 8 }) => with_component_and_metric!(enc, |C, D| {
                type $e = SparseEncoded<C, f32, D, ScalarU8SparseDataset<C, D>>;
                $body
            }),
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Scalar { .. }) => with_component_and_metric!(enc, |C, D| {
                type $e = SparseEncoded<C, f32, D, ScalarU16SparseDataset<C, D>>;
                $body
            }),
            // ---- sparse, inner product and (mostly) u16 only: checked by validation ----
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Blocked { values: BV::F16 }) => {
                type $e = SparseEncoded<u16, f16, DotProduct, BlockedSparseDataset>;
                $body
            }
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Blocked { values: BV::ScalarU8 }) => {
                type $e = SparseEncoded<u16, f32, DotProduct, BlockedSparseScalarU8Dataset>;
                $body
            }
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Blocked { values: BV::ScalarU16 }) => {
                type $e = SparseEncoded<u16, f32, DotProduct, BlockedSparseScalarU16Dataset>;
                $body
            }
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::DotPacking8 { values }) => match (values, enc.component_type) {
                (DV::FixedU8, _) => {
                    type $e = SparseEncoded<u16, f32, DotProduct, DotPacking8U16FixedU8Dataset>;
                    $body
                }
                (DV::ScalarU8, Some(CT::U32)) => {
                    type $e =
                        SparseEncoded<u32, f32, DotProduct, BridgedDotPacking8U32ScalarU8Dataset>;
                    $body
                }
                (DV::ScalarU8, _) => {
                    type $e = SparseEncoded<u16, f32, DotProduct, DotPacking8U16ScalarU8Dataset>;
                    $body
                }
                (DV::ScalarU16, Some(CT::U32)) => {
                    type $e =
                        SparseEncoded<u32, f32, DotProduct, BridgedDotPacking8U32ScalarU16Dataset>;
                    $body
                }
                (DV::ScalarU16, _) => {
                    type $e = SparseEncoded<u16, f32, DotProduct, DotPacking8U16ScalarU16Dataset>;
                    $body
                }
                (DV::F16, _) => {
                    type $e = SparseEncoded<u16, f32, DotProduct, DotPacking8U16F16Dataset>;
                    $body
                }
                (DV::Scalar4Bit, _) => {
                    type $e = SparseEncoded<u16, f32, DotProduct, DotPacking8U16Scalar4BitDataset>;
                    $body
                }
            },
            // ---- multi-vector (MaxSim) ----
            #[cfg(feature = "multivec")]
            (Rep::Multivector, E::Plain { dtype: VD::F16 }) => {
                type $e = MvPlain<f16>;
                $body
            }
            #[cfg(feature = "multivec")]
            (Rep::Multivector, E::Plain { .. }) => {
                type $e = MvPlain<f32>;
                $body
            }
            #[cfg(feature = "multivec")]
            (Rep::Multivector, E::Pq { subspaces }) => {
                with_mv_subspaces!(*subspaces, MvPq, |$e| $body)
            }
            #[cfg(feature = "multivec")]
            (Rep::Multivector, E::TwoLevelPq { subspaces, .. }) => {
                with_mv_subspaces!(*subspaces, MvTwoLevelPq, |$e| $body)
            }
            _ => Err(value_err(
                "unsupported encoding (validation should have rejected it)",
            )),
        }
    }};
}

// ----------------------------------------------------------------------------- HNSW and Flat

type StreamVByteGraph = GenericGraph<StreamVByteNeighbors>;

const HNSW_ACCEPTS: Accepts = Accepts {
    ef_search: true,
    early_exit_threshold: true,
    nprobe: false,
};
const FLAT_ACCEPTS: Accepts = Accepts {
    ef_search: false,
    early_exit_threshold: false,
    nprobe: false,
};

fn corrupt(e: String) -> pyo3::PyErr {
    PyIOError::new_err(format!("corrupt index payload: {e}"))
}

/// Builds a Flat or HNSW first stage over encoding `E`.
///
/// For the `permuted`/`compressed` graphs, the dataset is encoded *before* the EGB permutation,
/// as in `build_permuted_and_save` in `src/bin/hnsw_build.rs`. Encoders fitted to the data are
/// thus trained in the same way as on the `standard` path. The permutation then copies the
/// encoded rows.
fn build_graph_or_flat<E: Encoding>(
    input: Input<'_>,
    stage: &StageConfig,
) -> PyResult<Box<dyn AnyIndex>>
where
    E::Final: Dataset<Owned = E::Final> + 'static,
    SingleStage<HNSW<E::Final, Graph>, E::QueryFormat>: AnyIndex,
    SingleStage<HNSW<E::Final, StreamVByteGraph>, E::QueryFormat>: AnyIndex,
    SingleStage<FlatIndex<E::Final>, E::QueryFormat>: AnyIndex,
{
    let enc = stage.encoding();
    Ok(match stage.index {
        IndexStructureConfig::Flat => Box::new(SingleStage::<_, E::QueryFormat>::new(
            FlatIndex::from(E::dataset(input, &enc)?),
            FLAT_ACCEPTS,
        )),
        IndexStructureConfig::Hnsw {
            m,
            ef_construction,
            graph,
        } => {
            let config = HNSWBuildConfiguration::default()
                .with_num_neighbors(m)
                .with_ef_construction(ef_construction);
            let built = E::hnsw(input, &enc, &config)?;
            match graph {
                GraphKind::Standard => {
                    Box::new(SingleStage::<_, E::QueryFormat>::new(built, HNSW_ACCEPTS))
                }
                GraphKind::Permuted => Box::new(SingleStage::<_, E::QueryFormat>::new(
                    built.permute_and_encode::<PlainNeighbors>(),
                    HNSW_ACCEPTS,
                )),
                GraphKind::Compressed => Box::new(SingleStage::<_, E::QueryFormat>::new(
                    built.permute_and_encode::<StreamVByteNeighbors>(),
                    HNSW_ACCEPTS,
                )),
            }
        }
        IndexStructureConfig::Ivf { .. } => unreachable!("IVF is dispatched by with_ivf!"),
    })
}

fn load_graph_or_flat<E: Encoding>(
    reader: &mut dyn Read,
    stage: &StageConfig,
) -> PyResult<Box<dyn AnyIndex>>
where
    E::Final: 'static,
    HNSW<E::Final, Graph>: DeserializeOwned,
    HNSW<E::Final, StreamVByteGraph>: DeserializeOwned,
    FlatIndex<E::Final>: DeserializeOwned,
    SingleStage<HNSW<E::Final, Graph>, E::QueryFormat>: AnyIndex,
    SingleStage<HNSW<E::Final, StreamVByteGraph>, E::QueryFormat>: AnyIndex,
    SingleStage<FlatIndex<E::Final>, E::QueryFormat>: AnyIndex,
{
    Ok(match stage.index {
        IndexStructureConfig::Flat => Box::new(SingleStage::<_, E::QueryFormat>::new(
            read_payload::<FlatIndex<E::Final>>(reader).map_err(corrupt)?,
            FLAT_ACCEPTS,
        )),
        IndexStructureConfig::Hnsw {
            graph: GraphKind::Compressed,
            ..
        } => Box::new(SingleStage::<_, E::QueryFormat>::new(
            read_payload::<HNSW<E::Final, StreamVByteGraph>>(reader).map_err(corrupt)?,
            HNSW_ACCEPTS,
        )),
        // `standard` and `permuted` share a Rust type: the permutation lives in the index data.
        IndexStructureConfig::Hnsw { .. } => Box::new(SingleStage::<_, E::QueryFormat>::new(
            read_payload::<HNSW<E::Final, Graph>>(reader).map_err(corrupt)?,
            HNSW_ACCEPTS,
        )),
        IndexStructureConfig::Ivf { .. } => unreachable!("IVF is dispatched by with_ivf!"),
    })
}

// ----------------------------------------------------------------------------- rescoring

fn build_rerank<E: Encoding>(
    input: Input<'_>,
    enc: &EncodingConfig<'_>,
) -> PyResult<Box<dyn AnyRescorer>>
where
    E::Final: 'static,
    Rescorer<E::Final, E::QueryFormat>: AnyRescorer,
{
    Ok(Box::new(Rescorer::<_, E::QueryFormat>::new(E::dataset(
        input, enc,
    )?)))
}

fn load_rerank<E: Encoding>(reader: &mut dyn Read) -> PyResult<Box<dyn AnyRescorer>>
where
    E::Final: DeserializeOwned + 'static,
    Rescorer<E::Final, E::QueryFormat>: AnyRescorer,
{
    Ok(Box::new(Rescorer::<_, E::QueryFormat>::new(
        read_payload::<E::Final>(reader).map_err(corrupt)?,
    )))
}

// ----------------------------------------------------------------------------- IVF

/// Dense IVF centroids: `f16`, under the reported metric `M` (as `ivf_build` stores them).
type DenseCentroids<M> = ScalarDenseDataset<f32, f16, M>;
/// Sparse IVF centroids: `f32`, `u16` components, dot product.
type SparseCentroids = PlainSparseDataset<u16, f32, DotProduct>;

/// The dataset an IVF is built from, before its clusters are encoded.
struct IvfParts<S, DQ> {
    /// The vectors (or residuals) in cluster order, in dot-product form.
    stored: S,
    centroids: DQ,
    cluster_offsets: Box<[u32]>,
    external_ids: Box<[u32]>,
    residual: bool,
}

fn kmeans_builder(stage: &StageConfig) -> (usize, KMeansBuilder, Option<usize>) {
    let IndexStructureConfig::Ivf {
        n_lists,
        ref kmeans,
        ..
    } = stage.index
    else {
        unreachable!("an IVF recipe is only dispatched for an IVF spec")
    };
    let mut b = KMeansBuilder::new()
        .n_redo(kmeans.n_redo)
        .seed(kmeans.seed)
        .spherical(kmeans.spherical);
    if let Some(e) = kmeans.epochs {
        b = b.epochs(e);
    }
    (n_lists, b, kmeans.sample_size)
}

fn check_n_lists(n_lists: usize, n: usize) -> PyResult<()> {
    if n_lists > n {
        return Err(value_err(format!(
            "IVF(n_lists={n_lists}) needs at least as many vectors, got {n}"
        )));
    }
    Ok(())
}

/// k-means and cluster assignment for one representation, following `ivf_build`.
trait IvfSource: Dataset + Sized {
    type QueryFormat: QueryFormat;
    type Stored: Dataset;
    fn convert(input: Input<'_>, stage: &StageConfig) -> PyResult<IvfParts<Self::Stored, Self>>;
}

/// Dense: cluster under squared Euclidean, route with `f16` centroids under the reported `M`.
impl<M> IvfSource for DenseCentroids<M>
where
    M: ScalarDenseSupportedDistance + ReportedMetric,
    PlainDenseDataset<f32, M>: Into<PlainDenseDataset<f32, DotProduct>>,
    DenseCentroids<SquaredEuclideanDistance>: Into<DenseCentroids<M>>,
{
    type QueryFormat = DenseFormat;
    type Stored = PlainDenseDataset<f32, DotProduct>;

    fn convert(input: Input<'_>, stage: &StageConfig) -> PyResult<IvfParts<Self::Stored, Self>> {
        let IndexStructureConfig::Ivf { residual, .. } = stage.index else {
            unreachable!()
        };
        let data = dense_dataset::<f32, M>(input)?;
        stage
            .encoding()
            .validate_dim(data.input_dim())
            .map_err(value_err)?;
        if let EncoderConfig::Pq { .. } = stage.encoder {
            check_training_size("PQ", data.len(), PQ_MIN_TRAINING)?;
        }
        let (n_lists, kmeans, sample) = kmeans_builder(stage);
        check_n_lists(n_lists, data.len())?;
        let dim = data.input_dim();
        let a = prepare_ivf::<_, PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>>(
            &data,
            n_lists,
            kmeans,
            ScalarDenseQuantizer::new(dim),
            sample,
            AnnAssignment::Auto.enabled(n_lists),
            |_, _| {},
        );
        let stored: Self::Stored = if residual {
            a.residuals::<_, PlainDenseDatasetGrowable<f32, DotProduct>>(
                &data,
                ScalarDenseQuantizer::new(dim),
            )
        } else {
            a.reorder(&data).into()
        };
        let centroids: Self =
            DenseCentroids::<SquaredEuclideanDistance>::convert_from(&a.centroids, ()).into();
        Ok(IvfParts {
            stored,
            centroids,
            cluster_offsets: a.cluster_offsets,
            external_ids: a.external_ids,
            residual,
        })
    }
}

/// Sparse: cluster under inner product, route with the `f32` centroids, store the vectors.
impl IvfSource for SparseCentroids {
    type QueryFormat = SparseFormat<u16>;
    type Stored = PlainSparseDataset<u16, f32, DotProduct>;

    fn convert(input: Input<'_>, stage: &StageConfig) -> PyResult<IvfParts<Self::Stored, Self>> {
        let IndexStructureConfig::Ivf {
            centroid_nnz_factor,
            ..
        } = stage.index
        else {
            unreachable!()
        };
        let data = sparse_dataset::<u16, f32, DotProduct>(input)?;
        let (n_lists, kmeans, sample) = kmeans_builder(stage);
        check_n_lists(n_lists, data.len())?;
        let prune = centroid_nnz_factor.map(|f| vectorium::sparse_centroid_pruning(&data, f));
        let a = prepare_ivf::<_, PlainSparseDatasetGrowable<u16, f32, DotProduct>>(
            &data,
            n_lists,
            kmeans,
            PlainSparseQuantizer::new(data.input_dim(), data.input_dim()),
            sample,
            AnnAssignment::Auto.enabled(n_lists),
            |members, centroid| {
                if let Some(prune) = &prune {
                    prune(members, centroid)
                }
            },
        );
        let stored = a.reorder(&data);
        Ok(IvfParts {
            stored,
            centroids: a.centroids,
            cluster_offsets: a.cluster_offsets,
            external_ids: a.external_ids,
            residual: false,
        })
    }
}

/// How an IVF cluster encoding is configured: its conversion config, and the query
/// parameters `encode_clusters` scores the L2 correction with.
trait ClusterEncoding: Dataset {
    type Config;
    fn config(enc: &EncoderConfig) -> Self::Config;
    fn correction_query_params() -> <Self::Encoder as VectorEncoder>::QueryParams;
}

macro_rules! cluster_encoding {
    ([$($gen:tt)*] $ty:ty, $cfg:ty, |$enc:ident| $config:expr) => {
        impl<$($gen)*> ClusterEncoding for $ty
        where
            $ty: Dataset,
            <<$ty as Dataset>::Encoder as VectorEncoder>::QueryParams: Default,
        {
            type Config = $cfg;
            fn config($enc: &EncoderConfig) -> $cfg {
                $config
            }
            fn correction_query_params(
            ) -> <<$ty as Dataset>::Encoder as VectorEncoder>::QueryParams {
                Default::default()
            }
        }
    };
}

// The L2 correction is scored with the encoder's default query parameters (1-bit queries for
// 1-bit RaBitQ), as `ivf_build` does with its defaults.
cluster_encoding!([] PlainDenseDataset<f32, DotProduct>, (), |_e| ());
cluster_encoding!([] ScalarDenseDataset<f32, f16, DotProduct>, (), |_e| ());
cluster_encoding!([] DenseDataset<ProductQuantizer<8, DotProduct>>, (), |_e| ());
cluster_encoding!([] DenseDataset<ProductQuantizer<16, DotProduct>>, (), |_e| ());
cluster_encoding!([] DenseDataset<ProductQuantizer<32, DotProduct>>, (), |_e| ());
cluster_encoding!([] DenseDataset<ProductQuantizer<64, DotProduct>>, (), |_e| ());
cluster_encoding!([] DenseDataset<ProductQuantizer<128, DotProduct>>, (), |_e| ());
cluster_encoding!([] DenseDataset<RabitqQuantizer<DotProduct>>, RabitqConfig, |e| rabitq_config(e));
cluster_encoding!([] DenseDataset<RabitqExtQuantizer<DotProduct>>, RabitqExtConfig, |e| rabitq_ext_config(e));
cluster_encoding!([] ScalarSparseDataset<u16, f32, f32, DotProduct>, (), |_e| ());
cluster_encoding!([] ScalarSparseDataset<u16, f32, f16, DotProduct>, (), |_e| ());

fn ivf_accepts(centroids: &CentroidIndexConfig) -> Accepts {
    let base = match centroids {
        CentroidIndexConfig::Flat => FLAT_ACCEPTS,
        CentroidIndexConfig::Hnsw { .. } => HNSW_ACCEPTS,
    };
    Accepts {
        nprobe: true,
        ..base
    }
}

/// Builds an IVF reporting `DQ`'s metric with clusters stored as `DC`.
fn build_ivf<DQ, DC>(input: Input<'_>, stage: &StageConfig) -> PyResult<Box<dyn AnyIndex>>
where
    DQ: IvfSource + vectorium::MeanDataset + Sync + SpaceUsage + 'static,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    DC: ClusterEncoding
        + for<'a> ConvertFrom<&'a DQ::Stored, Config = <DC as ClusterEncoding>::Config>
        + 'static,
    DC::Encoder: WithDistance<SquaredEuclideanDistance>,
    for<'a> <DC::Encoder as VectorEncoder>::QueryVector<'a>:
        From<&'a <DQ as vectorium::MeanDataset>::Vector>,
    FlatIndex<DQ>: vectorium::Index,
    HNSW<DQ, Graph>: vectorium::Index,
    SingleStage<IVF<DQ, DC, FlatIndex<DQ>>, DQ::QueryFormat>: AnyIndex,
    SingleStage<IVF<DQ, DC, HNSW<DQ, Graph>>, DQ::QueryFormat>: AnyIndex,
{
    let IndexStructureConfig::Ivf { ref centroids, .. } = stage.index else {
        unreachable!()
    };
    let parts = DQ::convert(input, stage)?;
    // The correction terms read the stored centroids, so they match what the search reads.
    let (cluster, correction) =
        encode_clusters::<<DQ::Encoder as VectorEncoder>::Distance, _, DC, _, _>(
            &parts.stored,
            DC::config(&stage.encoder),
            &DC::correction_query_params(),
            &parts.centroids,
            &parts.cluster_offsets,
            parts.residual,
        );
    drop(parts.stored);
    let accepts = ivf_accepts(centroids);
    Ok(match *centroids {
        CentroidIndexConfig::Flat => Box::new(SingleStage::<_, DQ::QueryFormat>::new(
            IVF::<DQ, DC, _>::from_parts(
                FlatIndex::from(parts.centroids),
                cluster,
                parts.cluster_offsets,
                parts.external_ids,
                correction,
                parts.residual,
            ),
            accepts,
        )),
        CentroidIndexConfig::Hnsw { m, ef_construction } => {
            let cfg = HNSWBuildConfiguration::default()
                .with_num_neighbors(m)
                .with_ef_construction(ef_construction);
            let ci: HNSW<DQ, Graph> = HNSW::build_index(parts.centroids, &cfg);
            Box::new(SingleStage::<_, DQ::QueryFormat>::new(
                IVF::<DQ, DC, _>::from_parts(
                    ci,
                    cluster,
                    parts.cluster_offsets,
                    parts.external_ids,
                    correction,
                    parts.residual,
                ),
                accepts,
            ))
        }
    })
}

fn load_ivf<DQ, DC>(reader: &mut dyn Read, stage: &StageConfig) -> PyResult<Box<dyn AnyIndex>>
where
    DQ: IvfSource + 'static,
    DC: Dataset + 'static,
    FlatIndex<DQ>: vectorium::Index,
    HNSW<DQ, Graph>: vectorium::Index,
    IVF<DQ, DC, FlatIndex<DQ>>: DeserializeOwned,
    IVF<DQ, DC, HNSW<DQ, Graph>>: DeserializeOwned,
    SingleStage<IVF<DQ, DC, FlatIndex<DQ>>, DQ::QueryFormat>: AnyIndex,
    SingleStage<IVF<DQ, DC, HNSW<DQ, Graph>>, DQ::QueryFormat>: AnyIndex,
{
    let IndexStructureConfig::Ivf { ref centroids, .. } = stage.index else {
        unreachable!()
    };
    let accepts = ivf_accepts(centroids);
    Ok(match centroids {
        CentroidIndexConfig::Flat => Box::new(SingleStage::<_, DQ::QueryFormat>::new(
            read_payload::<IVF<DQ, DC, FlatIndex<DQ>>>(reader).map_err(corrupt)?,
            accepts,
        )),
        CentroidIndexConfig::Hnsw { .. } => Box::new(SingleStage::<_, DQ::QueryFormat>::new(
            read_payload::<IVF<DQ, DC, HNSW<DQ, Graph>>>(reader).map_err(corrupt)?,
            accepts,
        )),
    })
}

/// Runs `$body` with `$dq`/`$dc` bound to `DenseCentroids<$m>` and PQ clusters of `$s`
/// subspaces.
macro_rules! ivf_dense_pq {
    ($m:ident, $s:expr, |$dq:ident, $dc:ident| $body:expr) => {
        match $s {
            8 => {
                type $dq = DenseCentroids<$m>;
                type $dc = DenseDataset<ProductQuantizer<8, DotProduct>>;
                $body
            }
            16 => {
                type $dq = DenseCentroids<$m>;
                type $dc = DenseDataset<ProductQuantizer<16, DotProduct>>;
                $body
            }
            32 => {
                type $dq = DenseCentroids<$m>;
                type $dc = DenseDataset<ProductQuantizer<32, DotProduct>>;
                $body
            }
            64 => {
                type $dq = DenseCentroids<$m>;
                type $dc = DenseDataset<ProductQuantizer<64, DotProduct>>;
                $body
            }
            128 => {
                type $dq = DenseCentroids<$m>;
                type $dc = DenseDataset<ProductQuantizer<128, DotProduct>>;
                $body
            }
            other => Err(value_err(format!(
                "PQ(subspaces={other}) is not compiled in"
            ))),
        }
    };
}

/// The IVF table: runs `$body` with `$dq` bound to the centroid dataset and `$dc` to the
/// cluster dataset a (validated) IVF stage names.
macro_rules! with_ivf {
    ($stage:expr, |$dq:ident, $dc:ident| $body:expr) => {{
        use $crate::pylib::composable::config::{
            EncoderConfig as E, Representation as Rep, ValueDtype as VD,
        };
        let stage: &StageConfig = $stage;
        match (stage.representation, &stage.encoder) {
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Plain { dtype: VD::F16 }) => with_metric!(stage.metric, |M| {
                type $dq = DenseCentroids<M>;
                type $dc = ScalarDenseDataset<f32, f16, DotProduct>;
                $body
            }),
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Plain { .. }) => with_metric!(stage.metric, |M| {
                type $dq = DenseCentroids<M>;
                type $dc = PlainDenseDataset<f32, DotProduct>;
                $body
            }),
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Pq { subspaces }) => {
                with_metric!(stage.metric, |M| ivf_dense_pq!(
                    M,
                    *subspaces,
                    |$dq, $dc| $body
                ))
            }
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Rabitq { bits: 1, .. }) => with_metric!(stage.metric, |M| {
                type $dq = DenseCentroids<M>;
                type $dc = DenseDataset<RabitqQuantizer<DotProduct>>;
                $body
            }),
            #[cfg(feature = "dense")]
            (Rep::Dense, E::Rabitq { .. }) => with_metric!(stage.metric, |M| {
                type $dq = DenseCentroids<M>;
                type $dc = DenseDataset<RabitqExtQuantizer<DotProduct>>;
                $body
            }),
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Plain { dtype: VD::F16 }) => {
                type $dq = SparseCentroids;
                type $dc = ScalarSparseDataset<u16, f32, f16, DotProduct>;
                $body
            }
            #[cfg(feature = "sparse")]
            (Rep::Sparse, E::Plain { .. }) => {
                type $dq = SparseCentroids;
                type $dc = ScalarSparseDataset<u16, f32, f32, DotProduct>;
                $body
            }
            _ => Err(value_err(
                "unsupported IVF (validation should have rejected it)",
            )),
        }
    }};
}

// ----------------------------------------------------------------------------- residuals

/// The first-stage input after its transform, and what the transform left out.
fn transform_first_stage(
    input: Input<'_>,
    stage: &StageConfig,
) -> Option<(OwnedInput, OwnedInput)> {
    let Some(TransformConfig::L1Prune { percent }) = stage.transform else {
        return None;
    };
    let Input::Sparse {
        components,
        values,
        offsets,
    } = input
    else {
        return None; // validation restricts L1Prune to sparse data
    };
    let mut kept = (Vec::new(), Vec::new(), vec![0usize]);
    let mut rest = (Vec::new(), Vec::new(), vec![0usize]);
    for w in offsets.windows(2) {
        let v = SparseVectorView::new(&components[w[0]..w[1]], &values[w[0]..w[1]]);
        let (k, r) = l1_prune_split_vector(v, percent);
        kept.0.extend_from_slice(k.components());
        kept.1.extend_from_slice(k.values());
        kept.2.push(kept.0.len());
        rest.0.extend_from_slice(r.components());
        rest.1.extend_from_slice(r.values());
        rest.2.push(rest.0.len());
    }
    let to_input = |(components, values, offsets)| OwnedInput::Sparse {
        components,
        values,
        offsets,
    };
    Some((to_input(kept), to_input(rest)))
}

/// `x - x̂` for every dense vector, where `x̂` is its reconstruction by the first stage's RaBitQ
/// encoder. The codes are recomputed with the same configuration and seed, so they are the
/// ones the first stage stores.
fn dense_rabitq_residual(input: Input<'_>, first: &StageConfig) -> PyResult<OwnedInput> {
    use rayon::prelude::*;

    fn residual<T>(
        plain: &PlainDenseDataset<f32, DotProduct>,
        enc: &EncoderConfig,
        reconstruct: impl Fn(&T, usize) -> Vec<f32> + Sync,
    ) -> Vec<f32>
    where
        T: DenseTarget<DotProduct> + Sync,
    {
        let codes = T::encode(plain, enc);
        let dim = plain.input_dim();
        let raw = plain.values();
        let mut out = vec![0f32; raw.len()];
        out.par_chunks_mut(dim).enumerate().for_each(|(id, o)| {
            let x_hat = reconstruct(&codes, id);
            for ((o, &x), h) in o.iter_mut().zip(&raw[id * dim..(id + 1) * dim]).zip(x_hat) {
                *o = x - h;
            }
        });
        out
    }

    let plain = dense_dataset::<f32, DotProduct>(input)?;
    let enc = first.encoding();
    enc.validate_dim(plain.input_dim()).map_err(value_err)?;
    let dim = plain.input_dim();
    let values = match first.encoder {
        EncoderConfig::Rabitq { bits: 1, .. } => residual::<
            DenseDataset<RabitqQuantizer<DotProduct>>,
        >(&plain, &first.encoder, |c, id| {
            c.encoder().reconstruct(c.get(id as u64).values())
        }),
        EncoderConfig::Rabitq { .. } => residual::<DenseDataset<RabitqExtQuantizer<DotProduct>>>(
            &plain,
            &first.encoder,
            |c, id| c.encoder().reconstruct(c.get(id as u64).values()),
        ),
        _ => unreachable!("validation restricts dense residuals to a RaBitQ first stage"),
    };
    Ok(OwnedInput::Dense { values, dim })
}

// ----------------------------------------------------------------------------- assembly

#[cfg(feature = "ivf")]
// With `ivf` but neither `dense` nor `sparse`, the table has no arms left.
#[cfg_attr(
    not(any(feature = "dense", feature = "sparse")),
    allow(unused_variables)
)]
fn build_ivf_stage(input: Input<'_>, stage: &StageConfig) -> PyResult<Box<dyn AnyIndex>> {
    with_ivf!(stage, |DQ, DC| build_ivf::<DQ, DC>(input, stage))
}

#[cfg(feature = "ivf")]
// With `ivf` but neither `dense` nor `sparse`, the table has no arms left.
#[cfg_attr(
    not(any(feature = "dense", feature = "sparse")),
    allow(unused_variables)
)]
fn load_ivf_stage(reader: &mut dyn Read, stage: &StageConfig) -> PyResult<Box<dyn AnyIndex>> {
    with_ivf!(stage, |DQ, DC| load_ivf::<DQ, DC>(reader, stage))
}

#[cfg(not(feature = "ivf"))]
fn build_ivf_stage(_: Input<'_>, _: &StageConfig) -> PyResult<Box<dyn AnyIndex>> {
    Err(value_err(
        "this build of kannolo was compiled without the `ivf` feature",
    ))
}

#[cfg(not(feature = "ivf"))]
fn load_ivf_stage(_: &mut dyn Read, _: &StageConfig) -> PyResult<Box<dyn AnyIndex>> {
    Err(value_err(
        "this build of kannolo was compiled without the `ivf` feature",
    ))
}

fn build_first(input: Input<'_>, stage: &StageConfig) -> PyResult<Box<dyn AnyIndex>> {
    if input.representation() != stage.representation {
        return Err(value_err(format!(
            "the first stage is {:?} but its data is {:?}",
            stage.representation,
            input.representation()
        )));
    }
    if let IndexStructureConfig::Ivf { .. } = stage.index {
        build_ivf_stage(input, stage)
    } else {
        with_encoding!(&stage.encoding(), |E| build_graph_or_flat::<E>(
            input, stage
        ))
    }
}

fn load_first(reader: &mut dyn Read, stage: &StageConfig) -> PyResult<Box<dyn AnyIndex>> {
    if let IndexStructureConfig::Ivf { .. } = stage.index {
        load_ivf_stage(reader, stage)
    } else {
        with_encoding!(&stage.encoding(), |E| load_graph_or_flat::<E>(
            reader, stage
        ))
    }
}

/// Adds rescoring stage `level` (`r`, built from `input`) on top of `first`.
fn add_stage(
    first: Arc<dyn AnyIndex>,
    input: Input<'_>,
    r: &RerankConfig,
    level: usize,
) -> PyResult<Arc<dyn AnyIndex>> {
    if input.representation() != r.representation {
        return Err(value_err(format!(
            "rescoring stage {} is {:?} but its data is {:?}",
            level + 1,
            r.representation,
            input.representation()
        )));
    }
    if input.len() != first.n_elements() {
        return Err(value_err(format!(
            "rescoring stage {} has {} vectors but the first stage has {}: both must describe \
             the same collection in the same order",
            level + 1,
            input.len(),
            first.n_elements()
        )));
    }
    let enc = r.encoding();
    let rerank = with_encoding!(&enc, |E| build_rerank::<E>(input, &enc))?;
    Ok(Arc::new(TwoStage {
        first,
        rerank,
        level,
        residual: r.residual,
    }))
}

/// What the first stage leaves out of `original`, for a residual rescoring stage: the
/// components its L1 pruning dropped (sparse), or `x - x̂` for its RaBitQ reconstruction `x̂`
/// (dense). Validation allows residuals only in these two cases.
fn residual_of(original: Input<'_>, first: &StageConfig) -> PyResult<OwnedInput> {
    match transform_first_stage(original, first) {
        Some((_, pruned_away)) => Ok(pruned_away),
        None => dense_rabitq_residual(original, first),
    }
}

/// Builds the index a validated spec describes.
///
/// `data[0]` is the first stage's input; `data[i]` (`i >= 1`) the input of rescoring stage `i`,
/// or `None` to rescore the first stage's (untransformed) input.
pub(crate) fn build(spec: &IndexConfig, data: &[Option<Input<'_>>]) -> PyResult<Arc<dyn AnyIndex>> {
    let original = data
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| value_err("the first stage needs data"))?;
    let transformed = transform_first_stage(original, &spec.first);
    let first_input = transformed
        .as_ref()
        .map_or(original, |(kept, _)| kept.view());
    let first: Arc<dyn AnyIndex> = Arc::from(build_first(first_input, &spec.first)?);

    // What the first stage leaves out, for a residual rescoring stage (validation allows only
    // one, the first). The pruned-away components come with the transform; `x - x̂` is
    // computed here.
    let dense_residual = match (&transformed, spec.rerank.first()) {
        (None, Some(r)) if r.residual => Some(dense_rabitq_residual(original, &spec.first)?),
        _ => None,
    };
    let residual = transformed
        .as_ref()
        .map(|(_, pruned_away)| pruned_away)
        .or(dense_residual.as_ref());

    let stage_data: Vec<Option<Input<'_>>> = spec
        .rerank
        .iter()
        .enumerate()
        .map(|(level, r)| {
            if r.residual {
                residual.map(OwnedInput::view)
            } else {
                Some(data.get(level + 1).copied().flatten().unwrap_or(original))
            }
        })
        .collect();
    add_stages(first, spec, 0, &stage_data)
}

/// Adds rescoring stages `from_level..` of `spec` on top of `first`, each built from its entry
/// of `stage_data` (already the residual for a residual stage).
fn add_stages(
    mut index: Arc<dyn AnyIndex>,
    spec: &IndexConfig,
    from_level: usize,
    stage_data: &[Option<Input<'_>>],
) -> PyResult<Arc<dyn AnyIndex>> {
    for (level, r) in spec.rerank.iter().enumerate().skip(from_level) {
        let input = stage_data
            .get(level - from_level)
            .copied()
            .flatten()
            .ok_or_else(|| value_err(format!("rescoring stage {} needs data", level + 1)))?;
        index = add_stage(index, input, r, level)?;
    }
    Ok(index)
}

/// Adds the rescoring stages `from_level..` of `spec` to an existing index `first`, whose own
/// spec is `spec` without them. `data[j]` is the collection of stage `from_level + j` in its own
/// representation. For a residual stage, `data[j]` is the original collection, and the residual
/// is computed from it as in `build`.
///
/// `first` is shared through an `Arc`, so the same index can be extended in several ways
/// without being copied.
pub(crate) fn extend(
    first: Arc<dyn AnyIndex>,
    spec: &IndexConfig,
    from_level: usize,
    data: &[Input<'_>],
) -> PyResult<Arc<dyn AnyIndex>> {
    if data.len() != spec.rerank.len() - from_level {
        return Err(value_err(format!(
            "{} datasets for {} rescoring stages",
            data.len(),
            spec.rerank.len() - from_level
        )));
    }
    let mut residuals: Vec<Option<OwnedInput>> = Vec::with_capacity(data.len());
    for (j, r) in spec.rerank[from_level..].iter().enumerate() {
        residuals.push(if r.residual {
            if data[j].representation() != spec.first.representation {
                return Err(value_err(
                    "a residual stage takes the original collection of the first stage and \
                     computes the residual from it",
                ));
            }
            Some(residual_of(data[j], &spec.first)?)
        } else {
            None
        });
    }
    let stage_data: Vec<Option<Input<'_>>> = data
        .iter()
        .zip(&residuals)
        .map(|(d, resid)| Some(resid.as_ref().map_or(*d, OwnedInput::view)))
        .collect();
    add_stages(first, spec, from_level, &stage_data)
}

/// Reads the payloads written by `AnyIndex::write_payload`, in the same order.
pub(crate) fn load(reader: &mut dyn Read, spec: &IndexConfig) -> PyResult<Arc<dyn AnyIndex>> {
    let mut index: Arc<dyn AnyIndex> = Arc::from(load_first(reader, &spec.first)?);
    for (level, r) in spec.rerank.iter().enumerate() {
        let rerank = with_encoding!(&r.encoding(), |E| load_rerank::<E>(reader))?;
        index = Arc::new(TwoStage {
            first: index,
            rerank,
            level,
            residual: r.residual,
        });
    }
    Ok(index)
}
