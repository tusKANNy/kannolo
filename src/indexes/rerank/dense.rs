//! [`DenseRerankIndex`]: a compressed dense HNSW first stage reranked against a second stage that
//! is plain `f16` by default and RaBitQ-ext codes under `rerank_bits`.
//!
//! The graph is always built over `f16` and only its dataset is then replaced by codes, so the
//! compressed representation never takes part in construction. The rerank dataset is the same
//! collection in the original vector order. Because construction reads only the `f16` copy,
//! `rerank_bits` leaves the graph and the first stage unchanged and only changes how candidates
//! are rescored.
//!
//! The index is generic over its metric: `DenseRerankIndex<DotProduct>` or
//! `DenseRerankIndex<SquaredEuclideanDistance>`.

use half::f16;
use rayon::prelude::*;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::graph::GraphTrait;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};

use crate::rerank_index::RerankIndex;
use vectorium::SpaceUsage;
use vectorium::core::dataset::ConvertFrom;
use vectorium::dataset::ScoredVector;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::vector::DenseVectorView;
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use vectorium::{
    Dataset, DenseDataset, Index, IndexSerializer, IndexStats, RabitqConfig, RabitqExtConfig,
    RabitqExtQuantizer, RabitqQuantizer, RabitqQueryParams, RabitqSupportedDistance,
};

use super::{GraphLayout, RerankError, RerankSearchParams, Result, check_graph_layout, invalid};

const PQ_SUBSPACES: &str = "4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256";

/// The first-stage encoder and its width.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DenseEncoder {
    /// Product quantization with `subspaces` subspaces.
    Pq { subspaces: usize },
    /// 1-bit RaBitQ; `query_bits` (1..=8) is the query width, which can differ per search.
    Rabitq { query_bits: u32 },
    /// RaBitQ-ext with `total_bits` (2, 4 or 8) per component.
    RabitqExt { total_bits: u32 },
}

/// Build options for [`DenseRerankIndex::build`].
#[derive(Copy, Clone, Debug)]
pub struct DenseRerankConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub graph: GraphLayout,
    pub encoder: DenseEncoder,
    /// The rerank stage's encoding: `0` for plain `f16`, or `2`, `4` or `8` for RaBitQ-ext codes.
    pub rerank_bits: u32,
    /// Encode `x − x̂` in the rerank stage and score a candidate as first-stage plus rerank score.
    /// Needs `rerank_bits > 0`, a RaBitQ or RaBitQ-ext first stage, and a `DotProduct` index.
    pub residual_rerank: bool,
}

impl DenseRerankConfig {
    /// `encoder` over an `f16` rerank stage, with `m = 32`, `ef_construction = 200` and the
    /// standard graph layout.
    pub fn new(encoder: DenseEncoder) -> Self {
        Self {
            m: 32,
            ef_construction: 200,
            graph: GraphLayout::Standard,
            encoder,
            rerank_bits: 0,
            residual_rerank: false,
        }
    }
}

trait DenseRerankSearcher<D>: Send + Sync {
    fn search(&self, query: &[f32], k: usize, params: &RerankSearchParams) -> Vec<ScoredVector<D>>;
    fn space_usage_bytes(&self) -> usize;
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn save(&self, path: &str) -> Result<()>;
    /// Changes the query bit width. Returns `false` if the encoder does not quantize the query.
    fn set_query_bits(&mut self, bits: u32) -> bool;
}

/// Lets `set_query_bits` change the first-stage encoder's `QueryParams` generically.
///
/// The query width only changes how the query is encoded so it can be changed on a built index.
trait SettableQueryBits {
    fn set_bits(&mut self, bits: u32) -> bool;
}

/// The encoders whose queries are not quantized (`pq`, `rabitq-ext`) carry `()` here.
impl SettableQueryBits for () {
    fn set_bits(&mut self, _bits: u32) -> bool {
        false
    }
}

impl SettableQueryBits for RabitqQueryParams {
    fn set_bits(&mut self, bits: u32) -> bool {
        // Callers validate the range; `RabitqQueryParams::new` panics outside 1..=8.
        *self = RabitqQueryParams::new(bits);
        true
    }
}

/// One first-stage encoder `E` over one graph backend `G`, plus a rerank dataset under encoder `R`.
///
/// `R` is `PlainDenseQuantizer<f16, _>` for the exact rerank stage and `RabitqExtQuantizer<_>` for
/// the compressed one. Both take an unquantized `f32` query and have `QueryParams = ()`.
struct Stage<E, R, G>
where
    E: DenseVectorEncoder,
    <E as VectorEncoder>::Distance: ScalarDenseSupportedDistance,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    R: DenseVectorEncoder
        + VectorEncoder<Distance = <E as VectorEncoder>::Distance, QueryParams = ()>,
    DenseDataset<R>: Dataset<Encoder = R> + Sync + SpaceUsage,
    G: GraphTrait,
{
    index: RerankIndex<HNSW<DenseDataset<E>, G>, DenseDataset<R>>,
    query_params: <E as VectorEncoder>::QueryParams,
    /// Score a candidate as first-stage score plus rerank score, instead of the rerank score
    /// alone. Set when the rerank dataset holds `x − x̂`; see `residual_f16_dataset`.
    residuals: bool,
    dim: usize,
    space_usage_bytes: usize,
}

impl<E, R, G> DenseRerankSearcher<<E as VectorEncoder>::Distance> for Stage<E, R, G>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send + SettableQueryBits,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    R: DenseVectorEncoder
        + VectorEncoder<Distance = <E as VectorEncoder>::Distance, QueryParams = ()>
        + Sync
        + Send
        + 'static,
    DenseDataset<R>: Dataset<Encoder = R> + Sync + SpaceUsage + serde::Serialize,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize,
    G: serde::Serialize,
    <E as DenseVectorEncoder>::OutputValueType: serde::Serialize,
{
    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &RerankSearchParams,
    ) -> Vec<ScoredVector<<E as VectorEncoder>::Distance>> {
        let first_stage_params = HNSWSearchConfiguration {
            ef_search: params.ef_search,
            early_termination: params.early_termination,
            query_params: self.query_params.clone(),
        };
        let view = DenseVectorView::new(query);
        self.index.search(
            view,
            view,
            params.k_candidates,
            k,
            &first_stage_params,
            &(),
            params.alpha,
            params.beta,
            self.residuals,
        )
    }

    fn space_usage_bytes(&self) -> usize {
        self.space_usage_bytes
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.index.rerank_dataset().len()
    }

    fn save(&self, path: &str) -> Result<()> {
        self.index
            .save_index(path)
            .map_err(|e| RerankError(format!("error saving index: {e:?}")))
    }

    fn set_query_bits(&mut self, bits: u32) -> bool {
        self.query_params.set_bits(bits)
    }
}

/// A dense two-stage index; see the module documentation. `D` is the metric, `DotProduct` or
/// `SquaredEuclideanDistance`.
pub struct DenseRerankIndex<D> {
    inner: Box<dyn DenseRerankSearcher<D>>,
}

/// Converts the input to a plain `f16` dataset, in the original vector order.
fn plain_f16_dataset<D>(values: &[f32], dim: usize) -> DenseDataset<PlainDenseQuantizer<f16, D>>
where
    D: ScalarDenseSupportedDistance,
{
    let data: Vec<f16> = values.iter().map(|&x| f16::from_f32(x)).collect();
    let n_vecs = data.len() / dim;
    DenseDataset::from_raw(
        data.into_boxed_slice(),
        n_vecs,
        PlainDenseQuantizer::<f16, D>::new(dim),
    )
}

/// Builds the residual dataset `x − x̂` at `f16`, in the original vector order, where `x̂` is the
/// reconstruction of `x` by `first`.
///
/// With residual reranking, a candidate's score is the first-stage score plus the rerank score.
/// Under dot product, `⟨q, x⟩ = ⟨q, x̂⟩ + ⟨q, x − x̂⟩`, so the sum is exact when the first stage
/// scores `⟨q, x̂⟩`, as RaBitQ-ext does. The residual has a much smaller norm than `x`, so the same
/// number of bits encodes it more accurately.
///
/// `first` must use the same configuration and seed as the graph's first stage, so that the
/// reconstructions match up to float rounding in the trained means. Dot product only; the caller
/// checks this.
fn residual_f16_dataset<E, D>(
    plain: &DenseDataset<PlainDenseQuantizer<f16, D>>,
    first: &DenseDataset<E>,
    dim: usize,
) -> DenseDataset<PlainDenseQuantizer<f16, D>>
where
    D: ScalarDenseSupportedDistance + RabitqSupportedDistance,
    E: DenseVectorEncoder<OutputValueType = u64> + ReconstructOriginal + Sync,
    DenseDataset<E>: Dataset<Encoder = E> + Sync,
{
    let n_vecs = plain.len();
    let raw = plain.values();
    // f16 is enough: the residual is then encoded with at most 8 bits.
    let mut data: Vec<f16> = vec![f16::ZERO; n_vecs * dim];
    data.par_chunks_mut(dim).enumerate().for_each(|(id, out)| {
        let x_hat = first
            .encoder()
            .reconstruct_original(first.get(id as u64).values());
        let x = &raw[id * dim..(id + 1) * dim];
        for ((o, &xi), &hi) in out.iter_mut().zip(x).zip(x_hat.iter()) {
            *o = f16::from_f32(xi.to_f32() - hi);
        }
    });
    DenseDataset::from_raw(
        data.into_boxed_slice(),
        n_vecs,
        PlainDenseQuantizer::<f16, D>::new(dim),
    )
}

/// Lets `residual_f16_dataset` take either RaBitQ first-stage encoder.
///
/// `RabitqExtQuantizer` scores an unquantized query, so its score is exactly `⟨q, x̂⟩` and the
/// residual sum is exact. `RabitqQuantizer` also quantizes the query, so the sum keeps the
/// query-side error and is only an approximation.
trait ReconstructOriginal {
    /// `x̂` in the original data space.
    fn reconstruct_original(&self, code: &[u64]) -> Vec<f32>;
}

impl<D: RabitqSupportedDistance> ReconstructOriginal for RabitqExtQuantizer<D> {
    fn reconstruct_original(&self, code: &[u64]) -> Vec<f32> {
        self.reconstruct(code)
    }
}

impl<D: RabitqSupportedDistance> ReconstructOriginal for RabitqQuantizer<D> {
    fn reconstruct_original(&self, code: &[u64]) -> Vec<f32> {
        self.reconstruct(code)
    }
}

/// Builds the graph over `$graph_src` (plain `f16`), applies the requested graph layout, replaces
/// its dataset with `$target` codes, and pairs it with the `$rerank` dataset.
macro_rules! build_stage {
    ($dist:ty, $target:ty, $cfg:expr, $qp:expr, $graph_src:expr, $rerank:expr, $resid:expr, $build_cfg:expr, $gt:expr, $dim:expr) => {{
        let plain: HNSW<DenseDataset<PlainDenseQuantizer<f16, $dist>>, Graph> =
            HNSW::build_index($graph_src, $build_cfg);
        match $gt {
            GraphLayout::Standard => finish(
                plain.convert_dataset_into_ref::<$target, _>($cfg),
                $rerank,
                $qp,
                $resid,
                $dim,
            ),
            GraphLayout::Permuted => finish(
                plain
                    .permute_and_encode::<PlainNeighbors>()
                    .convert_dataset_into_ref::<$target, _>($cfg),
                $rerank,
                $qp,
                $resid,
                $dim,
            ),
            GraphLayout::Compressed => finish(
                plain
                    .permute_and_encode::<StreamVByteNeighbors>()
                    .convert_dataset_into_ref::<$target, _>($cfg),
                $rerank,
                $qp,
                $resid,
                $dim,
            ),
        }
    }};
}

fn finish<E, R, G>(
    index: HNSW<DenseDataset<E>, G>,
    rerank: DenseDataset<R>,
    query_params: <E as VectorEncoder>::QueryParams,
    residuals: bool,
    dim: usize,
) -> Box<dyn DenseRerankSearcher<<E as VectorEncoder>::Distance>>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send + SettableQueryBits,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    R: DenseVectorEncoder
        + VectorEncoder<Distance = <E as VectorEncoder>::Distance, QueryParams = ()>
        + Sync
        + Send
        + 'static,
    DenseDataset<R>: Dataset<Encoder = R> + Sync + SpaceUsage + serde::Serialize,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize,
    G: serde::Serialize,
    <E as DenseVectorEncoder>::OutputValueType: serde::Serialize,
{
    // Both stages. With an `f16` rerank stage this is a second full copy of the collection.
    let space_usage_bytes = index.space_usage_bytes() + rerank.space_usage_bytes();
    Box::new(Stage {
        index: RerankIndex::new(index, rerank),
        query_params,
        residuals,
        dim,
        space_usage_bytes,
    })
}

/// Wraps an index loaded from disk, reading `dim` and the space usage from the index itself.
fn finish_loaded<E, R, G>(
    index: RerankIndex<HNSW<DenseDataset<E>, G>, DenseDataset<R>>,
    query_params: <E as VectorEncoder>::QueryParams,
    residuals: bool,
) -> Box<dyn DenseRerankSearcher<<E as VectorEncoder>::Distance>>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send + SettableQueryBits,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    R: DenseVectorEncoder
        + VectorEncoder<Distance = <E as VectorEncoder>::Distance, QueryParams = ()>
        + Sync
        + Send
        + 'static,
    DenseDataset<R>: Dataset<Encoder = R> + Sync + SpaceUsage + serde::Serialize,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize,
    G: serde::Serialize,
    <E as DenseVectorEncoder>::OutputValueType: serde::Serialize,
{
    // `input_dim`, not `output_dim`: for RaBitQ-ext codes the latter is the code length.
    let dim = index.rerank_dataset().input_dim();
    let space_usage_bytes =
        index.first_stage_index().space_usage_bytes() + index.rerank_dataset().space_usage_bytes();
    Box::new(Stage {
        index,
        query_params,
        residuals,
        dim,
        space_usage_bytes,
    })
}

/// Loads one concrete (encoder, rerank encoder, graph) instantiation from `path`.
fn load_stage<E, R, G>(
    path: &str,
    query_params: <E as VectorEncoder>::QueryParams,
    residuals: bool,
) -> Result<Box<dyn DenseRerankSearcher<<E as VectorEncoder>::Distance>>>
where
    E: DenseVectorEncoder + Sync + Send + 'static,
    <E as VectorEncoder>::QueryParams: Clone + Sync + Send + SettableQueryBits,
    <E as VectorEncoder>::Distance:
        ScalarDenseSupportedDistance + Distance + From<f32> + Sync + Send,
    DenseDataset<E>: Dataset<Encoder = E> + Sync + SpaceUsage,
    R: DenseVectorEncoder
        + VectorEncoder<Distance = <E as VectorEncoder>::Distance, QueryParams = ()>
        + Sync
        + Send
        + 'static,
    DenseDataset<R>:
        Dataset<Encoder = R> + Sync + SpaceUsage + serde::Serialize + serde::de::DeserializeOwned,
    G: GraphTrait + Sync + Send + 'static,
    E: serde::Serialize + serde::de::DeserializeOwned,
    G: serde::Serialize + serde::de::DeserializeOwned,
    <E as DenseVectorEncoder>::OutputValueType:
        serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    let index: RerankIndex<HNSW<DenseDataset<E>, G>, DenseDataset<R>> =
        RerankIndex::load_index(path).map_err(load_err)?;
    Ok(finish_loaded::<E, R, G>(index, query_params, residuals))
}

pub(crate) fn load_err<E: std::fmt::Debug>(e: E) -> RerankError {
    RerankError(format!(
        "error loading index: {e:?}. Index files are not self-describing: check that the \
         arguments given here match the ones used when the index was built, and that the file \
         was written by this version of kannolo."
    ))
}

/// Loads a PQ first stage, dispatching on the number of subspaces.
macro_rules! load_pq_by_m {
    ($dist:ty, $rr:ty, $graph:ty, $path:expr, $m:expr, $resid:expr, [$($subspaces:literal),+ $(,)?]) => {
        match $m {
            $($subspaces => load_stage::<ProductQuantizer<$subspaces, $dist>, $rr, $graph>($path, (), $resid)?,)+
            other => {
                return invalid(format!("unsupported PQ subspaces {other}; supported: {PQ_SUBSPACES}"));
            }
        }
    };
}

/// Loads an index for one metric, rerank encoder and graph layout, dispatching on the encoder.
///
/// The trained encoders are stored with the index. Only the query parameters and the residual
/// flag must be given again, because they are not stored.
macro_rules! load_for_metric_graph {
    ($dist:ty, $rr:ty, $graph:ty, $path:expr, $encoder:expr, $resid:expr) => {{
        match $encoder {
            DenseEncoder::Pq { subspaces } => load_pq_by_m!(
                $dist,
                $rr,
                $graph,
                $path,
                subspaces,
                $resid,
                [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
            ),
            DenseEncoder::Rabitq { query_bits } => {
                load_stage::<RabitqQuantizer<$dist>, $rr, $graph>(
                    $path,
                    RabitqQueryParams::new(query_bits),
                    $resid,
                )?
            }
            DenseEncoder::RabitqExt { .. } => {
                load_stage::<RabitqExtQuantizer<$dist>, $rr, $graph>($path, (), $resid)?
            }
        }
    }};
}

/// Picks the rerank encoder type from `rerank_bits`, then dispatches on the encoder.
///
/// The RaBitQ-ext width is stored with the encoder, so `2`, `4` and `8` load the same type.
macro_rules! load_for_rerank {
    ($dist:ty, $graph:ty, $path:expr, $encoder:expr, $rerank_bits:expr, $resid:expr) => {{
        if $rerank_bits == 0 {
            load_for_metric_graph!($dist, PlainDenseQuantizer<f16, $dist>, $graph, $path, $encoder, $resid)
        } else {
            load_for_metric_graph!($dist, RabitqExtQuantizer<$dist>, $graph, $path, $encoder, $resid)
        }
    }};
}

/// Builds the index for one metric and rerank dataset, dispatching on the first-stage encoder.
macro_rules! encoder_table {
    ($dist:ty, $graph_src:expr, $rerank:expr, $resid:expr, $dim:expr, $build_cfg:expr, $gt:expr, $encoder:expr) => {{
        macro_rules! pq {
            ($m:literal) => {
                build_stage!(
                    $dist,
                    DenseDataset<ProductQuantizer<$m, $dist>>,
                    (),
                    (),
                    $graph_src,
                    $rerank,
                    $resid,
                    $build_cfg,
                    $gt,
                    $dim
                )
            };
        }
        match $encoder {
            DenseEncoder::Pq { subspaces } => match subspaces {
                4 => pq!(4),
                8 => pq!(8),
                16 => pq!(16),
                24 => pq!(24),
                32 => pq!(32),
                48 => pq!(48),
                64 => pq!(64),
                96 => pq!(96),
                128 => pq!(128),
                192 => pq!(192),
                256 => pq!(256),
                m => {
                    return invalid(format!(
                        "unsupported PQ subspaces {m}; supported: {PQ_SUBSPACES}"
                    ));
                }
            },
            DenseEncoder::Rabitq { query_bits } => build_stage!(
                $dist,
                DenseDataset<RabitqQuantizer<$dist>>,
                RabitqConfig::default(),
                RabitqQueryParams::new(query_bits),
                $graph_src,
                $rerank,
                $resid,
                $build_cfg,
                $gt,
                $dim
            ),
            DenseEncoder::RabitqExt { total_bits } => build_stage!(
                $dist,
                DenseDataset<RabitqExtQuantizer<$dist>>,
                RabitqExtConfig {
                    total_bits,
                    ..Default::default()
                },
                (),
                $graph_src,
                $rerank,
                $resid,
                $build_cfg,
                $gt,
                $dim
            ),
        }
    }};
}

/// Builds the index for one metric, dispatching on the rerank encoding.
macro_rules! build_for_metric {
    ($dist:ty, $values:expr, $dim:expr, $build_cfg:expr, $cfg:expr) => {{
        let plain = plain_f16_dataset::<$dist>($values, $dim);
        let gt = $cfg.graph;

        match ($cfg.rerank_bits, $cfg.residual_rerank) {
            (0, _) => {
                let graph_src = plain.clone();
                encoder_table!(
                    $dist,
                    graph_src,
                    plain,
                    false,
                    $dim,
                    $build_cfg,
                    gt,
                    $cfg.encoder
                )
            }
            (bits, false) => {
                let rerank: DenseDataset<RabitqExtQuantizer<$dist>> = ConvertFrom::convert_from(
                    &plain,
                    RabitqExtConfig {
                        total_bits: bits,
                        ..Default::default()
                    },
                );
                encoder_table!(
                    $dist,
                    plain,
                    rerank,
                    false,
                    $dim,
                    $build_cfg,
                    gt,
                    $cfg.encoder
                )
            }
            (bits, true) => {
                // Encode the first stage again in the original order to get `x̂`, since the
                // graph's copy may be permuted. Same configuration and seed as the graph's.
                let residual = match $cfg.encoder {
                    DenseEncoder::RabitqExt { total_bits } => {
                        let first: DenseDataset<RabitqExtQuantizer<$dist>> =
                            ConvertFrom::convert_from(
                                &plain,
                                RabitqExtConfig {
                                    total_bits,
                                    ..Default::default()
                                },
                            );
                        residual_f16_dataset(&plain, &first, $dim)
                    }
                    // Approximate rather than exact; see `ReconstructOriginal`.
                    _ => {
                        let first: DenseDataset<RabitqQuantizer<$dist>> =
                            ConvertFrom::convert_from(&plain, RabitqConfig::default());
                        residual_f16_dataset(&plain, &first, $dim)
                    }
                };
                let rerank: DenseDataset<RabitqExtQuantizer<$dist>> = ConvertFrom::convert_from(
                    &residual,
                    RabitqExtConfig {
                        total_bits: bits,
                        // A different rotation from the first stage's: the residual is aligned
                        // with the first stage's quantization grid, and the same rotation would
                        // break RaBitQ's assumption that rotated components are concentrated.
                        seed: RabitqExtConfig::default().seed ^ 0x5265_7369_6475_616C,
                        ..Default::default()
                    },
                );
                drop(residual);
                encoder_table!(
                    $dist,
                    plain,
                    rerank,
                    true,
                    $dim,
                    $build_cfg,
                    gt,
                    $cfg.encoder
                )
            }
        }
    }};
}

/// Rejects out-of-range encoder widths before they reach vectorium.
///
/// The encoders assert these ranges with `panic!`, and the release profile sets
/// `panic = "abort"`. `check_total_bits` is false on load, where the width is stored.
fn validate_encoder(encoder: &DenseEncoder, check_total_bits: bool) -> Result<()> {
    match *encoder {
        DenseEncoder::Rabitq { query_bits } if !(1..=8).contains(&query_bits) => invalid(format!(
            "RaBitQ query bits must be in 1..=8, got {query_bits}"
        )),
        DenseEncoder::RabitqExt { total_bits }
            if check_total_bits && !matches!(total_bits, 2 | 4 | 8) =>
        {
            invalid(format!(
                "RaBitQ-ext total bits must be 2, 4 or 8, got {total_bits}; for a 1-bit code use \
                 the RaBitQ encoder"
            ))
        }
        _ => Ok(()),
    }
}

/// Rejects a dimension the encoders cannot handle: RaBitQ and RaBitQ-ext (first stage or rerank
/// stage) need a multiple of 64, PQ a multiple of the number of subspaces.
fn validate_dim(encoder: &DenseEncoder, rerank_bits: u32, dim: usize) -> Result<()> {
    let rabitq = !matches!(encoder, DenseEncoder::Pq { .. }) || rerank_bits > 0;
    if rabitq && !dim.is_multiple_of(64) {
        return invalid(format!(
            "RaBitQ and RaBitQ-ext need a dimension that is a multiple of 64, got {dim}"
        ));
    }
    if let DenseEncoder::Pq { subspaces } = *encoder
        && subspaces > 0
        && !dim.is_multiple_of(subspaces)
    {
        return invalid(format!(
            "PQ needs a dimension divisible by the number of subspaces, got {dim} and {subspaces}"
        ));
    }
    Ok(())
}

/// Rejects invalid rerank-stage settings.
///
/// Residual reranking sums two scores, which is only meaningful under dot product and when the
/// first stage's reconstruction is available.
fn validate_rerank(
    encoder: &DenseEncoder,
    dot_product: bool,
    rerank_bits: u32,
    residual_rerank: bool,
) -> Result<()> {
    if !matches!(rerank_bits, 0 | 2 | 4 | 8) {
        return invalid(format!(
            "unsupported rerank bits {rerank_bits}; use 0 for the exact f16 rerank stage, or 2, 4 \
             or 8 for a RaBitQ-ext one"
        ));
    }
    if residual_rerank {
        if rerank_bits == 0 {
            return invalid(
                "residual reranking needs a compressed rerank stage (rerank bits 2, 4 or 8); an \
                 f16 rerank stage is exact and has no residual",
            );
        }
        if matches!(encoder, DenseEncoder::Pq { .. }) {
            return invalid(
                "residual reranking needs a RaBitQ or RaBitQ-ext first stage, whose \
                 reconstruction is available",
            );
        }
        if !dot_product {
            return invalid(
                "residual reranking is dot-product only: the first-stage and rerank scores are \
                 summed, and the sum of two squared-Euclidean distances is not a distance",
            );
        }
    }
    Ok(())
}

/// `build` and `load` for one concrete metric; `$dot` says whether it is dot product.
macro_rules! impl_build_load {
    ($dist:ty, $dot:expr) => {
        impl DenseRerankIndex<$dist> {
            /// Builds the index from a flat row-major array of `values.len() / dim` vectors.
            pub fn build(values: &[f32], dim: usize, cfg: &DenseRerankConfig) -> Result<Self> {
                if dim == 0 {
                    return invalid("dim must be greater than zero");
                }
                if values.is_empty() || !values.len().is_multiple_of(dim) {
                    return invalid(format!(
                        "data length {} is not a positive multiple of dim {dim}",
                        values.len()
                    ));
                }
                validate_encoder(&cfg.encoder, true)?;
                validate_dim(&cfg.encoder, cfg.rerank_bits, dim)?;
                validate_rerank(&cfg.encoder, $dot, cfg.rerank_bits, cfg.residual_rerank)?;
                check_graph_layout(cfg.graph, cfg.m)?;
                let build_cfg = HNSWBuildConfiguration::default()
                    .with_num_neighbors(cfg.m)
                    .with_ef_construction(cfg.ef_construction);
                let inner = build_for_metric!($dist, values, dim, &build_cfg, cfg);
                Ok(Self { inner })
            }

            /// Loads an index written by [`Self::save`].
            ///
            /// Index files carry no header, so the metric, `graph`, the encoder (and PQ
            /// subspaces), `residual_rerank` and whether `rerank_bits` is zero must match the
            /// build. RaBitQ-ext's `total_bits` and the exact `rerank_bits` are stored with the
            /// index; RaBitQ's `query_bits` may differ from the build.
            pub fn load(
                path: &str,
                graph: GraphLayout,
                encoder: DenseEncoder,
                rerank_bits: u32,
                residual_rerank: bool,
            ) -> Result<Self> {
                validate_encoder(&encoder, false)?;
                validate_rerank(&encoder, $dot, rerank_bits, residual_rerank)?;
                let inner = match graph {
                    GraphLayout::Standard | GraphLayout::Permuted => {
                        load_for_rerank!($dist, Graph, path, encoder, rerank_bits, residual_rerank)
                    }
                    GraphLayout::Compressed => load_for_rerank!(
                        $dist,
                        GenericGraph<StreamVByteNeighbors>,
                        path,
                        encoder,
                        rerank_bits,
                        residual_rerank
                    ),
                };
                Ok(Self { inner })
            }
        }
    };
}

impl_build_load!(DotProduct, true);
impl_build_load!(SquaredEuclideanDistance, false);

impl<D: Distance> DenseRerankIndex<D> {
    /// Writes the first-stage graph and the rerank dataset to one file.
    pub fn save(&self, path: &str) -> Result<()> {
        self.inner.save(path)
    }

    /// Changes the RaBitQ query bit width. The stored codes are unaffected, so one index serves
    /// every width.
    ///
    /// Fails if `bits` is outside `1..=8`, or if the encoder does not quantize the query (PQ and
    /// RaBitQ-ext).
    pub fn set_query_bits(&mut self, bits: u32) -> Result<()> {
        if !(1..=8).contains(&bits) {
            return invalid(format!("query bits must be in 1..=8, got {bits}"));
        }
        if !self.inner.set_query_bits(bits) {
            return invalid(
                "this encoder does not quantize the query; only the 1-bit RaBitQ encoder does",
            );
        }
        Ok(())
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Number of vectors.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held by the first-stage index plus the rerank dataset.
    pub fn space_usage_bytes(&self) -> usize {
        self.inner.space_usage_bytes()
    }
}

/// Two-stage search: the query, of length [`IndexStats::dim`], is used for both stages.
impl<D: Distance> Index for DenseRerankIndex<D> {
    type Query<'q> = DenseVectorView<'q, f32>;
    type Distance = D;
    type SearchParams = RerankSearchParams;

    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<D>> {
        assert_eq!(
            query.values().len(),
            self.inner.dim(),
            "query dimension does not match the index"
        );
        self.inner.search(query.values(), k, search_params)
    }
}

impl<D: Distance> IndexStats for DenseRerankIndex<D> {
    fn n_elements(&self) -> usize {
        self.inner.len()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }
}
