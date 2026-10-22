//! [`SparseRerankIndex`]: a sparse HNSW first stage reranked against a second sparse dataset.
//!
//! Both stages are built from one collection. First and second stage are regulated by three
//! options in [`SparseRerankConfig`]:
//!
//! - `prune_percentage < 100` L1-prunes the first stage to the components carrying the top
//!   `prune_percentage`% of each vector's L1 mass (`vectorium::l1_prune_split`).
//! - `first_stage` stores the first stage as blocked sparse (the usual choice), `f16` or
//!   fixed-point `u8`. The rerank stage holds `f16` values; with a blocked first stage it is
//!   stored blocked too, so a short query is scored without densifying it in either stage.
//! - `residual_rerank` stores only the pruned-away components in the rerank stage and scores a
//!   candidate as first-stage plus rerank score. With an `f16` or blocked first stage this is the
//!   exact score at `f16` precision, `<q, x> = <q, x_pruned> + <q, x_residual>`, and each
//!   non-zero of the collection is stored once. Without it the rerank stage holds the full vectors
//!   and its score replaces the first-stage one.
//!
//! Only dot product is supported.

use half::f16;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};

use crate::rerank_index::RerankIndex;
use vectorium::distances::DotProduct;
use vectorium::vector::SparseVectorView;
use vectorium::{
    BlockedSparseDataset, Dataset, DatasetGrowable, FixedU8Q, Float, FromF32, Index,
    IndexSerializer, IndexStats, PlainSparseDataset, PlainSparseDatasetGrowable,
    PlainSparseQuantizer, ScalarSparseDataset, ScalarSparseDatasetGrowable, ScalarSparseQuantizer,
    SpaceUsage, ValueType, l1_prune_split,
};

use super::dense::load_err;
use super::{GraphLayout, RerankError, RerankSearchParams, Result, check_graph_layout, invalid};
use vectorium::dataset::ScoredVector;

/// The rerank dataset, always `f16`.
type RerankData = PlainSparseDataset<u16, f16, DotProduct>;

/// The first-stage representations.
type FirstStageF16 = PlainSparseDataset<u16, f16, DotProduct>;
type FirstStageU8 = ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>;
/// Blocked sparse: the `f16` values and `u16` components of [`FirstStageF16`] packed into
/// 64-byte blocks; used for both stages.
type FirstStageBlocked = BlockedSparseDataset;
type RerankBlocked = BlockedSparseDataset;

enum Inner {
    F16(RerankIndex<HNSW<FirstStageF16, Graph>, RerankData>),
    F16StreamVByte(
        RerankIndex<HNSW<FirstStageF16, GenericGraph<StreamVByteNeighbors>>, RerankData>,
    ),
    FixedU8(RerankIndex<HNSW<FirstStageU8, Graph>, RerankData>),
    FixedU8StreamVByte(
        RerankIndex<HNSW<FirstStageU8, GenericGraph<StreamVByteNeighbors>>, RerankData>,
    ),
    Blocked(RerankIndex<HNSW<FirstStageBlocked, Graph>, RerankBlocked>),
}

const BLOCKED_COMPRESSED: &str = "a blocked first stage supports the standard and permuted graph \
     layouts; use an f16 or fixed u8 first stage for the compressed layout";

/// The value type the first stage stores.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparseFirstStage {
    F16,
    FixedU8,
    /// Blocked sparse for both stages. The graph is built over the `f16` first stage and the
    /// two datasets are then re-encoded, so it is the graph of an `F16` build with the same
    /// parameters, and scores are the same `f16` products.
    Blocked,
}

/// Build options for [`SparseRerankIndex::build`].
#[derive(Copy, Clone, Debug)]
pub struct SparseRerankConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub graph: GraphLayout,
    pub first_stage: SparseFirstStage,
    /// In `[0, 100]`; the first stage keeps, per vector, the components carrying the top
    /// `prune_percentage`% of its L1 mass. 100 means no pruning.
    pub prune_percentage: f32,
    /// Store only the pruned-away components in the rerank stage and sum the two scores. Needs
    /// `prune_percentage < 100`.
    pub residual_rerank: bool,
}

impl SparseRerankConfig {
    /// A blocked first stage L1-pruned at `prune_percentage`, with residual reranking, `m = 32`,
    /// `ef_construction = 200`, standard graph.
    pub fn l1_pruned(prune_percentage: f32) -> Self {
        Self {
            m: 32,
            ef_construction: 200,
            graph: GraphLayout::Standard,
            first_stage: SparseFirstStage::Blocked,
            prune_percentage,
            residual_rerank: true,
        }
    }

    /// Rejects options that give an invalid or pointless index.
    fn validate(&self) -> Result<()> {
        let p = self.prune_percentage;
        if !(0.0..=100.0).contains(&p) {
            return invalid(format!("prune percentage must be in [0, 100], got {p}"));
        }
        if p == 100.0 {
            if self.residual_rerank {
                return invalid(
                    "residual reranking needs a prune percentage below 100: without pruning the \
                     residual is empty",
                );
            }
            if matches!(
                self.first_stage,
                SparseFirstStage::F16 | SparseFirstStage::Blocked
            ) {
                return invalid(
                    "with no pruning and an f16 or blocked first stage the rerank stage would be \
                     an identical copy; prune the first stage, quantize it to fixed u8, or use a \
                     single-stage HNSW index",
                );
            }
        }
        if self.first_stage == SparseFirstStage::Blocked && self.graph == GraphLayout::Compressed {
            return invalid(BLOCKED_COMPRESSED);
        }
        check_graph_layout(self.graph, self.m)
    }
}

/// A sparse two-stage index.
pub struct SparseRerankIndex {
    inner: Inner,
    residual_rerank: bool,
}

/// Re-encodes a sparse dataset's values into `V`, keeping components, ids and dimensionality.
fn convert_sparse<V>(
    source: &PlainSparseDataset<u16, f32, DotProduct>,
) -> PlainSparseDataset<u16, V, DotProduct>
where
    V: ValueType + Float + FromF32 + SpaceUsage,
{
    let dim = source.input_dim();
    let mut out: PlainSparseDatasetGrowable<u16, V, DotProduct> =
        DatasetGrowable::with_capacity(PlainSparseQuantizer::new(dim, dim), source.len());
    let mut values: Vec<V> = Vec::new();
    for vector in source.iter() {
        values.clear();
        values.extend(vector.values().iter().map(|&v| V::from_f32_saturating(v)));
        out.push(SparseVectorView::new(vector.components(), &values));
    }
    out.into()
}

/// Re-encodes a sparse dataset into the scalar-quantized `FixedU8Q` representation.
fn convert_sparse_scalar(source: &PlainSparseDataset<u16, f32, DotProduct>) -> FirstStageU8 {
    let dim = source.input_dim();
    let mut out: ScalarSparseDatasetGrowable<u16, f32, FixedU8Q, DotProduct> =
        DatasetGrowable::with_capacity(ScalarSparseQuantizer::new(dim, dim), source.len());
    for vector in source.iter() {
        out.push(vector);
    }
    out.into()
}

/// Applies the graph layout to a built graph and pairs it with `rerank`.
macro_rules! assemble {
    ($plain:expr, $rerank:expr, $gt:expr, $flat:ident, $svb:ident) => {
        match $gt {
            GraphLayout::Standard => Inner::$flat(RerankIndex::new($plain, $rerank)),
            GraphLayout::Permuted => Inner::$flat(RerankIndex::new(
                $plain.permute_and_encode::<PlainNeighbors>(),
                $rerank,
            )),
            GraphLayout::Compressed => Inner::$svb(RerankIndex::new(
                $plain.permute_and_encode::<StreamVByteNeighbors>(),
                $rerank,
            )),
        }
    };
}

/// Runs `$body` with `$index` bound to whichever variant `$inner` holds.
macro_rules! with_index {
    ($inner:expr, $index:ident => $body:expr) => {
        match $inner {
            Inner::F16($index) => $body,
            Inner::F16StreamVByte($index) => $body,
            Inner::FixedU8($index) => $body,
            Inner::FixedU8StreamVByte($index) => $body,
            Inner::Blocked($index) => $body,
        }
    };
}

impl SparseRerankIndex {
    /// Builds both stages from `collection`, as described in the module documentation.
    pub fn build(
        collection: PlainSparseDataset<u16, f32, DotProduct>,
        cfg: &SparseRerankConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        let build_cfg = HNSWBuildConfiguration::default()
            .with_num_neighbors(cfg.m)
            .with_ef_construction(cfg.ef_construction);

        let (first_stage, rerank) = if cfg.prune_percentage < 100.0 {
            let (pruned, residual) = l1_prune_split(&collection, cfg.prune_percentage);
            let rerank = if cfg.residual_rerank {
                convert_sparse::<f16>(&residual)
            } else {
                convert_sparse::<f16>(&collection)
            };
            (pruned, rerank)
        } else {
            let rerank = convert_sparse::<f16>(&collection);
            (collection, rerank)
        };

        let inner = match cfg.first_stage {
            SparseFirstStage::F16 => {
                let dataset: FirstStageF16 = convert_sparse::<f16>(&first_stage);
                drop(first_stage);
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &build_cfg);
                assemble!(plain, rerank, cfg.graph, F16, F16StreamVByte)
            }
            SparseFirstStage::FixedU8 => {
                let dataset: FirstStageU8 = convert_sparse_scalar(&first_stage);
                drop(first_stage);
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &build_cfg);
                assemble!(plain, rerank, cfg.graph, FixedU8, FixedU8StreamVByte)
            }
            SparseFirstStage::Blocked => {
                let dataset: FirstStageF16 = convert_sparse::<f16>(&first_stage);
                drop(first_stage);
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &build_cfg);
                let plain = match cfg.graph {
                    GraphLayout::Permuted => plain.permute_and_encode::<PlainNeighbors>(),
                    _ => plain, // Compressed is rejected by `validate`
                };
                let blocked: HNSW<FirstStageBlocked, Graph> = plain.convert_dataset_into(());
                let rerank: RerankBlocked = rerank.into();
                Inner::Blocked(RerankIndex::new(blocked, rerank))
            }
        };
        Ok(Self {
            inner,
            residual_rerank: cfg.residual_rerank,
        })
    }

    /// Loads an index written by [`Self::save`].
    ///
    /// Index files carry no header, so `first_stage`, `graph` and `residual_rerank` must match the
    /// build. `Standard` and `Permuted` load as the same type.
    pub fn load(
        path: &str,
        first_stage: SparseFirstStage,
        graph: GraphLayout,
        residual_rerank: bool,
    ) -> Result<Self> {
        let inner = match (first_stage, graph) {
            (SparseFirstStage::F16, GraphLayout::Standard | GraphLayout::Permuted) => {
                Inner::F16(RerankIndex::load_index(path).map_err(load_err)?)
            }
            (SparseFirstStage::F16, GraphLayout::Compressed) => {
                Inner::F16StreamVByte(RerankIndex::load_index(path).map_err(load_err)?)
            }
            (SparseFirstStage::FixedU8, GraphLayout::Standard | GraphLayout::Permuted) => {
                Inner::FixedU8(RerankIndex::load_index(path).map_err(load_err)?)
            }
            (SparseFirstStage::FixedU8, GraphLayout::Compressed) => {
                Inner::FixedU8StreamVByte(RerankIndex::load_index(path).map_err(load_err)?)
            }
            (SparseFirstStage::Blocked, GraphLayout::Standard | GraphLayout::Permuted) => {
                Inner::Blocked(RerankIndex::load_index(path).map_err(load_err)?)
            }
            (SparseFirstStage::Blocked, GraphLayout::Compressed) => {
                return invalid(BLOCKED_COMPRESSED);
            }
        };
        Ok(Self {
            inner,
            residual_rerank,
        })
    }

    /// Writes the first-stage graph and the rerank dataset to one file.
    pub fn save(&self, path: &str) -> Result<()> {
        with_index!(&self.inner, index => index.save_index(path))
            .map_err(|e| RerankError(format!("error saving index: {e:?}")))
    }

    /// Whether the rerank stage holds the residual components.
    pub fn residual_rerank(&self) -> bool {
        self.residual_rerank
    }

    /// Number of vectors.
    pub fn len(&self) -> usize {
        with_index!(&self.inner, index => index.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes held by the first-stage index and the rerank dataset.
    pub fn space_usage_bytes(&self) -> usize {
        with_index!(&self.inner, index => {
            index.first_stage_index().space_usage_bytes() + index.rerank_dataset().space_usage_bytes()
        })
    }
}

/// Two-stage search: the same query is used for both stages. Query components outside the
/// collection's dimensionality are ignored, since they contribute nothing to a dot product.
impl Index for SparseRerankIndex {
    type Query<'q> = SparseVectorView<'q, u16, f32>;
    type Distance = DotProduct;
    type SearchParams = RerankSearchParams;

    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        params: &Self::SearchParams,
    ) -> Vec<ScoredVector<DotProduct>> {
        let dim = IndexStats::dim(self);
        let (kept_c, kept_v): (Vec<u16>, Vec<f32>);
        let query = if query.components().iter().any(|&c| c as usize >= dim) {
            (kept_c, kept_v) = query
                .components()
                .iter()
                .zip(query.values())
                .filter(|&(&c, _)| (c as usize) < dim)
                .unzip();
            SparseVectorView::new(&kept_c, &kept_v)
        } else {
            query
        };
        // Both first-stage encoders take `()` query parameters.
        let first_stage_params = HNSWSearchConfiguration::default()
            .with_ef_search(params.ef_search)
            .with_early_termination(params.early_termination);
        with_index!(&self.inner, index => index.search(
            query,
            query,
            params.k_candidates,
            k,
            &first_stage_params,
            &(),
            params.alpha,
            params.beta,
            self.residual_rerank,
        ))
    }
}

impl IndexStats for SparseRerankIndex {
    fn n_elements(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        with_index!(&self.inner, index => index.rerank_dataset().input_dim())
    }
}
