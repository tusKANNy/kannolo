use std::collections::BinaryHeap;
use std::marker::PhantomData;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index as rand_index;
use serde::{Deserialize, Serialize};
use vectorium::core::dataset::{ConvertFrom, ScoredItemGeneric, ScoredVector};
use vectorium::core::index::Index;
use vectorium::core::vector_encoder::{VectorEncoder, WithDistance};
use vectorium::distances::DotProduct;
use vectorium::{
    CentroidIndexBuilder, Dataset, DatasetGrowable, Distance, FlatIndex, IndexSerializer, KMeans,
    KMeansBuilder, MeanDataset, QueryEvaluator, ScalarDenseDataset, ScalarDenseDatasetGrowable,
    ScalarSparseDataset, ScalarSparseDatasetGrowable, Solver, SpaceUsage, SquaredEuclideanDistance,
    VectorId,
};

use crate::graph::Graph;
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};

/// The distance reported by an IVF over centroid/query dataset `DQ`.
type Reported<DQ> = <<DQ as Dataset>::Encoder as VectorEncoder>::Distance;

// ---------------------------------------------------------------------------
// ReportedMetric
// ---------------------------------------------------------------------------

/// The metric [`IVF::search`] reports. The cluster dataset **always** stores
/// dot-product vectors, so every candidate contributes a raw dot product
/// `raw = ⟨q, w⟩` (where `w` is the stored vector `v`, or the residual `r`).
/// The final reported distance is reconstructed from three pieces:
///
/// `score = centroid + correction[pos] + β·raw`
///
/// - `centroid` — the cluster centroid's contribution (`⟨q,c⟩` or `‖q−c‖²`),
///   added only in *residual* mode (see [`IVF::is_residual`]); `0.0` otherwise.
/// - `correction[pos]` — per-vector offline term (L2 only, see below); `0.0`
///   for dot product.
/// - `β` — `+1` for dot product, `−2` for squared Euclidean.
///
/// The metric is a compile-time type (so `β`, the output type, and correction
/// use are resolved at compile time), while *residual-vs-plain* is a runtime
/// flag on the index.
///
/// ## Squared-Euclidean correction & the `‖q‖²` omission
///
/// `‖q − w'‖² = ‖q‖² + ‖w'‖² − 2⟨q,w'⟩`. We drop the query-constant `‖q‖²` (it
/// does not affect ranking) and fold the rest into `correction`:
///
/// - **plain**  (`w' = v`, `centroid = 0`): `correction[pos] = ‖v‖²`  →
///   `score = ‖v‖² − 2⟨q,v⟩`  (equals `‖q−v‖² − ‖q‖²`; add `‖q‖²` per query if
///   the *exact* squared distance is required — ranking is identical without it).
/// - **residual** (`w' = c + r̂`, `centroid = ‖q−c‖²`):
///   `correction[pos] = ‖r̂‖² + 2⟨c,r̂⟩`  →  `score = ‖q−c‖² + ‖r̂‖²+2⟨c,r̂⟩ − 2⟨q,r̂⟩`
///   `= ‖q − (c+r̂)‖²`. The correction must use the *reconstructed* `r̂` (the same
///   residual reconstruction the search scores against); mixing exact `‖r‖²` with
///   the approximate `⟨q,r̂⟩` leaks a `2⟨q, r−r̂⟩` ranking error.
pub trait ReportedMetric: Distance {
    /// Whether this metric consumes the per-vector `correction` term.
    const USES_CORRECTION: bool;

    /// Reconstruct the reported distance from the centroid contribution, the
    /// per-vector correction, and the raw dot product `⟨q, w⟩`.
    fn combine(centroid: f32, correction: f32, raw_dot: f32) -> Self;
}

impl ReportedMetric for SquaredEuclideanDistance {
    const USES_CORRECTION: bool = true;

    #[inline]
    fn combine(centroid: f32, correction: f32, raw_dot: f32) -> Self {
        SquaredEuclideanDistance::from(centroid + correction - 2.0 * raw_dot)
    }
}

impl ReportedMetric for DotProduct {
    const USES_CORRECTION: bool = false;

    #[inline]
    fn combine(centroid: f32, _correction: f32, raw_dot: f32) -> Self {
        DotProduct::from(centroid + raw_dot)
    }
}

// ---------------------------------------------------------------------------
// IVF
// ---------------------------------------------------------------------------

/// Inverted Vector File (IVF) index: a centroid index + a dataset for all clusters.
///
/// Conceptually a pair **(CI, Q)**: a centroid index `CI` and a quantizer `Q`
/// (the encoder of the cluster dataset `D`). Whether the dataset stores the
/// original vectors or residuals `r = v − c` is a **runtime** property
/// ([`is_residual`](Self::is_residual)) — *not* part of the type — so a single build
/// path and a single search cover both. The cluster dataset always uses
/// **dot-product** storage; the reported metric is `DQ`'s metric (see
/// [`ReportedMetric`]).
///
/// Every configuration implements vectorium's [`Index`] over the **query/input
/// dataset** `DQ` (whose metric is the reported metric), and also exposes the
/// inherent [`search`](IVF::search).
///
/// ## Type parameters
/// - `DQ`: centroid / query dataset; `DQ`'s metric is the reported metric.
/// - `D`:  cluster dataset (dot-product storage: f32/f16/PQ).
/// - `CI`: centroid index — e.g. `FlatIndex<DQ>` or `HNSW<DQ, Graph>`.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "CI: Serialize, D: Serialize",
    deserialize = "CI: for<'de2> Deserialize<'de2>, D: for<'de2> Deserialize<'de2>"
))]
pub struct IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    centroid_index: CI,
    dataset: D,
    /// CSR offsets: cluster `c` owns `dataset[cluster_offsets[c]..cluster_offsets[c+1]]`.
    /// Length is `n_clusters + 1`.
    cluster_offsets: Box<[u32]>,
    /// `external_ids[i]` = original dataset ID of the vector at reordered position `i`.
    /// Only accessed to label the final top-`k` results.
    external_ids: Box<[u32]>,
    /// Per-vector L2 correction term (`‖v‖²` for plain, `‖r̂‖²+2⟨c,r̂⟩` for
    /// residual). `Some` for squared-Euclidean indexes, `None` for dot product.
    correction_terms: Option<Box<[f32]>>,
    /// `true` when `dataset` stores residuals `r = v − c` (so the centroid
    /// contribution is added at search time); `false` when it stores `v`.
    residual: bool,
    _phantom: PhantomData<DQ>,
}

impl<DQ, D, CI> IndexSerializer for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
}

/// Search parameters for [`IVF`].
///
/// `P` is the cluster dataset encoder's [`VectorEncoder::QueryParams`], defaulting to `()`
/// for the encoders that take no query-side configuration.
pub struct IVFSearchParams<CI, P = ()>
where
    CI: Index,
{
    /// Number of nearest centroids to probe.
    pub nprobe: usize,
    pub centroid_search_params: CI::SearchParams,
    /// Query-side parameters for the cluster dataset's encoder, forwarded to
    /// [`VectorEncoder::query_evaluator`].
    pub cluster_query_params: P,
}

// ---------------------------------------------------------------------------
// Core methods
// ---------------------------------------------------------------------------

impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    pub fn n_clusters(&self) -> usize {
        self.cluster_offsets.len().saturating_sub(1)
    }

    pub fn n_elements(&self) -> usize {
        self.external_ids.len()
    }

    /// Whether the cluster dataset stores residuals `r = v − c` (`true`) or the
    /// original vectors `v` (`false`).
    pub fn is_residual(&self) -> bool {
        self.residual
    }

    pub fn print_space_usage_bytes(&self) {
        let offsets = self.cluster_offsets.len() * std::mem::size_of::<u32>();
        let ids = self.external_ids.len() * std::mem::size_of::<u32>();
        let corr = self.correction_terms.as_ref().map_or(0, |c| c.len() * 4);
        println!("IVF: cluster_offsets={offsets} B  external_ids={ids} B  correction={corr} B");
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
    // The centroid index accepts the same (raw f32) query the IVF does.
    for<'q> CI: Index<Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>>,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    D::Encoder: VectorEncoder<Distance = DotProduct>,
{
    /// Search the index.
    ///
    /// The reported distance is `DQ`'s metric ([`ReportedMetric`]); the cluster
    /// dataset stores dot products, and residual-vs-plain is handled at runtime
    /// via [`is_residual`](Self::is_residual). The per-candidate `residual` and
    /// `USES_CORRECTION` decisions fold to constants, so the hot loop stays a
    /// single `centroid + correction − 2·raw` (L2) or `centroid + raw` (dot).
    pub fn search<'q>(
        &self,
        query: <DQ::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        params: &IVFSearchParams<CI, <D::Encoder as VectorEncoder>::QueryParams>,
    ) -> Vec<ScoredVector<<DQ::Encoder as VectorEncoder>::Distance>>
    where
        <DQ::Encoder as VectorEncoder>::QueryVector<'q>:
            Into<<D::Encoder as VectorEncoder>::QueryVector<'q>> + Copy,
    {
        let nearest_centroids =
            self.centroid_index
                .search(query, params.nprobe, &params.centroid_search_params);

        let evaluator = self
            .dataset
            .encoder()
            .query_evaluator(query.into(), &params.cluster_query_params);
        let correction: &[f32] = self.correction_terms.as_deref().unwrap_or(&[]);
        let residual = self.residual;
        let uses_correction = <Reported<DQ> as ReportedMetric>::USES_CORRECTION;

        type HeapItem<Dd> = ScoredItemGeneric<Dd, u32>;
        let mut heap: BinaryHeap<HeapItem<Reported<DQ>>> = BinaryHeap::with_capacity(k + 1);

        // Turn the raw dot product into the reported distance.
        let push = |heap: &mut BinaryHeap<HeapItem<Reported<DQ>>>,
                    pos: usize,
                    centroid: f32,
                    raw: DotProduct| {
            let corr = if uses_correction {
                correction[pos]
            } else {
                0.0
            };
            let dist = <Reported<DQ> as ReportedMetric>::combine(centroid, corr, raw.distance());
            let item = HeapItem {
                distance: dist,
                vector: pos as u32,
            };
            if heap.len() < k {
                heap.push(item);
            } else if item < *heap.peek().unwrap() {
                heap.pop();
                heap.push(item);
            }
        };

        for scored_centroid in &nearest_centroids {
            let cluster_id = scored_centroid.vector as usize;
            // Centroid contribution is added only in residual mode.
            let centroid = if residual {
                scored_centroid.distance.distance()
            } else {
                0.0
            };
            let start = self.cluster_offsets[cluster_id] as usize;
            let end = self.cluster_offsets[cluster_id + 1] as usize;

            let batch_end = start + ((end - start) / 6) * 6;
            let mut pos = start;
            while pos < batch_end {
                let dists = evaluator.compute_distances_batch6([
                    self.dataset.get(pos as VectorId),
                    self.dataset.get((pos + 1) as VectorId),
                    self.dataset.get((pos + 2) as VectorId),
                    self.dataset.get((pos + 3) as VectorId),
                    self.dataset.get((pos + 4) as VectorId),
                    self.dataset.get((pos + 5) as VectorId),
                ]);
                for (i, &raw) in dists.iter().enumerate() {
                    push(&mut heap, pos + i, centroid, raw);
                }
                pos += 6;
            }
            for p in batch_end..end {
                let raw = evaluator.compute_distance(self.dataset.get(p as VectorId));
                push(&mut heap, p, centroid, raw);
            }
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|item| ScoredItemGeneric {
                distance: item.distance,
                vector: self.external_ids[item.vector as usize] as VectorId,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    /// Construct from pre-built components.
    ///
    /// - `residual = false`: `dataset` stores the (dot-product-encoded) original
    ///   vectors `v`. For L2, `correction_terms[i] = ‖v‖²`; for dot product,
    ///   `correction_terms = None`.
    /// - `residual = true`: `dataset` stores residuals `r = v − c`. For L2,
    ///   `correction_terms[i] = ‖r̂‖² + 2⟨c,r̂⟩`; for dot product, `None`.
    ///
    /// `cluster_offsets` must be CSR-format with length `n_clusters + 1`.
    /// `external_ids[i]` is the original ID of the vector at reordered position `i`.
    pub fn from_parts(
        centroid_index: CI,
        dataset: D,
        cluster_offsets: Box<[u32]>,
        external_ids: Box<[u32]>,
        correction_terms: Option<Box<[f32]>>,
        residual: bool,
    ) -> Self {
        debug_assert!(!cluster_offsets.is_empty());
        debug_assert_eq!(
            *cluster_offsets.last().unwrap() as usize,
            external_ids.len()
        );
        debug_assert!(
            correction_terms
                .as_ref()
                .is_none_or(|c| c.len() == external_ids.len())
        );
        Self {
            centroid_index,
            dataset,
            cluster_offsets,
            external_ids,
            correction_terms,
            residual,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// Number of clusters from which [`AnnAssignment::Auto`] assigns vectors to centroids through an
/// ANN index over the centroids.
///
/// Below it, an exhaustive scan over the centroids is used.
pub const KMEANS_ANN_THRESHOLD: usize = 4_096;

/// Whether k-means assigns vectors to centroids through an ANN index over the centroids instead
/// of an exhaustive scan.
///
/// The index is chosen by solver: **HNSW for Lloyd**, which refreshes it
/// once per epoch, so search speed decides; **an IVF over the centroids for e-means**
/// ([`IvfCentroidIndexBuilder`]), which refreshes it after every
/// mini-batch, so refresh cost decides and the IVF only re-files the moved centroids. The final
/// assignment of every vector uses an HNSW for both solvers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum AnnAssignment {
    /// On from [`KMEANS_ANN_THRESHOLD`] clusters, off below.
    #[default]
    Auto,
    /// Always use an ANN index.
    On,
    /// Always scan every centroid.
    Off,
}

impl AnnAssignment {
    /// Whether to use an ANN index for `n_clusters` clusters.
    pub fn enabled(self, n_clusters: usize) -> bool {
        match self {
            AnnAssignment::Auto => n_clusters >= KMEANS_ANN_THRESHOLD,
            AnnAssignment::On => true,
            AnnAssignment::Off => false,
        }
    }
}

/// HNSW parameters used for centroid assignment during k-means.
/// If you want better quality, increase ef_search and eventually ef_construction.
const KMEANS_HNSW_M: usize = 32;
const KMEANS_HNSW_EF_CONSTRUCTION: usize = 200;
const KMEANS_HNSW_EF_SEARCH: usize = 16;

fn kmeans_hnsw_build_config() -> HNSWBuildConfiguration {
    HNSWBuildConfiguration::default()
        .with_num_neighbors(KMEANS_HNSW_M)
        .with_ef_construction(KMEANS_HNSW_EF_CONSTRUCTION)
}

fn kmeans_hnsw_search_config() -> HNSWSearchConfiguration {
    HNSWSearchConfiguration::default().with_ef_search(KMEANS_HNSW_EF_SEARCH)
}

/// The clustering produced by [`prepare_ivf`]: centroids plus the order that groups the vectors
/// into contiguous clusters. Materialize the cluster vectors with [`reorder`](Self::reorder)
/// (or [`IVFInput::residuals`]), encode them, and pass everything to [`IVF::from_parts`].
pub struct BuildArtifacts<C> {
    /// k-means centroids, under the clustering metric. Convert them to the centroid index's
    /// metric and precision to build the centroid index.
    pub centroids: C,
    /// CSR offsets, length `n_clusters + 1`.
    pub cluster_offsets: Box<[u32]>,
    /// `external_ids[i]` = original vector ID at reordered position `i`.
    pub external_ids: Box<[u32]>,
}

impl<C> BuildArtifacts<C> {
    /// `dataset` reordered so that each cluster occupies a contiguous range.
    pub fn reorder<D: Dataset>(&self, dataset: &D) -> D::Owned {
        // `permute` moves the row at `old` to `permutation[old]`.
        let mut permutation = vec![0usize; self.external_ids.len()];
        for (new_pos, &orig) in self.external_ids.iter().enumerate() {
            permutation[orig as usize] = new_pos;
        }
        dataset.permute(&permutation)
    }

    /// Like [`reorder`](Self::reorder), but each vector `v` is replaced by its residual
    /// `v − c` against its cluster's centroid, pushed into a `G` built with `encoder`.
    pub fn residuals<D, G>(&self, dataset: &D, encoder: G::Encoder) -> G::Owned
    where
        C: MeanDataset<Vector = D::Vector>,
        D: MeanDataset,
        G: DatasetGrowable + Into<G::Owned>,
        for<'a> <G::Encoder as VectorEncoder>::InputVector<'a>: From<&'a D::Vector>,
    {
        let offsets = &self.cluster_offsets;
        let mut residuals = G::with_capacity(encoder, self.external_ids.len());
        for cid in 0..offsets.len() - 1 {
            let c = self.centroids.vector(cid as u32);
            for &orig in &self.external_ids[offsets[cid] as usize..offsets[cid + 1] as usize] {
                residuals.push((&dataset.residual(orig, &c)).into());
            }
        }
        residuals.into()
    }
}

/// Run k-means and assign every vector of `dataset` to its nearest centroid.
///
/// The clustering metric is `centroid_encoder`'s, independent of `dataset`'s.
///
/// `sample_size`: if `Some(s)` and `s < n`, k-means is trained on a random
/// sample of `s` vectors drawn without replacement from `dataset`. Assignment
/// of all `n` vectors to the resulting centroids always uses the full dataset.
/// A sample of ~256 × n_clusters is typically sufficient for good centroid
/// quality and reduces training time from O(n · k) to O(s · k) per iteration.
///
/// `ann_assignment`: assign vectors to centroids through an ANN index over the centroids instead
/// of a flat scan (resolve it with [`AnnAssignment::enabled`]). During k-means the index is an
/// HNSW for Lloyd and an IVF over the centroids for e-means ([`IvfCentroidIndexBuilder`]); the
/// final pass over all vectors always uses an HNSW.
///
/// `refine` runs on every centroid after each k-means update, with the IDs (in `dataset`) of
/// the vectors assigned to it; see [`vectorium::sparse_centroid_pruning`]. Pass `|_, _| {}`
/// for none.
pub fn prepare_ivf<D, G>(
    dataset: &D,
    n_clusters: usize,
    kmeans_builder: KMeansBuilder,
    centroid_encoder: G::Encoder,
    sample_size: Option<usize>,
    ann_assignment: bool,
    refine: impl Fn(&[u32], &mut D::Vector) + Sync,
) -> BuildArtifacts<G::Owned>
where
    D: MeanDataset + Sync,
    G: DatasetGrowable + MeanDataset<Vector = D::Vector> + Clone + Sync + Into<G::Owned>,
    G::Owned: Clone + Sync + SpaceUsage + IvfCentroids,
    G::Encoder: VectorEncoder<QueryParams = ()> + Clone,
    <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    for<'a> <G::Encoder as VectorEncoder>::InputVector<'a>: From<&'a D::Vector>,
    for<'a> <G::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a D::Vector>,
{
    let n = dataset.len();
    let kmeans = kmeans_builder.build();

    let centroids: G::Owned = match sample_size.filter(|&s| s < n) {
        Some(ss) => {
            println!("Sampling {ss} / {n} vectors for k-means training");
            let mut rng = StdRng::from_entropy();
            let ids: Vec<u32> = rand_index::sample(&mut rng, n, ss)
                .into_iter()
                .map(|i| i as u32)
                .collect();
            let mut sample = G::with_capacity(centroid_encoder.clone(), ss);
            for &i in &ids {
                sample.push((&dataset.vector(i)).into());
            }
            // `refine` takes IDs in `dataset`: translate the sample's.
            train_centroids::<_, G>(
                &kmeans,
                &sample,
                n_clusters,
                centroid_encoder,
                ann_assignment,
                |members: &[u32], centroid: &mut D::Vector| {
                    let members: Vec<u32> = members.iter().map(|&m| ids[m as usize]).collect();
                    refine(&members, centroid)
                },
            )
        }
        None => train_centroids::<_, G>(
            &kmeans,
            dataset,
            n_clusters,
            centroid_encoder,
            ann_assignment,
            &refine,
        ),
    };

    let assignments = if ann_assignment {
        let centroid_index =
            HNSW::<G::Owned, Graph>::build_index(centroids.clone(), &kmeans_hnsw_build_config());
        KMeans::assign_with_index(dataset, &centroid_index, &kmeans_hnsw_search_config())
    } else {
        KMeans::compute_assignments(dataset, &centroids)
    };

    group_by_centroid(centroids, n_clusters, &assignments)
}

/// Groups vectors by their assigned centroid: CSR offsets plus the order that makes each
/// cluster contiguous.
fn group_by_centroid<C>(
    centroids: C,
    n_clusters: usize,
    assignments: &[(f32, usize)],
) -> BuildArtifacts<C> {
    let mut cluster_offsets = vec![0u32; n_clusters + 1];
    for &(_, c) in assignments {
        cluster_offsets[c + 1] += 1;
    }
    for c in 0..n_clusters {
        cluster_offsets[c + 1] += cluster_offsets[c];
    }
    // Unstable sort: order within a cluster is arbitrary and irrelevant for correctness.
    let mut order: Vec<u32> = (0..assignments.len() as u32).collect();
    order.sort_unstable_by_key(|&i| assignments[i as usize].1);
    BuildArtifacts {
        centroids,
        cluster_offsets: cluster_offsets.into_boxed_slice(),
        external_ids: order.into_boxed_slice(),
    }
}

/// Assigns every vector of `dataset` to its nearest vector of `centroids` (exhaustive scan,
/// under `centroids`' metric) and groups the vectors by centroid: the clustering of
/// [`prepare_ivf`] for centroids that are already known.
pub fn assign_to_centroids<D, C>(dataset: &D, centroids: C) -> BuildArtifacts<C>
where
    D: MeanDataset + Sync,
    C: Dataset + Sync,
    C::Encoder: VectorEncoder<QueryParams = ()>,
    <C::Encoder as VectorEncoder>::Distance: Distance,
    for<'a> &'a D::Vector: Into<<C::Encoder as VectorEncoder>::QueryVector<'a>>,
{
    let assignments = KMeans::compute_assignments(dataset, &centroids);
    let n_clusters = centroids.len();
    group_by_centroid(centroids, n_clusters, &assignments)
}

/// k-means on `training`, with flat or ANN centroid assignment (see [`AnnAssignment`] for which
/// index each solver uses).
fn train_centroids<T, G>(
    kmeans: &KMeans,
    training: &T,
    n_clusters: usize,
    centroid_encoder: G::Encoder,
    ann_assignment: bool,
    refine: impl Fn(&[u32], &mut T::Vector) + Sync,
) -> G::Owned
where
    T: MeanDataset + Sync,
    G: DatasetGrowable + MeanDataset<Vector = T::Vector> + Clone + Sync + Into<G::Owned>,
    G::Owned: Sync + SpaceUsage + IvfCentroids,
    G::Encoder: VectorEncoder<QueryParams = ()> + Clone,
    <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    for<'a> <G::Encoder as VectorEncoder>::InputVector<'a>: From<&'a T::Vector>,
    for<'a> <G::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a T::Vector>,
{
    let centroids: G = if ann_assignment && !matches!(kmeans.solver(), Solver::Lloyd) {
        kmeans.train_with_index_refined(
            training,
            n_clusters,
            None,
            centroid_encoder,
            IvfCentroidIndexBuilder::default(),
            &<G::Owned as IvfCentroids>::ivf_search_params(CENTROID_IVF_N_PROBE),
            refine,
        )
    } else if ann_assignment {
        let build_cfg = kmeans_hnsw_build_config();
        kmeans.train_with_index_refined(
            training,
            n_clusters,
            None,
            centroid_encoder,
            |centroids: G| HNSW::<G::Owned, Graph>::build_index(centroids.into(), &build_cfg),
            &kmeans_hnsw_search_config(),
            refine,
        )
    } else {
        kmeans.train_refined(training, n_clusters, None, centroid_encoder, refine)
    };
    centroids.into()
}

/// Encode the reordered f32 vectors `stored` (see [`IVFInput::storage`]) as cluster storage `D`
/// under the encoder configuration `config` and, when the reported metric `M` needs one
/// ([`ReportedMetric::USES_CORRECTION`]), compute the per-vector L2 correction.
///
/// `query_params` are the cluster encoder's query parameters used to compute the correction
/// (see [`l2_correction`]). `centroids` **must be at the centroid-index precision** (e.g. the
/// f16 centroid dataset itself), so the correction's centroid terms match the `‖q − c‖²` the
/// search reads from the centroid index.
pub fn encode_clusters<M, S, D, Cfg, C>(
    stored: &S,
    config: Cfg,
    query_params: &<D::Encoder as VectorEncoder>::QueryParams,
    centroids: &C,
    offsets: &[u32],
    residual: bool,
) -> (D, Option<Box<[f32]>>)
where
    M: ReportedMetric,
    S: Dataset,
    D: Dataset + for<'a> ConvertFrom<&'a S, Config = Cfg>,
    D::Encoder: WithDistance<SquaredEuclideanDistance>,
    C: MeanDataset,
    for<'a> <D::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a C::Vector>,
{
    let cluster = D::convert_from(stored, config);
    let correction = M::USES_CORRECTION
        .then(|| l2_correction(&cluster, query_params, centroids, offsets, residual));
    (cluster, correction)
}

/// Per-vector L2 correction `‖c + w‖² − ‖c‖² = ‖w‖² + 2⟨c, w⟩` for each stored vector `w`
/// (`c = 0` for plain storage), so the reported score `‖q − c‖² + correction − 2⟨q, w⟩` is
/// `‖q − (c + w)‖²`.
///
/// `‖c + w‖²` is the cluster encoder's own L2 distance between the query `−c` and the stored
/// `w` (via [`WithDistance`]), so it refers to the same `w` the dot-product scan scores: the
/// reconstruction for scalar / PQ storage (where the result is exact), the encoder's estimate
/// for estimators such as RaBitQ (where the code errors of the two terms cancel).
fn l2_correction<D, C>(
    cluster: &D,
    query_params: &<D::Encoder as VectorEncoder>::QueryParams,
    centroids: &C,
    offsets: &[u32],
    residual: bool,
) -> Box<[f32]>
where
    D: Dataset,
    D::Encoder: WithDistance<SquaredEuclideanDistance>,
    C: MeanDataset,
    for<'a> <D::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a C::Vector>,
{
    let l2 = WithDistance::<SquaredEuclideanDistance>::with_distance(cluster.encoder());
    let n_clusters = offsets.len() - 1;
    let mut corr = vec![0.0f32; cluster.len()].into_boxed_slice();
    for cid in 0..n_clusters {
        // The query −c (or 0 for plain storage), built from the centroid's own values.
        let mut neg_c = centroids.vector(cid as u32);
        let values = neg_c.as_mut();
        let c_norm: f32 = if residual {
            values.iter().map(|&x| x * x).sum()
        } else {
            0.0
        };
        for x in values.iter_mut() {
            *x = if residual { -*x } else { 0.0 };
        }
        let evaluator = l2.query_evaluator((&neg_c).into(), query_params);
        for pos in offsets[cid] as usize..offsets[cid + 1] as usize {
            corr[pos] = evaluator
                .compute_distance(cluster.get(pos as VectorId))
                .distance()
                - c_norm;
        }
    }
    corr
}

// ---------------------------------------------------------------------------
// vectorium Index trait (all configurations)
// ---------------------------------------------------------------------------

/// Build parameters for [`IVF::build_index`]. The centroid index itself is
/// supplied separately, via a `centroid_index_builder` closure passed to
/// `build_index` — see [`IVF::build_index`].
///
/// Carries the k-means configuration (rebuilt into a [`KMeansBuilder`] at build
/// time so `build_index` can take `&self`) plus whether to store residuals.
pub struct IVFBuildParams {
    pub n_clusters: usize,
    /// k-means solver for the coarse centroids (`Solver::default()` is vectorium's default).
    pub kmeans_solver: Solver,
    /// Epoch budget; `None` uses the solver's default ([`Solver::default_epochs`]).
    pub kmeans_epochs: Option<f64>,
    pub kmeans_n_redo: usize,
    pub kmeans_seed: Option<u64>,
    pub kmeans_spherical: bool,
    /// ANN-accelerated centroid assignment during k-means (default: [`AnnAssignment::Auto`]).
    pub ann_assignment: AnnAssignment,
    /// Store residuals `r = v − c` (`true`) or original vectors (`false`).
    pub residual: bool,
}

impl<DQ, D, CI> vectorium::IndexStats for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    fn n_elements(&self) -> usize {
        self.external_ids.len()
    }
    fn dim(&self) -> usize {
        self.dataset.input_dim()
    }
}

/// A single `Index` impl covering every IVF configuration: the bounds are
/// exactly those the inherent `search` above already needs (`D` generically,
/// via `D::Encoder: VectorEncoder<Distance = DotProduct>`). Unlike
/// `build_index`, `search` never needs to name a concrete encoder like
/// `ScalarDenseQuantizer` or `ProductQuantizer` — that specificity is only
/// needed at build time.
impl<DQ, D, CI> Index for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
    for<'q> CI: Index<Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>>,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    D::Encoder: VectorEncoder<Distance = DotProduct>,
    for<'q> <DQ::Encoder as VectorEncoder>::QueryVector<'q>:
        Into<<D::Encoder as VectorEncoder>::QueryVector<'q>> + Copy,
    for<'q, 'b> <D::Encoder as VectorEncoder>::Evaluator<'q>:
        QueryEvaluator<<D::Encoder as VectorEncoder>::EncodedVector<'b>, Distance = DotProduct>,
{
    type Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>;
    type Distance = <DQ::Encoder as VectorEncoder>::Distance;
    type SearchParams = IVFSearchParams<CI, <D::Encoder as VectorEncoder>::QueryParams>;

    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<Self::Distance>> {
        // Inherent `search` takes priority, so this is not recursive.
        self.search(query, k, search_params)
    }
}

impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: MeanDataset + Sync,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    D: Dataset,
    CI: Index,
{
    /// Build an IVF.
    ///
    /// - `centroid_encoder` builds the k-means centroids `G`; its metric is the clustering
    ///   metric (dense IVF usually clusters under squared Euclidean whatever it reports).
    /// - Clusters are stored by encoding `S`, the reordered vectors (or residuals) as an f32
    ///   dot-product dataset, with `cluster_config` (`()` for scalar and PQ storage, e.g. a
    ///   `RabitqConfig` for RaBitQ). `cluster_query_params` are the storage encoder's query
    ///   parameters for computing the L2 correction (see [`encode_clusters`]).
    /// - `centroid_index_builder` builds the centroid index from the post-k-means centroids,
    ///   relabelled to `DQ` — how it does so is entirely up to the caller.
    ///
    /// For control over sampling or centroid refinement, use [`prepare_ivf`],
    /// [`encode_clusters`] and [`IVF::from_parts`].
    #[allow(clippy::too_many_arguments)]
    pub fn build_index<G, S, Cfg>(
        dataset: DQ,
        p: &IVFBuildParams,
        centroid_encoder: G::Encoder,
        cluster_config: Cfg,
        cluster_query_params: &<D::Encoder as VectorEncoder>::QueryParams,
        centroid_index_builder: impl FnOnce(DQ) -> CI,
    ) -> Self
    where
        G: DatasetGrowable + MeanDataset<Vector = DQ::Vector> + Clone + Sync + Into<G::Owned>,
        G::Owned: MeanDataset<Vector = DQ::Vector>
            + Clone
            + Sync
            + SpaceUsage
            + IvfCentroids
            + Into<DQ>
            + Into<S>,
        G::Encoder: VectorEncoder<QueryParams = ()> + Clone,
        <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
        for<'a> <G::Encoder as VectorEncoder>::InputVector<'a>: From<&'a DQ::Vector>,
        for<'a> <G::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a DQ::Vector>,
        DQ::Owned: Into<S>,
        S: Dataset,
        D: for<'a> ConvertFrom<&'a S, Config = Cfg>,
        D::Encoder: WithDistance<SquaredEuclideanDistance>,
        for<'a> <D::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a DQ::Vector>,
    {
        let mut kmeans = KMeansBuilder::new().solver(p.kmeans_solver.clone());
        if let Some(epochs) = p.kmeans_epochs {
            kmeans = kmeans.epochs(epochs);
        }
        let kmeans = kmeans
            .n_redo(p.kmeans_n_redo)
            .seed(p.kmeans_seed)
            .spherical(p.kmeans_spherical);
        let a = prepare_ivf::<DQ, G>(
            &dataset,
            p.n_clusters,
            kmeans,
            centroid_encoder.clone(),
            None,
            p.ann_assignment.enabled(p.n_clusters),
            |_, _| {},
        );
        let stored: S = if p.residual {
            a.residuals::<DQ, G>(&dataset, centroid_encoder).into()
        } else {
            a.reorder(&dataset).into()
        };
        let (cluster, correction) = encode_clusters::<Reported<DQ>, _, D, _, _>(
            &stored,
            cluster_config,
            cluster_query_params,
            &a.centroids,
            &a.cluster_offsets,
            p.residual,
        );
        drop(stored);
        IVF::from_parts(
            centroid_index_builder(a.centroids.into()),
            cluster,
            a.cluster_offsets,
            a.external_ids,
            correction,
            p.residual,
        )
    }
}

// ---------------------------------------------------------------------------
// IVF over a fixed coarse quantizer, and as the centroid index of e-means
// ---------------------------------------------------------------------------

impl<DQ, D> IVF<DQ, D, FlatIndex<DQ>>
where
    DQ: MeanDataset<Owned = DQ> + Sync,
    DQ::Encoder: VectorEncoder<QueryParams = ()>,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    for<'a> &'a DQ::Vector: Into<<DQ::Encoder as VectorEncoder>::QueryVector<'a>>,
    D: Dataset,
    D::Encoder: VectorEncoder<QueryParams = ()> + WithDistance<SquaredEuclideanDistance>,
    for<'a> <D::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a DQ::Vector>,
{
    /// Builds a plain (non-residual) IVF that files `vectors` under the given coarse
    /// quantizer `coarse` (searched exhaustively): each vector goes to its nearest coarse
    /// centroid, and the lists are stored as `D` via the dot-product relabelling `S`.
    pub fn over_centroids<S>(coarse: DQ, vectors: &DQ) -> Self
    where
        S: Dataset + From<DQ>,
        D: for<'a> ConvertFrom<&'a S, Config = ()>,
    {
        let a = assign_to_centroids(vectors, coarse);
        let stored: S = a.reorder(vectors).into();
        let (cluster, correction) = encode_clusters::<Reported<DQ>, S, D, (), DQ>(
            &stored,
            (),
            &(),
            &a.centroids,
            &a.cluster_offsets,
            false,
        );
        IVF::from_parts(
            FlatIndex::from(a.centroids),
            cluster,
            a.cluster_offsets,
            a.external_ids,
            correction,
            false,
        )
    }

    /// Re-files `vectors` (for instance, centroids that moved) under this IVF's coarse
    /// quantizer, which is kept as is. Costs one exhaustive assignment against the coarse
    /// quantizer plus one encoding pass over `vectors`.
    pub fn refile<S>(self, vectors: &DQ) -> Self
    where
        S: Dataset + From<DQ>,
        D: for<'a> ConvertFrom<&'a S, Config = ()>,
    {
        Self::over_centroids::<S>(self.centroid_index.into_dataset(), vectors)
    }
}

/// Lists probed per query when an IVF serves as the centroid index of e-means.
pub const CENTROID_IVF_N_PROBE: usize = 16;

/// Lists of an IVF over `k` centroids: `4·sqrt(k)` rounded up to a power of two, capped at `k`.
pub fn centroid_ivf_n_lists(k: usize) -> usize {
    ((4.0 * (k as f64).sqrt()).ceil() as usize)
        .next_power_of_two()
        .clamp(1, k.max(1))
}

/// Centroid datasets that an [`IVF`] can index, so that e-means can assign vectors through it.
///
/// Names the IVF type (dot-product cluster storage `D`, exhaustive coarse search) for each
/// centroid representation, and builds and refreshes it. Implemented for the dense and sparse
/// scalar datasets kANNolo and TACHIOM cluster with.
pub trait IvfCentroids: MeanDataset + Sized {
    /// The IVF over these centroids.
    type Ivf: for<'q> Index<
            Query<'q> = <<Self as Dataset>::Encoder as VectorEncoder>::QueryVector<'q>,
            SearchParams: Sync,
        > + Sync;

    /// Trains a coarse quantizer of `n_lists` lists on `vectors` and files them into it.
    fn build_ivf(vectors: Self, n_lists: usize, seed: u64) -> Self::Ivf;

    /// Re-files moved `vectors`, keeping the coarse quantizer.
    fn refile_ivf(ivf: Self::Ivf, vectors: Self) -> Self::Ivf;

    /// Search parameters probing `n_probe` lists.
    fn ivf_search_params(n_probe: usize) -> <Self::Ivf as Index>::SearchParams;
}

/// Coarse quantizer of an IVF over `vectors`: Lloyd, 10 epochs, on 20 vectors sampled per list.
fn train_coarse<G>(vectors: &G::Owned, encoder: G::Encoder, n_lists: usize, seed: u64) -> G::Owned
where
    G: DatasetGrowable + MeanDataset + Clone + Sync + Into<G::Owned>,
    G::Owned: MeanDataset<Vector = G::Vector>,
    G::Encoder: VectorEncoder<QueryParams = ()> + Clone,
    <G::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    for<'a> &'a G::Vector: Into<<G::Encoder as VectorEncoder>::InputVector<'a>>
        + Into<<G::Encoder as VectorEncoder>::QueryVector<'a>>,
{
    let k = vectors.len();
    let n_sample = (20 * n_lists).clamp(n_lists, k);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ids = rand_index::sample(&mut rng, k, n_sample).into_vec();
    ids.sort_unstable();
    let mut sample = G::with_capacity(encoder.clone(), n_sample);
    for &i in &ids {
        sample.push((&vectors.vector(i as u32)).into());
    }
    KMeansBuilder::new()
        .solver(Solver::Lloyd)
        .epochs(10.0)
        .seed(Some(seed))
        .build()
        .train::<G, G>(&sample, n_lists, None, encoder)
        .into()
}

macro_rules! impl_ivf_centroids {
    ($owned:ty, $growable:ty, $stored:ty, $cluster:ty) => {
        impl IvfCentroids for $owned {
            type Ivf = IVF<$owned, $cluster, FlatIndex<$owned>>;

            fn build_ivf(vectors: Self, n_lists: usize, seed: u64) -> Self::Ivf {
                let n_lists = n_lists.clamp(1, vectors.len().max(1));
                let coarse =
                    train_coarse::<$growable>(&vectors, vectors.encoder().clone(), n_lists, seed);
                IVF::over_centroids::<$stored>(coarse, &vectors)
            }

            fn refile_ivf(ivf: Self::Ivf, vectors: Self) -> Self::Ivf {
                ivf.refile::<$stored>(&vectors)
            }

            fn ivf_search_params(n_probe: usize) -> <Self::Ivf as Index>::SearchParams {
                IVFSearchParams {
                    nprobe: n_probe,
                    centroid_search_params: (),
                    cluster_query_params: (),
                }
            }
        }
    };
}

macro_rules! impl_ivf_centroids_dense {
    ($vout:ty) => {
        impl_ivf_centroids!(
            ScalarDenseDataset<f32, $vout, SquaredEuclideanDistance>,
            ScalarDenseDatasetGrowable<f32, $vout, SquaredEuclideanDistance>,
            ScalarDenseDataset<f32, $vout, DotProduct>,
            ScalarDenseDataset<$vout, $vout, DotProduct>
        );
        impl_ivf_centroids!(
            ScalarDenseDataset<f32, $vout, DotProduct>,
            ScalarDenseDatasetGrowable<f32, $vout, DotProduct>,
            ScalarDenseDataset<f32, $vout, DotProduct>,
            ScalarDenseDataset<$vout, $vout, DotProduct>
        );
    };
}
impl_ivf_centroids_dense!(f32);
impl_ivf_centroids_dense!(half::f16);

macro_rules! impl_ivf_centroids_sparse {
    ($c:ty) => {
        impl_ivf_centroids!(
            ScalarSparseDataset<$c, f32, f32, SquaredEuclideanDistance>,
            ScalarSparseDatasetGrowable<$c, f32, f32, SquaredEuclideanDistance>,
            ScalarSparseDataset<$c, f32, f32, DotProduct>,
            ScalarSparseDataset<$c, f32, f32, DotProduct>
        );
        impl_ivf_centroids!(
            ScalarSparseDataset<$c, f32, f32, DotProduct>,
            ScalarSparseDatasetGrowable<$c, f32, f32, DotProduct>,
            ScalarSparseDataset<$c, f32, f32, DotProduct>,
            ScalarSparseDataset<$c, f32, f32, DotProduct>
        );
    };
}
impl_ivf_centroids_sparse!(u16);
impl_ivf_centroids_sparse!(u32);

/// [`CentroidIndexBuilder`] that assigns k-means training vectors through an [`IVF`] over the
/// centroids, for e-means: the coarse quantizer is trained on the first build, and every refresh
/// only re-files the moved centroids ([`IVF::refile`]).
///
/// E-means refreshes the centroid index after every mini-batch, tens of times per epoch, so the
/// refresh cost decides: rebuilding an HNSW each time would be expensive. Lloyd refreshes once
/// per epoch and uses an HNSW.
#[derive(Clone, Debug)]
pub struct IvfCentroidIndexBuilder {
    /// Number of lists; `None` uses [`centroid_ivf_n_lists`].
    pub n_lists: Option<usize>,
    /// Seed of the coarse quantizer's sample and k-means.
    pub seed: u64,
}

impl Default for IvfCentroidIndexBuilder {
    fn default() -> Self {
        Self {
            n_lists: None,
            seed: 42,
        }
    }
}

impl<G> CentroidIndexBuilder<G> for IvfCentroidIndexBuilder
where
    G: Dataset + Into<G::Owned>,
    G::Owned: IvfCentroids,
{
    type Index = <G::Owned as IvfCentroids>::Ivf;

    fn build(&mut self, centroids: G) -> Self::Index {
        let k = centroids.len();
        let n_lists = self.n_lists.unwrap_or_else(|| centroid_ivf_n_lists(k));
        <G::Owned as IvfCentroids>::build_ivf(centroids.into(), n_lists, self.seed)
    }

    fn refresh(&mut self, index: Self::Index, centroids: G) -> Self::Index {
        <G::Owned as IvfCentroids>::refile_ivf(index, centroids.into())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use vectorium::{
        PlainDenseDataset, PlainDenseDatasetGrowable, PlainSparseDataset,
        PlainSparseDatasetGrowable, PlainSparseQuantizer, ScalarDenseDataset, ScalarDenseQuantizer,
        ScalarSparseDataset,
    };

    type DensePlain = PlainDenseDataset<f32, SquaredEuclideanDistance>;
    type DensePlainDP = PlainDenseDataset<f32, DotProduct>;
    type SparsePlainDP<C> = PlainSparseDataset<C, f32, DotProduct>;
    /// k-means centroids: dense data clusters under squared Euclidean, sparse under dot product.
    type DenseCentroids = PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>;
    type SparseCentroids = PlainSparseDatasetGrowable<u16, f32, DotProduct>;

    // Sign / formula checks for the reported-metric reconstruction. The cluster
    // dataset always yields a raw dot product `raw = ⟨q,w⟩`.

    #[test]
    fn dot_product_combine_sums_centroid_and_raw() {
        // plain:    ⟨q,v⟩            (centroid 0)
        // residual: ⟨q,c⟩ + ⟨q,r⟩   (centroid = ⟨q,c⟩)
        assert_eq!(DotProduct::combine(0.0, 0.0, 3.0), DotProduct::from(3.0));
        assert_eq!(DotProduct::combine(2.0, 0.0, 3.0), DotProduct::from(5.0));
        const { assert!(!DotProduct::USES_CORRECTION) };
    }

    #[test]
    fn squared_euclidean_combine_applies_correction_and_minus_two() {
        // plain:    ‖v‖² − 2⟨q,v⟩          (centroid 0, correction ‖v‖²)
        //           = 10 − 2·3 = 4
        assert_eq!(
            SquaredEuclideanDistance::combine(0.0, 10.0, 3.0),
            SquaredEuclideanDistance::from(4.0)
        );
        // residual: ‖q−c‖² + (‖r̂‖²+2⟨c,r̂⟩) − 2⟨q,r̂⟩
        //           = 4 + 10 − 2·3 = 8
        assert_eq!(
            SquaredEuclideanDistance::combine(4.0, 10.0, 3.0),
            SquaredEuclideanDistance::from(8.0)
        );
        const { assert!(SquaredEuclideanDistance::USES_CORRECTION) };
    }

    #[test]
    fn ivf_is_a_vectorium_index() {
        use vectorium::core::vector::DenseVectorView;
        use vectorium::{FlatIndex, IndexStats};

        // Two well-separated clusters of 2-D points.
        let dim = 2;
        let enc = ScalarDenseQuantizer::<f32, f32, SquaredEuclideanDistance>::new(dim);
        let mut g = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::new(enc);
        let pts: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
        ];
        for p in &pts {
            g.push(DenseVectorView::new(p));
        }
        let dataset: DensePlain = g.into();

        type Ivf = IVF<DensePlain, DensePlainDP, FlatIndex<DensePlain>>;

        // Exercise both plain and residual storage through the same Index interface.
        for residual in [false, true] {
            let params = IVFBuildParams {
                n_clusters: 2,
                kmeans_solver: Solver::default(),
                kmeans_epochs: Some(10.0),
                kmeans_n_redo: 1,
                kmeans_seed: Some(42),
                kmeans_spherical: false,
                ann_assignment: AnnAssignment::Auto,
                residual,
            };
            let index = Ivf::build_index::<DenseCentroids, DensePlainDP, _>(
                dataset.clone(),
                &params,
                ScalarDenseQuantizer::new(dim),
                (),
                &(),
                FlatIndex::from,
            );

            assert_eq!(IndexStats::n_elements(&index), 6);
            assert_eq!(IndexStats::dim(&index), 2);
            assert_eq!(index.is_residual(), residual);

            let sp = IVFSearchParams {
                nprobe: 2, // probe both clusters => exact
                centroid_search_params: (),
                cluster_query_params: (),
            };
            let query = DenseVectorView::new(&[10.05f32, 10.0]);
            let res = <Ivf as Index>::search(&index, query, 1, &sp);

            assert_eq!(res.len(), 1);
            // Nearest to (10.05, 10.0) is the original vector id 3 == (10.0, 10.0).
            assert_eq!(res[0].vector, 3, "residual={residual}");
        }
    }

    /// The generic `Index` impls cover **all** configs, not just f32/L2: build an
    /// f16-cluster L2 index and a dot-product index entirely through the trait.
    #[test]
    fn ivf_index_covers_f16_and_dot_product() {
        use half::f16;
        use vectorium::FlatIndex;
        use vectorium::core::vector::DenseVectorView;

        let dim = 2;
        let pts: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
        ];
        let mut g = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::new(
            ScalarDenseQuantizer::new(dim),
        );
        for p in &pts {
            g.push(DenseVectorView::new(p));
        }
        let dataset: DensePlain = g.into();

        let params = IVFBuildParams {
            n_clusters: 2,
            kmeans_solver: Solver::default(),
            kmeans_epochs: Some(10.0),
            kmeans_n_redo: 1,
            kmeans_seed: Some(42),
            kmeans_spherical: false,
            ann_assignment: AnnAssignment::Auto,
            residual: false,
        };

        // (a) f16 cluster storage, L2 metric.
        type IvfF16 =
            IVF<DensePlain, ScalarDenseDataset<f32, f16, DotProduct>, FlatIndex<DensePlain>>;
        let idx = IvfF16::build_index::<DenseCentroids, DensePlainDP, _>(
            dataset.clone(),
            &params,
            ScalarDenseQuantizer::new(dim),
            (),
            &(),
            FlatIndex::from,
        );
        let sp = IVFSearchParams {
            nprobe: 2,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        let res = <IvfF16 as Index>::search(&idx, DenseVectorView::new(&[10.05f32, 10.0]), 1, &sp);
        assert_eq!(res[0].vector, 3); // L2 nearest

        // (b) dot-product metric.
        type IvfDp = IVF<DensePlainDP, DensePlainDP, FlatIndex<DensePlainDP>>;
        let ds_dp = DensePlainDP::from_raw(
            dataset.values().to_vec().into_boxed_slice(),
            dataset.len(),
            ScalarDenseQuantizer::new(dim),
        );
        let idx = IvfDp::build_index::<DenseCentroids, DensePlainDP, _>(
            ds_dp,
            &params,
            ScalarDenseQuantizer::new(dim),
            (),
            &(),
            FlatIndex::from,
        );
        let sp = IVFSearchParams {
            nprobe: 2,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        let res = <IvfDp as Index>::search(&idx, DenseVectorView::new(&[10.05f32, 10.0]), 1, &sp);
        // Max ⟨q, v⟩ is id 5 = (9.9, 10.2): 10.05·9.9 + 10·10.2 = 201.5.
        assert_eq!(res[0].vector, 5);
    }

    /// Gaussian blobs in `dim` dimensions (RaBitQ needs `dim` to be a multiple of 64).
    fn random_dense(n: usize, dim: usize, seed: u64) -> DensePlain {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let centers: Vec<Vec<f32>> = (0..8)
            .map(|_| (0..dim).map(|_| rng.gen_range(-5.0f32..5.0)).collect())
            .collect();
        let mut g = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::new(
            ScalarDenseQuantizer::new(dim),
        );
        for i in 0..n {
            let v: Vec<f32> = centers[i % 8]
                .iter()
                .map(|&c| c + rng.gen_range(-1.0f32..1.0))
                .collect();
            g.push(vectorium::DenseVectorView::new(&v));
        }
        g.into()
    }

    fn dense_params(residual: bool) -> IVFBuildParams {
        IVFBuildParams {
            n_clusters: 8,
            kmeans_solver: Solver::default(),
            kmeans_epochs: Some(10.0),
            kmeans_n_redo: 1,
            kmeans_seed: Some(42),
            kmeans_spherical: false,
            ann_assignment: AnnAssignment::Auto,
            residual,
        }
    }

    /// RaBitQ cluster storage under dot product: a full probe returns the top-k of a `FlatIndex`
    /// over RaBitQ codes of the same data (plain storage needs no correction). Scores match up
    /// to rounding only: RaBitQ trains its centering mean on the reordered vectors, which sums
    /// them in a different order.
    #[test]
    fn rabitq_ivf_dot_product_full_probe_matches_flat_index() {
        use vectorium::{FlatIndex, RabitqConfig, RabitqQuantizer, RabitqQueryParams};

        let dim = 128;
        let data: DensePlainDP = random_dense(2000, dim, 1).into();
        let queries = random_dense(20, dim, 2);
        type Cluster = vectorium::DenseDataset<RabitqQuantizer<DotProduct>>;
        type Ivf = IVF<DensePlainDP, Cluster, FlatIndex<DensePlainDP>>;
        let config = RabitqConfig::default();
        let qp = RabitqQueryParams { query_bits: 4 };
        let ivf = Ivf::build_index::<DenseCentroids, DensePlainDP, _>(
            data.clone(),
            &dense_params(false),
            ScalarDenseQuantizer::new(dim),
            config,
            &qp,
            FlatIndex::from,
        );
        let flat = FlatIndex::from(<Cluster as ConvertFrom<_>>::convert_from(&data, config));
        let sp = IVFSearchParams {
            nprobe: 8,
            centroid_search_params: (),
            cluster_query_params: qp,
        };
        for q in queries.iter() {
            let got = Index::search(&ivf, q, 10, &sp);
            let want = flat.search(q, 10, &qp);
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.vector, w.vector);
                let (g, w) = (g.distance.distance(), w.distance.distance());
                assert!((g - w).abs() <= 1e-5 * w.abs(), "{g} vs {w}");
            }
        }
    }

    /// RaBitQ-ext scores an unquantized query, so its estimate is linear in the query and the
    /// decomposition is an identity: plain L2 scores plus the dropped `‖q‖²` equal RaBitQ-ext's
    /// own L2 estimate, up to rounding.
    #[test]
    fn rabitq_ext_ivf_euclidean_equals_native_estimate() {
        use vectorium::{FlatIndex, RabitqExtConfig, RabitqExtQuantizer};

        let dim = 128;
        let data = random_dense(2000, dim, 5);
        let queries = random_dense(20, dim, 6);
        type Cluster = vectorium::DenseDataset<RabitqExtQuantizer<DotProduct>>;
        type Native = vectorium::DenseDataset<RabitqExtQuantizer<SquaredEuclideanDistance>>;
        type Ivf = IVF<DensePlain, Cluster, FlatIndex<DensePlain>>;
        let config = RabitqExtConfig::default();
        let ivf = Ivf::build_index::<DenseCentroids, DensePlainDP, _>(
            data.clone(),
            &dense_params(false),
            ScalarDenseQuantizer::new(dim),
            config,
            &(),
            FlatIndex::from,
        );
        let native = FlatIndex::from(<Native as ConvertFrom<_>>::convert_from(&data, config));
        let sp = IVFSearchParams {
            nprobe: 8,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        for q in queries.iter() {
            let q_norm: f32 = q.values().iter().map(|x| x * x).sum();
            let got = Index::search(&ivf, q, 10, &sp);
            let want = native.search(q, 10, &());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.vector, w.vector);
                let (g, w) = (g.distance.distance() + q_norm, w.distance.distance());
                assert!((g - w).abs() <= 1e-4 * w.abs(), "{g} vs {w}");
            }
        }
    }

    /// The L2 correction comes from the encoder's own L2 estimate, so its code errors cancel
    /// with the dot-product scan's. With an 8-bit RaBitQ-ext storage, a full probe must then
    /// recover nearly the exact L2 top-10, plain and residual.
    #[test]
    fn rabitq_ivf_euclidean_full_probe_recovers_exact_neighbors() {
        use vectorium::{FlatIndex, RabitqExtConfig, RabitqExtQuantizer};

        let dim = 128;
        let data = random_dense(2000, dim, 3);
        let queries = random_dense(20, dim, 4);
        let exact = FlatIndex::from(data.clone());
        type Cluster = vectorium::DenseDataset<RabitqExtQuantizer<DotProduct>>;
        type Ivf = IVF<DensePlain, Cluster, FlatIndex<DensePlain>>;
        let config = RabitqExtConfig {
            total_bits: 8,
            ..RabitqExtConfig::default()
        };
        for residual in [false, true] {
            let ivf = Ivf::build_index::<DenseCentroids, DensePlainDP, _>(
                data.clone(),
                &dense_params(residual),
                ScalarDenseQuantizer::new(dim),
                config,
                &(),
                FlatIndex::from,
            );
            let sp = IVFSearchParams {
                nprobe: 8,
                centroid_search_params: (),
                cluster_query_params: (),
            };
            let mut hits = 0;
            for q in queries.iter() {
                let want: std::collections::HashSet<_> =
                    exact.search(q, 10, &()).iter().map(|r| r.vector).collect();
                hits += Index::search(&ivf, q, 10, &sp)
                    .iter()
                    .filter(|r| want.contains(&r.vector))
                    .count();
            }
            let recall = hits as f32 / (10 * queries.len()) as f32;
            assert!(recall >= 0.9, "residual={residual}: recall@10 {recall}");
        }
    }

    /// Random SPLADE-like sparse data: `n` vectors over `dim` terms, positive values.
    fn random_sparse(n: usize, dim: usize, seed: u64) -> SparsePlainDP<u16> {
        use rand::Rng;
        use vectorium::PlainSparseQuantizer;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut g = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(
            PlainSparseQuantizer::new(dim, dim),
        );
        for _ in 0..n {
            let mut comps: Vec<u16> = (0..30).map(|_| rng.gen_range(0..dim as u16)).collect();
            comps.sort_unstable();
            comps.dedup();
            let vals: Vec<f32> = comps.iter().map(|_| rng.gen_range(0.01f32..3.0)).collect();
            g.push(vectorium::SparseVectorView::new(&comps, &vals));
        }
        g.into()
    }

    fn sparse_params(n_clusters: usize, residual: bool) -> IVFBuildParams {
        IVFBuildParams {
            n_clusters,
            kmeans_solver: Solver::default(),
            kmeans_epochs: Some(10.0),
            kmeans_n_redo: 1,
            kmeans_seed: Some(42),
            kmeans_spherical: false,
            ann_assignment: AnnAssignment::Auto,
            residual,
        }
    }

    /// Same IDs in the same order, scores equal up to rounding: the IVF scans clusters with the
    /// batched kernel, the flat index one vector at a time, and without FMA the two can differ
    /// in the last bits.
    fn assert_same_top_k(got: &[ScoredVector<DotProduct>], want: &[ScoredVector<DotProduct>]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.vector, w.vector);
            let (g, w) = (g.distance.distance(), w.distance.distance());
            assert!((g - w).abs() <= 1e-5 * w.abs(), "{g} vs {w}");
        }
    }

    /// With `nprobe` = number of clusters, sparse IVF scans every vector and must return
    /// the top-k of a `FlatIndex` over the same data: same IDs, same scores up to rounding.
    #[test]
    fn sparse_ivf_full_probe_matches_flat_index() {
        use vectorium::FlatIndex;

        let dim = 1000;
        let data = random_sparse(3000, dim, 1);
        let queries = random_sparse(50, dim, 2);
        let n_clusters = 16;

        type SparseIvf = IVF<SparsePlainDP<u16>, SparsePlainDP<u16>, FlatIndex<SparsePlainDP<u16>>>;
        let ivf = SparseIvf::build_index::<SparseCentroids, SparsePlainDP<u16>, _>(
            data.clone(),
            &sparse_params(n_clusters, false),
            PlainSparseQuantizer::new(dim, dim),
            (),
            &(),
            FlatIndex::from,
        );
        assert_eq!(ivf.n_clusters(), n_clusters);
        assert_eq!(ivf.n_elements(), data.len());
        let flat = FlatIndex::from(data);

        let sp = IVFSearchParams {
            nprobe: n_clusters,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        for q in queries.iter() {
            let got = <SparseIvf as Index>::search(&ivf, q, 10, &sp);
            assert_same_top_k(&got, &flat.search(q, 10, &()));
        }
    }

    /// Same check through an f16 cluster storage, against a `FlatIndex` over f16 data.
    #[test]
    fn sparse_ivf_f16_full_probe_matches_flat_index() {
        use vectorium::FlatIndex;
        use vectorium::core::dataset::ConvertFrom;

        let dim = 500;
        let data = random_sparse(1000, dim, 3);
        let queries = random_sparse(20, dim, 4);
        let n_clusters = 8;

        type Cluster16 = ScalarSparseDataset<u16, f32, half::f16, DotProduct>;
        type SparseIvf16 = IVF<SparsePlainDP<u16>, Cluster16, FlatIndex<SparsePlainDP<u16>>>;
        let ivf = SparseIvf16::build_index::<SparseCentroids, SparsePlainDP<u16>, _>(
            data.clone(),
            &sparse_params(n_clusters, false),
            PlainSparseQuantizer::new(dim, dim),
            (),
            &(),
            FlatIndex::from,
        );
        let flat = FlatIndex::from(<Cluster16 as ConvertFrom<_>>::convert_from(&data, ()));

        let sp = IVFSearchParams {
            nprobe: n_clusters,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        for q in queries.iter() {
            assert_eq!(
                <SparseIvf16 as Index>::search(&ivf, q, 10, &sp),
                flat.search(q, 10, &())
            );
        }
    }

    /// Clustering metric is independent of the reported one: cluster under squared Euclidean,
    /// route and score under dot product; a full probe still equals the flat top-k.
    #[test]
    fn sparse_ivf_euclidean_clustering_full_probe_matches_flat_index() {
        use vectorium::FlatIndex;

        let dim = 500;
        let data = random_sparse(1000, dim, 6);
        let queries = random_sparse(20, dim, 7);
        let n_clusters = 8;

        let kmeans = KMeansBuilder::new().epochs(5.0).seed(Some(42));
        let a = prepare_ivf::<_, PlainSparseDatasetGrowable<u16, f32, SquaredEuclideanDistance>>(
            &data,
            n_clusters,
            kmeans,
            PlainSparseQuantizer::new(dim, dim),
            None,
            false,
            |_, _| {},
        );
        let cluster = a.reorder(&data);
        let ivf: IVF<SparsePlainDP<u16>, SparsePlainDP<u16>, FlatIndex<SparsePlainDP<u16>>> =
            IVF::from_parts(
                FlatIndex::from(SparsePlainDP::<u16>::from(a.centroids)),
                cluster,
                a.cluster_offsets,
                a.external_ids,
                None,
                false,
            );
        let flat = FlatIndex::from(data);
        let sp = IVFSearchParams {
            nprobe: n_clusters,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        for q in queries.iter() {
            assert_same_top_k(&Index::search(&ivf, q, 10, &sp), &flat.search(q, 10, &()));
        }
    }

    /// Pruned centroids change which cluster a vector lands in, never what a full probe returns.
    #[test]
    fn sparse_ivf_pruned_centroids_full_probe_matches_flat_index() {
        use vectorium::FlatIndex;

        let dim = 500;
        let data = random_sparse(1000, dim, 8);
        let queries = random_sparse(20, dim, 9);
        let n_clusters = 8;

        let kmeans = KMeansBuilder::new().epochs(5.0).seed(Some(42));
        let a = prepare_ivf::<_, PlainSparseDatasetGrowable<u16, f32, DotProduct>>(
            &data,
            n_clusters,
            kmeans,
            PlainSparseQuantizer::new(dim, dim),
            None,
            false,
            vectorium::sparse_centroid_pruning(&data, 2.0),
        );
        // Each cluster's average nnz is at most 30 here, so no centroid may keep more than 60.
        assert!(a.centroids.iter().all(|c| c.len() <= 60));
        let cluster = a.reorder(&data);
        let ivf: IVF<SparsePlainDP<u16>, SparsePlainDP<u16>, FlatIndex<SparsePlainDP<u16>>> =
            IVF::from_parts(
                FlatIndex::from(a.centroids),
                cluster,
                a.cluster_offsets,
                a.external_ids,
                None,
                false,
            );
        let flat = FlatIndex::from(data);
        let sp = IVFSearchParams {
            nprobe: n_clusters,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        for q in queries.iter() {
            assert_same_top_k(&Index::search(&ivf, q, 10, &sp), &flat.search(q, 10, &()));
        }
    }

    /// Same check with HNSW-assisted k-means and an HNSW centroid index: probing every cluster
    /// with a beam that covers them all scans every vector, so the result equals the flat top-k.
    #[test]
    fn sparse_ivf_hnsw_centroids_full_probe_matches_flat_index() {
        use vectorium::FlatIndex;

        let dim = 500;
        let data = random_sparse(1000, dim, 10);
        let queries = random_sparse(20, dim, 11);
        let n_clusters = 16;

        let kmeans = KMeansBuilder::new().epochs(5.0).seed(Some(42));
        let a = prepare_ivf::<_, PlainSparseDatasetGrowable<u16, f32, DotProduct>>(
            &data,
            n_clusters,
            kmeans,
            PlainSparseQuantizer::new(dim, dim),
            None,
            true,
            vectorium::sparse_centroid_pruning(&data, 2.0),
        );
        let cluster = a.reorder(&data);
        let centroid_index: HNSW<SparsePlainDP<u16>, Graph> =
            HNSW::build_index(a.centroids, &HNSWBuildConfiguration::default());
        let ivf: IVF<SparsePlainDP<u16>, SparsePlainDP<u16>, HNSW<SparsePlainDP<u16>, Graph>> =
            IVF::from_parts(
                centroid_index,
                cluster,
                a.cluster_offsets,
                a.external_ids,
                None,
                false,
            );
        let flat = FlatIndex::from(data);
        let sp = IVFSearchParams {
            nprobe: n_clusters,
            centroid_search_params: HNSWSearchConfiguration::default().with_ef_search(n_clusters),
            cluster_query_params: (),
        };
        for q in queries.iter() {
            assert_same_top_k(&Index::search(&ivf, q, 10, &sp), &flat.search(q, 10, &()));
        }
    }

    /// Residuals are generic too: a sparse residual IVF (`⟨q,c⟩ + ⟨q,r⟩`) probed in full returns
    /// the flat top-k, scores up to rounding.
    #[test]
    fn sparse_residual_ivf_full_probe_matches_flat_index() {
        use vectorium::FlatIndex;

        let dim = 500;
        let data = random_sparse(1000, dim, 12);
        let queries = random_sparse(20, dim, 13);
        let n_clusters = 8;
        type SparseIvf = IVF<SparsePlainDP<u16>, SparsePlainDP<u16>, FlatIndex<SparsePlainDP<u16>>>;
        let ivf = SparseIvf::build_index::<SparseCentroids, SparsePlainDP<u16>, _>(
            data.clone(),
            &sparse_params(n_clusters, true),
            PlainSparseQuantizer::new(dim, dim),
            (),
            &(),
            FlatIndex::from,
        );
        assert!(ivf.is_residual());
        let flat = FlatIndex::from(data);
        let sp = IVFSearchParams {
            nprobe: n_clusters,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        for q in queries.iter() {
            let got = Index::search(&ivf, q, 10, &sp);
            let want = flat.search(q, 10, &());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.vector, w.vector);
                let (g, w) = (g.distance.distance(), w.distance.distance());
                assert!((g - w).abs() <= 1e-4 * w.abs(), "{g} vs {w}");
            }
        }
    }

    // ---- IVF over centroids (e-means centroid index) ----------------------------------

    #[test]
    fn centroid_ivf_lists_scale_with_sqrt_k() {
        assert_eq!(centroid_ivf_n_lists(32768), 1024);
        assert_eq!(centroid_ivf_n_lists(65536), 1024);
        assert_eq!(centroid_ivf_n_lists(4096), 256);
        assert_eq!(centroid_ivf_n_lists(1), 1);
    }

    #[test]
    fn ivf_over_centroids_files_each_once_and_full_probe_is_exact() {
        let (k, d) = (2000, 16);
        let vectors = random_dense(k, d, 1);
        let ivf = DensePlain::build_ivf(vectors.clone(), 64, 42);
        assert_eq!(ivf.n_elements(), k);
        let mut ids = ivf.external_ids.to_vec();
        ids.sort_unstable();
        assert!(ids.iter().enumerate().all(|(i, &c)| i as u32 == c));

        let flat = FlatIndex::from(vectors);
        let all = DensePlain::ivf_search_params(ivf.n_clusters());
        let queries = random_dense(50, d, 2);
        for q in 0..50 {
            let v = queries.vector(q);
            let view = vectorium::core::vector::DenseVectorView::new(v.values());
            let a = Index::search(&ivf, view, 1, &all)[0];
            let b = flat.search(view, 1, &())[0];
            assert_eq!(a.vector, b.vector);
            // The IVF reports ‖q − v‖² without the per-query constant ‖q‖² (rank-equivalent;
            // k-means recomputes the distance exactly anyway).
            let q2: f32 = v.values().iter().map(|x| x * x).sum();
            assert!((a.distance.distance() + q2 - b.distance.distance()).abs() < 1e-2);
        }
    }

    #[test]
    fn refile_keeps_the_coarse_quantizer_and_finds_moved_centroids() {
        let (k, d) = (1000, 8);
        let ivf = DensePlain::build_ivf(random_dense(k, d, 3), 32, 42);
        let coarse_before = ivf.centroid_index.dataset().values().to_vec();
        let moved = random_dense(k, d, 4);
        let ivf = DensePlain::refile_ivf(ivf, moved.clone());
        assert_eq!(ivf.centroid_index.dataset().values(), &coarse_before[..]);
        let p = DensePlain::ivf_search_params(CENTROID_IVF_N_PROBE);
        for c in [0u32, 17, 999] {
            let v = moved.vector(c);
            let r = Index::search(
                &ivf,
                vectorium::core::vector::DenseVectorView::new(v.values()),
                1,
                &p,
            );
            assert_eq!(r[0].vector, c as VectorId);
        }
    }
}
