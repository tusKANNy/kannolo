use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::graph::graph::Graph;
use crate::graph::neighbors::{NeighborData, Neighbors, PlainNeighbors};
use crate::graph::{GraphTrait, GrowableGraph};
use crate::utils::CompactArray;
use vectorium::IndexSerializer;
use vectorium::core::dataset::{ConvertFrom, ConvertInto, ScoredItemGeneric};
use vectorium::core::index::Index;
use vectorium::distances::Distance;
use vectorium::vector_encoder::VectorEncoder;
use vectorium::{Dataset, QueryEvaluator, SpaceUsage, VectorId};

/// A `HNSW` struct represents a Hierarchical Navigable Small World (HNSW) graph structure that is used
/// for approximate nearest neighbor (ANN) search.
///
/// This index is constructed from a dataset and configuration settings. It efficiently finds the k-closest
/// vectors in the graph for a given query vector.
///
/// # Type Parameters
/// * `D`: The type of the dataset (vectorium dataset).
/// * `G`: The type of the graph implementation (e.g., `Graph`, `GraphFixedDegree`).
#[derive(Serialize, Deserialize)]
pub struct HNSW<D, G> {
    /// A boxed slice containing the hierarchical levels of the HNSW graph.
    /// Each level is a graph structure. Level 0 is the highest level (most sparse),
    /// and the last level is the ground level (contains all nodes).
    levels: Box<[G]>,

    /// Maps local IDs in the first non-ground level (level 1) to the corresponding
    /// global IDs in the ground level (level 0). This is used to find an efficient
    /// entry point for the search on the ground level.
    level1_to_level0_mapping: Box<[usize]>,

    /// Maps each ground-level local ID back to its original external ID: the inverse of the
    /// EGB reordering applied by the `permuted`/`streamvbyte` graph types. `None` when the
    /// index was not built with a permuted dataset.
    original_ids: Option<CompactArray>,

    /// The dataset (dense or sparse) that the graph index is built upon.
    /// This holds the original vectors for distance calculations.
    dataset: D,
    /// The number of neighbors per vector at each level in the HNSW graph.
    /// This is the `M` parameter in the HNSW algorithm.
    num_neighbors_per_vec: usize,
    /// The global ID of the vector from which every search begins.
    /// This node is located on the highest level of the hierarchy.
    entry_point: usize,
}

// Batch scheduling constants for the batched HNSW build.
const BUILD_BATCH_INITIAL: usize = 4; // Starting size; doubles each iteration
const BUILD_BATCH_DIVISOR: usize = 200; // Adaptive cap = N / divisor (keeps B/N ≈ 0.5%)
const BUILD_BATCH_TASKS_PER_THREAD: usize = 5; // Floor = this many tasks per worker thread
const BUILD_BATCH_RATIO_CAP_DIVISOR: usize = 20; // Batch never exceeds N / this (5% of the graph)
const BUILD_PARALLEL_THRESHOLD: usize = 512; // Min level size to use batched parallel processing

/// Largest batch permitted once `inserted_nodes` are in the graph.
///
/// Three rules, in priority order:
/// 1. Aim for `N / BUILD_BATCH_DIVISOR` (0.5% of the graph). Nodes in a batch cannot see
///    each other, so quality damage scales with the batch-to-graph *ratio*.
/// 2. Never go below a floor of `BUILD_BATCH_TASKS_PER_THREAD` tasks per worker thread,
///    or short batches leave threads idle waiting on the slowest task.
/// 3. Never exceed `N / BUILD_BATCH_RATIO_CAP_DIVISOR`.
///
#[inline]
fn effective_batch_max(inserted_nodes: usize) -> usize {
    let floor = BUILD_BATCH_TASKS_PER_THREAD * rayon::current_num_threads();
    let ratio_cap = (inserted_nodes / BUILD_BATCH_RATIO_CAP_DIVISOR).max(BUILD_BATCH_INITIAL);
    (inserted_nodes / BUILD_BATCH_DIVISOR)
        .max(floor)
        .min(ratio_cap)
}

/// Configuration for building the HNSW index.
/// Use the builder pattern: `HNSWBuildConfiguration::default().with_num_neighbors(32).with_ef_construction(200)`
pub struct HNSWBuildConfiguration {
    /// The number of neighbors for each node on each layer of the graph.
    /// Also known as `M` in the HNSW paper.
    pub num_neighbors_per_vec: usize,
    /// The size of the dynamic candidate list for constructing the graph.
    /// Also known as `efConstruction` in the HNSW paper.
    pub ef_construction: usize,
}

impl HNSWBuildConfiguration {
    /// Sets the number of neighbors per vector (M parameter). Returns self for chaining.
    #[must_use]
    pub fn with_num_neighbors(mut self, num_neighbors_per_vec: usize) -> Self {
        self.num_neighbors_per_vec = num_neighbors_per_vec;
        self
    }

    /// Sets the ef_construction parameter. Returns self for chaining.
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }
}

impl Default for HNSWBuildConfiguration {
    fn default() -> Self {
        Self {
            num_neighbors_per_vec: 16,
            ef_construction: 150,
        }
    }
}

/// Strategy for early termination during HNSW search.
#[derive(Debug, Clone, Copy, Default)]
pub enum EarlyTerminationStrategy {
    /// Standard HNSW: stop when the best frontier candidate is worse
    /// than the worst candidate in the top-k result set.
    #[default]
    None,
    /// Distance-adaptive: allow exploration within a relaxed threshold
    /// controlled by `lambda` on the worst top candidate.
    ///
    /// Reference: "Distance Adaptive Beam Search for Provably Accurate
    /// Graph-Based Nearest Neighbor Search" (Al-Jazzazi et al.)
    DistanceAdaptive {
        /// Relaxation parameter. `lambda = 0` is equivalent to `None`.
        lambda: f32,
    },
}

impl EarlyTerminationStrategy {
    /// Returns the relaxation parameter (`0.0` for `None`).
    #[inline]
    pub fn lambda(&self) -> f32 {
        match self {
            EarlyTerminationStrategy::None => 0.0,
            EarlyTerminationStrategy::DistanceAdaptive { lambda } => *lambda,
        }
    }
}

/// Configuration for searching the HNSW index.
/// Use the builder pattern: `HNSWSearchConfiguration::default().with_ef_search(200)`
///
/// `P` is the query-side parameter type of the dataset encoder
/// ([`VectorEncoder::QueryParams`]). It defaults to `()`, so `HNSWSearchConfiguration`
/// written without a parameter still means "an encoder that takes no query configuration"
/// (plain, scalar, PQ, DotPacking8). Encoders that do take query configuration — RaBitQ's
/// `query_bits`, for instance — pin it to their own type, and the graph search forwards
/// [`query_params`](Self::query_params) verbatim when it builds the evaluator.
pub struct HNSWSearchConfiguration<P = ()> {
    /// The size of the dynamic candidate list for searching the graph.
    /// Also known as `ef` or `efSearch` in the HNSW paper. A larger
    /// value leads to more accurate results at the cost of speed.
    pub ef_search: usize,
    /// Early termination strategy for search.
    pub early_termination: EarlyTerminationStrategy,
    /// Query-side encoder parameters, handed straight to
    /// [`VectorEncoder::query_evaluator`]. `()` when the encoder takes none.
    ///
    /// These describe the *query*, not the stored data, so one index serves every setting
    /// concurrently — nothing here mutates the dataset.
    pub query_params: P,
}

impl<P> HNSWSearchConfiguration<P> {
    /// Sets the ef_search parameter. Returns self for chaining.
    #[must_use]
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Sets the early termination strategy. Returns self for chaining.
    #[must_use]
    pub fn with_early_termination(mut self, strategy: EarlyTerminationStrategy) -> Self {
        self.early_termination = strategy;
        self
    }

    /// Sets the encoder's query-side parameters. Returns self for chaining.
    #[must_use]
    pub fn with_query_params(mut self, query_params: P) -> Self {
        self.query_params = query_params;
        self
    }
}

impl<P: Default> Default for HNSWSearchConfiguration<P> {
    /// Provides a default `ef_search` value.
    fn default() -> Self {
        Self {
            ef_search: 100,
            early_termination: EarlyTerminationStrategy::None,
            query_params: P::default(),
        }
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset,
    G: GraphTrait,
{
    /// Return the maximum level of the HNSW graph (0-based).
    #[must_use]
    #[inline]
    pub fn max_level(&self) -> usize {
        if self.levels.is_empty() {
            0
        } else {
            self.levels.len() - 1
        }
    }

    /// Returns a vec with the number of nodes at each level, from highest to lowest (ground).
    #[must_use]
    pub fn nodes_per_level(&self) -> Vec<usize> {
        self.levels.iter().map(|g| g.n_nodes()).collect()
    }

    /// Maps a ground-level local ID to its original external ID.
    /// When the index was built with a co-permuted dataset, applies the stored inverse
    /// permutation; otherwise returns the local ID unchanged.
    #[must_use]
    #[inline]
    pub fn get_original_id(&self, local_id: usize) -> usize {
        self.original_ids
            .as_ref()
            .map_or(local_id, |inv| inv.get(local_id))
    }

    /// Returns the ground level graph (densest layer, contains every node).
    #[must_use]
    #[inline]
    pub fn get_ground_level(&self) -> &G {
        &self.levels[self.levels.len() - 1]
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset,
    G: GraphTrait,
{
    /// Converts an `HNSW` index from a different dataset type, preserving the graph structure.
    ///
    /// Only the dataset is replaced; levels, entry point, level mappings, and neighbor counts
    /// are moved unchanged. The caller must ensure that the new dataset `D` has the same
    /// number of vectors and the same logical vector order as `T`.
    ///
    /// `config` is the target dataset's own construction configuration
    /// ([`ConvertFrom::Config`]) — `()` for encoders that learn everything from the source
    /// data, `RabitqConfig` for RaBitQ, and so on. There is exactly one entry point, so a
    /// caller never has to know which kind of encoder it is holding.
    pub fn convert_dataset_from<T: Dataset>(
        hnsw: HNSW<T, G>,
        config: <D as ConvertFrom<T>>::Config,
    ) -> Self
    where
        D: Dataset + ConvertFrom<T>,
    {
        let HNSW {
            levels,
            level1_to_level0_mapping,
            original_ids,
            dataset,
            num_neighbors_per_vec,
            entry_point,
        } = hnsw;

        Self {
            levels,
            level1_to_level0_mapping,
            original_ids,
            dataset: ConvertInto::<D>::convert_into(dataset, config),
            num_neighbors_per_vec,
            entry_point,
        }
    }

    /// Converts this `HNSW` into one backed by a different dataset type (consuming self).
    ///
    /// This is the mirror of [`convert_dataset_from`]. Prefer this when you own the index
    /// and want to chain from a plain build:
    ///
    /// ```
    /// # use kannolo::graph::Graph;
    /// # use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
    /// # use vectorium::distances::DotProduct;
    /// # use vectorium::vector::SparseVectorView;
    /// # use vectorium::{
    /// #     DatasetGrowable, DotPacking8U16FixedU8Dataset, PlainSparseDataset,
    /// #     PlainSparseDatasetGrowable, PlainSparseQuantizer,
    /// # };
    /// # let mut growable = PlainSparseDatasetGrowable::new(
    /// #     PlainSparseQuantizer::<u16, f32, DotProduct>::new(8, 8),
    /// # );
    /// # for i in 0u16..16 {
    /// #     let mut components = [i % 8, (i + 3) % 8];
    /// #     components.sort_unstable();
    /// #     growable.push(SparseVectorView::new(&components, &[0.5f32, 0.25]));
    /// # }
    /// # let dataset: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
    /// # let config = HNSWBuildConfiguration::default()
    /// #     .with_num_neighbors(4)
    /// #     .with_ef_construction(8);
    /// let plain: HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph> =
    ///     HNSW::build_index(dataset, &config);
    /// // DotPacking8 learns everything from the source, so its `ConvertFrom::Config` is `()`.
    /// let compressed: HNSW<DotPacking8U16FixedU8Dataset, Graph> =
    ///     plain.convert_dataset_into(());
    /// ```
    pub fn convert_dataset_into<T>(self, config: <T as ConvertFrom<D>>::Config) -> HNSW<T, G>
    where
        T: Dataset + ConvertFrom<D>,
    {
        HNSW::<T, G>::convert_dataset_from(self, config)
    }

    /// Converts this `HNSW` into one backed by a different dataset type using a borrowed source dataset.
    ///
    /// Use this when the target dataset implements `ConvertFrom<&D>` instead of `ConvertFrom<D>`
    /// — the quantizers that train on the source before encoding it (PQ, RaBitQ) all do, since
    /// training and encoding are two passes over the same plain vectors.
    ///
    /// `Cfg` is named as its own parameter rather than written
    /// `<T as ConvertFrom<&'a D>>::Config` because that projection would depend on the
    /// higher-ranked `'a`, which cannot be named in the signature.
    pub fn convert_dataset_into_ref<T, Cfg>(self, config: Cfg) -> HNSW<T, G>
    where
        T: Dataset,
        for<'a> T: ConvertFrom<&'a D, Config = Cfg>,
    {
        let HNSW {
            levels,
            level1_to_level0_mapping,
            original_ids,
            dataset,
            num_neighbors_per_vec,
            entry_point,
        } = self;

        HNSW {
            levels,
            level1_to_level0_mapping,
            original_ids,
            dataset: T::convert_from(&dataset, config),
            num_neighbors_per_vec,
            entry_point,
        }
    }
}

impl<D, Nsrc> HNSW<D, Graph<Nsrc>>
where
    D: Dataset,
    Nsrc: Neighbors,
{
    /// Used by the `permuted`/`streamvbyte` graph types: computes an Enhanced Graph Bisection
    /// permutation (`old_id -> new_id`) from the ground level, reorders every level's node IDs
    /// accordingly, and rebuilds each level's adjacency lists into a fresh `Ndst` backend. The
    /// dataset is co-permuted through [`Dataset::permute`] so node IDs and dataset row indices
    /// stay in agreement.
    ///
    /// Only the ground level is reordered. Upper levels keep their local order and are merely
    /// relabelled and re-encoded (see [`Graph::permute_level`]), so the nested
    /// prefix relationship between levels — which the descent in [`HNSW::search`] relies on to
    /// carry a local ID from one level to the next — holds without any extra work.
    ///
    /// Ground-level local IDs then become permuted-dataset row indices, so the returned index
    /// stores the inverse permutation and translates results back to the original external IDs
    /// (see [`HNSW::get_original_id`]).
    pub fn permute_and_encode<Ndst>(&self) -> HNSW<D::Owned, Graph<Ndst>>
    where
        Nsrc: Sync,
        Ndst: Neighbors + From<NeighborData>,
    {
        let last = self.levels.len() - 1;
        let permutation = crate::graph::egb::compute_permutation(&self.levels[last]);
        let permutation = permutation.as_slice();

        let mut new_levels = Vec::with_capacity(self.levels.len());

        let mut level1_to_level0_mapping = Box::<[usize]>::from([]);
        let mut entry_point = self.entry_point;
        let mut ground_inv: Vec<usize> = Vec::new();

        for (i, level) in self.levels.iter().enumerate() {
            let (new_level, inv): (Graph<Ndst>, Option<Vec<usize>>) =
                level.permute_level::<Ndst>(permutation);

            if i == last {
                ground_inv = inv.expect(
                    "the ground level carries no id mapping, so it must have been reordered",
                );
            } else if i + 1 == last {
                level1_to_level0_mapping = (0..new_level.n_nodes())
                    .map(|id| new_level.get_external_id(id))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
            }

            new_levels.push(new_level);
        }

        // Upper levels keep their local order, so a multi-level entry point does not move. With a
        // single level the entry point is a ground local ID, which the permutation does move.
        if self.levels.len() == 1 {
            entry_point = permutation[self.entry_point];
        }

        HNSW {
            levels: new_levels.into_boxed_slice(),
            level1_to_level0_mapping,
            original_ids: Some(CompactArray::from(ground_inv)),
            dataset: self.dataset.permute(permutation),
            num_neighbors_per_vec: self.num_neighbors_per_vec,
            entry_point,
        }
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync,
    <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    G: GraphTrait,
{
    /// Performs ACORN-1 filtered approximate nearest-neighbor search.
    ///
    /// Returns the `k` approximate nearest neighbors of `query` that satisfy
    /// `predicate(vector_id) == true`. Unlike a simple post-filter, the predicate
    /// is applied *during* graph traversal: non-matching nodes are skipped and their
    /// neighbors are inspected via a two-hop expansion to maintain connectivity in
    /// the filtered sub-graph.
    ///
    /// The HNSW index does **not** need to be rebuilt; this method works on any
    /// standard HNSW index (ACORN-1 variant).
    ///
    /// # Arguments
    /// * `query` – The query vector.
    /// * `k` – Number of nearest neighbors to return.
    /// * `search_params` – Search configuration (`ef_search`, early termination).
    /// * `predicate` – `Fn(vector_id: usize) -> bool`. Called with the global
    ///   (dataset-level) vector ID; only vectors for which this returns `true`
    ///   will appear in results.
    pub fn search_filtered<'q, F>(
        &'q self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        search_params: &HNSWSearchConfiguration<<D::Encoder as VectorEncoder>::QueryParams>,
        predicate: F,
    ) -> Vec<vectorium::dataset::ScoredVector<<D::Encoder as VectorEncoder>::Distance>>
    where
        F: Fn(usize) -> bool,
    {
        let query_eval = self
            .dataset
            .encoder()
            .query_evaluator(query, &search_params.query_params);
        let num_levels = self.levels.len();

        // --- Stage 1: upper levels (unfiltered greedy search, same as standard HNSW) ---
        let entry_graph = if num_levels > 1 {
            &self.levels[0]
        } else {
            &self.levels[num_levels - 1]
        };
        let entry_external_id = entry_graph.get_external_id(self.entry_point) as VectorId;
        let entry_distance = query_eval.compute_distance(self.dataset.get(entry_external_id));
        let mut entry_node = ScoredItemGeneric {
            distance: entry_distance,
            vector: self.entry_point,
        };
        if num_levels > 1 {
            for level_graph in &self.levels[..num_levels - 1] {
                entry_node =
                    level_graph.greedy_search_nearest(&self.dataset, &query_eval, entry_node);
            }
        }

        // --- Stage 2: ground level (ACORN-1 filtered search) ---
        let ground_graph = &self.levels[num_levels - 1];
        let entry_global_id = if num_levels > 1 {
            self.level1_to_level0_mapping[entry_node.vector]
        } else {
            self.entry_point
        };
        let ground_entry_node = ScoredItemGeneric {
            distance: entry_node.distance,
            vector: entry_global_id,
        };

        let ef = search_params.ef_search.max(k);
        let lambda = search_params.early_termination.lambda();
        let mapped_predicate = |local_id: usize| predicate(self.get_original_id(local_id));
        let top_heap = ground_graph.acorn_search_candidates_filtered(
            &self.dataset,
            ground_entry_node,
            &query_eval,
            ef,
            k,
            lambda,
            &mapped_predicate,
        );

        let mut topk = top_heap.into_sorted_vec();
        topk.truncate(k);
        topk.drain(..)
            .map(|candidate| vectorium::dataset::ScoredVector {
                distance: candidate.distance,
                vector: self.get_original_id(candidate.vector) as VectorId,
            })
            .collect()
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync,
    <D::Encoder as VectorEncoder>::Distance: Distance,
    G: GraphTrait,
{
    /// Build pre-expanded neighbor lists for ACORN-γ filtered search.
    ///
    /// For each ground-level node `v`, the two-hop neighborhood (direct neighbors
    /// and their neighbors) is scored by distance to `v` and pruned to
    /// `max(gamma * M, 2M)` entries, sorted closest-first.
    ///
    /// Call this **once** after the standard HNSW build, then pass the result to
    /// [`Self::search_filtered_gamma`] for fast predicate-aware search.
    ///
    /// The expanded adjacency is an ordinary [`Graph`]: it is a ground level, so local IDs are
    /// external IDs and no ID mapping is stored, and `max_degree` is reported as the pruned-to
    /// size above. The lists are ordered by distance rather than by ID, so the backend has to
    /// store them verbatim — [`PlainNeighbors`] does, a delta-coded one would not.
    ///
    /// Three departures from ACORN-γ as published, all following from this being a pass over a
    /// finished index rather than a modified insertion algorithm:
    ///
    /// 1. **Candidate pool.** ACORN-γ collects `gamma * M` candidates during insertion, from a
    ///    metadata-agnostic search of its own graph. We take the two-hop neighborhood of a graph
    ///    whose lists were already cut by HNSW's robust-prune heuristic, so the pool is whatever
    ///    that heuristic left behind.
    /// 2. **HNSW's pruning is still in effect.** ACORN replaces the RNG-approximation prune with
    ///    its own rule precisely because a metadata-blind prune is wrong under a predicate: it
    ///    drops `v–b` whenever some `a` is closer to `b` than `v` is, and if `a` fails the query
    ///    predicate the path `v→a→b` that justified the drop does not exist. Both of our ACORN
    ///    variants sit on top of a graph pruned that way.
    /// 3. **No `M_beta` compression.** ACORN-γ keeps the nearest `M_beta` candidates verbatim and
    ///    prunes the rest with a rule that leaves every dropped edge recoverable by a partial
    ///    expansion at search time. We keep the nearest and drop the rest outright, so the
    ///    expanded list is *not* a superset of the ground level: the far neighbors the robust
    ///    prune deliberately kept for navigability are the first a distance sort discards.
    ///
    /// # Arguments
    /// * `gamma` – Expansion factor. Each node stores up to `gamma * M` neighbors, floored at
    ///   the ground level's own `2M` bound so that a small `gamma` cannot shrink the graph it
    ///   expands. Larger values improve recall at the cost of memory and build time; ACORN
    ///   chooses it as the reciprocal of the smallest selectivity to be served.
    pub fn build_acorn_gamma_neighbors(&self, gamma: usize) -> Graph<PlainNeighbors> {
        let n = self.dataset.len();
        let m = self.num_neighbors_per_vec;
        let ground_graph = &self.levels[self.levels.len() - 1];
        // Floored at the ground level's own degree bound (2M): `gamma * M` alone would make
        // gamma = 1 halve the budget of the graph being expanded, and gamma = 2 merely match it.
        // A no-op for gamma >= 2, which is the regime ACORN's gamma is chosen in.
        let gamma_m = (gamma * m).max(ground_graph.max_degree());

        let mut expanded: Vec<u32> = Vec::with_capacity(n * gamma_m);
        let mut offsets: Vec<usize> = Vec::with_capacity(n + 1);
        offsets.push(0);
        // Two scratches: the outer list stays alive while the inner one is produced.
        let mut outer_scratch: Vec<u32> = Vec::new();
        let mut inner_scratch: Vec<u32> = Vec::new();

        for v in 0..n {
            // Collect the two-hop neighborhood, excluding v itself.
            let mut seen: HashSet<usize> = HashSet::new();
            seen.insert(v);
            let mut candidates: Vec<usize> = Vec::new();

            outer_scratch.clear();
            outer_scratch.extend_from_slice(ground_graph.neighbors(v, &mut inner_scratch));
            for &neighbor in outer_scratch.iter() {
                let u = neighbor as usize;
                if seen.insert(u) {
                    candidates.push(u);
                }
                for &w in ground_graph.neighbors(u, &mut inner_scratch) {
                    if seen.insert(w as usize) {
                        candidates.push(w as usize);
                    }
                }
            }

            // Score each candidate by distance to v.
            let v_vec = self.dataset.get(v as VectorId);
            let eval = self.dataset.encoder().vector_evaluator(v_vec);

            let mut scored: Vec<(<D::Encoder as VectorEncoder>::Distance, usize)> = candidates
                .into_iter()
                .map(|u| {
                    let d = eval.compute_distance(self.dataset.get(u as VectorId));
                    (d, u)
                })
                .collect();

            // Sort ascending (closest first), truncate to gamma * M.
            scored.sort_unstable_by_key(|a| a.0);
            scored.truncate(gamma_m);

            expanded.extend(scored.into_iter().map(|(_, u)| u as u32));
            offsets.push(expanded.len());
        }

        Graph::from_neighbor_data(
            NeighborData {
                data: expanded.into_boxed_slice(),
                offsets: offsets.into_boxed_slice(),
            },
            gamma_m,
            None,
        )
    }

    /// ACORN-γ filtered approximate nearest-neighbor search.
    ///
    /// Unlike ACORN-1 ([`Self::search_filtered`]), the two-hop expansion is
    /// **pre-computed** at index construction time (via
    /// [`Self::build_acorn_gamma_neighbors`]). At search time, standard beam search
    /// runs over the pre-expanded neighbor lists; predicate-failing nodes are
    /// simply skipped — no on-the-fly two-hop expansion is needed.
    ///
    /// # Arguments
    /// * `query` – The query vector.
    /// * `k` – Number of nearest neighbors to return.
    /// * `search_params` – Search configuration (`ef_search`, early termination).
    /// * `acorn_gamma` – Pre-built expanded neighbor lists (from `build_acorn_gamma_neighbors`).
    /// * `predicate` – `Fn(vector_id: usize) -> bool`. Only vectors satisfying
    ///   this will appear in the results.
    pub fn search_filtered_gamma<'q, N2, F>(
        &'q self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        search_params: &HNSWSearchConfiguration<<D::Encoder as VectorEncoder>::QueryParams>,
        acorn_gamma: &Graph<N2>,
        predicate: F,
    ) -> Vec<vectorium::dataset::ScoredVector<<D::Encoder as VectorEncoder>::Distance>>
    where
        N2: Neighbors,
        F: Fn(usize) -> bool,
    {
        let query_eval = self
            .dataset
            .encoder()
            .query_evaluator(query, &search_params.query_params);
        let num_levels = self.levels.len();

        // --- Stage 1: upper levels (unfiltered greedy search, same as ACORN-1) ---
        let entry_graph = if num_levels > 1 {
            &self.levels[0]
        } else {
            &self.levels[num_levels - 1]
        };
        let entry_external_id = entry_graph.get_external_id(self.entry_point) as VectorId;
        let entry_distance = query_eval.compute_distance(self.dataset.get(entry_external_id));
        let mut entry_node = ScoredItemGeneric {
            distance: entry_distance,
            vector: self.entry_point,
        };
        if num_levels > 1 {
            for level_graph in &self.levels[..num_levels - 1] {
                entry_node =
                    level_graph.greedy_search_nearest(&self.dataset, &query_eval, entry_node);
            }
        }

        // --- Stage 2: ground level (ACORN-γ search on pre-expanded neighbor lists) ---
        let entry_global_id = if num_levels > 1 {
            self.level1_to_level0_mapping[entry_node.vector]
        } else {
            self.entry_point
        };
        let ground_entry_node = ScoredItemGeneric {
            distance: entry_node.distance,
            vector: entry_global_id,
        };

        let ef = search_params.ef_search.max(k);
        let lambda = search_params.early_termination.lambda();
        let mapped_predicate = |local_id: usize| predicate(self.get_original_id(local_id));
        let top_heap = acorn_gamma.acorn_gamma_search_filtered(
            &self.dataset,
            ground_entry_node,
            &query_eval,
            ef,
            k,
            lambda,
            &mapped_predicate,
        );

        let mut topk = top_heap.into_sorted_vec();
        topk.truncate(k);
        topk.drain(..)
            .map(|candidate| vectorium::dataset::ScoredVector {
                distance: candidate.distance,
                vector: self.get_original_id(candidate.vector) as VectorId,
            })
            .collect()
    }
}

impl<D, G> vectorium::IndexStats for HNSW<D, G>
where
    D: Dataset + Sync + SpaceUsage,
    G: GraphTrait,
{
    #[inline]
    fn n_elements(&self) -> usize {
        self.dataset.len()
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dataset.input_dim()
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync + SpaceUsage,
    G: GraphTrait,
{
    pub fn print_space_usage_bytes(&self) {
        let dataset_size = self.dataset.space_usage_bytes();
        let inv_perm_size = self
            .original_ids
            .as_ref()
            .map_or(0, CompactArray::space_usage_bytes);
        let index_size = self
            .levels
            .iter()
            .map(|g| g.get_space_usage_bytes())
            .sum::<usize>()
            + inv_perm_size;

        let total_size = dataset_size + index_size;
        println!(
            "[######] Space usage: Dataset: {dataset_size} bytes, Index: {index_size} bytes, Total: {total_size} bytes"
        );

        let total_edges: usize = self.levels.iter().map(|g| g.n_edges()).sum();
        println!(
            "[######] {:.2} bits/edge ({total_edges} edges)",
            (index_size * 8) as f64 / total_edges as f64
        );
    }
}

impl<D, G> Index for HNSW<D, G>
where
    D: Dataset + Sync + SpaceUsage,
    <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    G: GraphTrait,
{
    type Query<'q> = <D::Encoder as VectorEncoder>::QueryVector<'q>;
    type Distance = <D::Encoder as VectorEncoder>::Distance;
    /// Graph-side knobs plus the encoder's own query parameters, forwarded to
    /// [`VectorEncoder::query_evaluator`].
    type SearchParams = HNSWSearchConfiguration<<D::Encoder as VectorEncoder>::QueryParams>;

    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<vectorium::dataset::ScoredVector<Self::Distance>> {
        let query_eval = self
            .dataset
            .encoder()
            .query_evaluator(query, &search_params.query_params);
        let num_levels = self.levels.len();

        // --- Stage 1: Search upper levels ---
        // Start at the single entry point on the highest level.
        let entry_graph = if num_levels > 1 {
            &self.levels[0]
        } else {
            &self.levels[num_levels - 1]
        };
        let entry_external_id = entry_graph.get_external_id(self.entry_point) as VectorId;
        let entry_distance = query_eval.compute_distance(self.dataset.get(entry_external_id));
        let mut entry_node = ScoredItemGeneric {
            distance: entry_distance,
            vector: self.entry_point,
        };
        if num_levels > 1 {
            // Greedily search from the top level down to level 1.
            for level_graph in &self.levels[..num_levels - 1] {
                entry_node =
                    level_graph.greedy_search_nearest(&self.dataset, &query_eval, entry_node);
            }
        }

        // --- Stage 2: Search ground level ---
        // The ground level contains all the vectors.
        let ground_graph = &self.levels[num_levels - 1];
        let entry_global_id = if num_levels > 1 {
            // The entry_node now holds the local ID from the last searched upper level (level 1).
            // We need to map this to a global ID for the ground level to start the final search.
            self.level1_to_level0_mapping[entry_node.vector]
        } else {
            // No upper levels, the entry point is a ground-level ID.
            self.entry_point
        };

        // The distance from the previous level's search is a good starting point.
        let ground_entry_node = ScoredItemGeneric {
            distance: entry_node.distance,
            vector: entry_global_id,
        };

        // Perform the final, most extensive search on the ground level.
        // Ensure that `ef_search` is at least `k` to guarantee we can return `k` results.
        let ef = search_params.ef_search.max(k);
        let lambda = search_params.early_termination.lambda();
        let mut topk = ground_graph.greedy_search_topk(
            &self.dataset,
            ground_entry_node,
            &query_eval,
            k,
            ef,
            lambda,
        );

        // Map local IDs to global vector IDs and return scored vectors
        topk.drain(..)
            .map(|candidate| vectorium::dataset::ScoredVector {
                distance: candidate.distance,
                vector: self.get_original_id(candidate.vector) as VectorId,
            })
            .collect()
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync + SpaceUsage,
    <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    G: GraphTrait + From<GrowableGraph>,
{
    /// Builds the HNSW index from a source dataset.
    ///
    /// This function orchestrates the entire build process:
    /// 1. It computes the random level assignments for each vector.
    /// 2. It initializes the graph structures for each level.
    /// 3. It inserts the single entry point node.
    /// 4. It iterates through all HNSW levels, from highest to lowest, inserting nodes.
    ///    - A hybrid sequential/parallel strategy is used based on the number of nodes at each level.
    /// 5. It finalizes the graph structures and creates the final `HNSW` index struct.
    pub fn build_index(dataset: D, build_params: &HNSWBuildConfiguration) -> Self {
        let num_vectors = dataset.len();
        let m = build_params.num_neighbors_per_vec;
        let default_probabs =
            compute_levels_probabilities(1.0 / (m as f32).ln(), num_vectors as f32);

        // // 1. Get level assignments and sorted IDs.
        let (levels_mapping, ids_sorted_by_level, cumulative_ids_per_level, max_level) =
            compute_levels(&default_probabs, num_vectors);

        // 2. Setup graphs and mappings.
        let mut growable_levels: Vec<GrowableGraph> = Vec::with_capacity(max_level as usize + 1);

        // Initialize upper levels (from highest to lowest)
        for i in (1..=max_level).rev() {
            let mut graph = GrowableGraph::with_max_degree(m);
            let num_nodes_in_level = levels_mapping[i as usize - 1].len();
            graph.reserve(num_nodes_in_level);
            graph
                .set_mapping(levels_mapping[i as usize - 1].clone())
                .expect("Graph mapping size validation should have passed");
            growable_levels.push(graph);
        }

        // Initialize ground level
        let mut ground_graph = GrowableGraph::with_max_degree(2 * m);
        ground_graph.reserve(num_vectors);
        growable_levels.push(ground_graph);

        let level1_to_level0_mapping = if max_level > 0 {
            levels_mapping[0].clone()
        } else {
            Vec::new()
        };
        let entry_point_local_id = 0;

        // 3. Build all levels by iterating through nodes level by level.
        let entry_point_global_id = ids_sorted_by_level[0];

        // --- START: Progress Bar Setup ---
        let pb = ProgressBar::new(num_vectors as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - Building HNSW")
                .unwrap()
                .progress_chars("#>-"),
        );
        // --- END: Progress Bar Setup ---

        // Insert the entry point (the first node in the sorted list)
        Self::insert_entry_point(&mut growable_levels, entry_point_global_id, max_level, &pb);

        // Main build loop: iterate through HNSW levels from highest to lowest
        for level in (0..=max_level).rev() {
            let start_index = cumulative_ids_per_level[max_level as usize - level as usize];
            let start_index = if start_index == 0 { 1 } else { start_index };
            let end_index = cumulative_ids_per_level[max_level as usize - level as usize + 1];
            if start_index >= end_index {
                continue;
            }

            let nodes_to_insert_slice = &ids_sorted_by_level[start_index..end_index];

            if nodes_to_insert_slice.len() > BUILD_PARALLEL_THRESHOLD {
                Self::process_level_parallelly(
                    nodes_to_insert_slice,
                    level,
                    max_level,
                    m,
                    &mut growable_levels,
                    &dataset,
                    build_params,
                    entry_point_local_id,
                    &level1_to_level0_mapping,
                    &ids_sorted_by_level,
                    &pb,
                );
            } else {
                Self::process_level_sequentially(
                    nodes_to_insert_slice,
                    level,
                    max_level,
                    m,
                    &mut growable_levels,
                    &dataset,
                    build_params,
                    entry_point_local_id,
                    &level1_to_level0_mapping,
                    &ids_sorted_by_level,
                    &pb,
                );
            }
        }

        pb.finish_with_message("HNSW build complete.");

        // 4. Finalize and create the HNSW struct.
        let final_levels: Vec<G> = growable_levels.into_iter().map(Into::into).collect();

        Self {
            levels: final_levels.into_boxed_slice(),
            level1_to_level0_mapping: level1_to_level0_mapping.into_boxed_slice(),
            original_ids: None,
            dataset,
            num_neighbors_per_vec: m,
            entry_point: entry_point_local_id,
        }
    }
}

impl<D, G> IndexSerializer for HNSW<D, G> {}

impl<D, G> HNSW<D, G>
where
    D: Dataset + SpaceUsage,
    G: GraphTrait,
{
    /// Total space used by the dataset and all graph levels, in bytes.
    pub fn space_usage_bytes(&self) -> usize {
        let dataset_size = self.dataset.space_usage_bytes();
        let inv_perm_size = self
            .original_ids
            .as_ref()
            .map_or(0, CompactArray::space_usage_bytes);
        let graph_size: usize = self.levels.iter().map(|g| g.get_space_usage_bytes()).sum();
        dataset_size + graph_size + inv_perm_size
    }
}

/// Computes the probabilities for a node to be assigned to each level in the HNSW graph.
///
/// # Parameters
///
/// - `level_mult`: A multiplier that affects the exponential decay of probabilities for each level.
///
/// # Returns
///
/// - A vector of probabilities for each level, where each probability is computed based on the formula:
///   `probability = exp(-level / level_mult) * (1 - exp(- 1 / level_mult))`.
///
///   The probabilities decrease exponentially with increasing level, controlled by `level_mult`.
///
/// The function continues to compute these values for increasing levels until the calculated
/// probability for a level falls below a small threshold.
///
/// # Example
///
/// After calling this function with a `level_mult` of `1.0`, the probabilities decrease exponentially,
/// e.g., starting around [0.6321, 0.3679, 0.1353, ...].
///
/// ```text
/// // Example (illustrative values):
/// // probabs_levels ≈ [0.6321, 0.3679, 0.1353, ...]
/// ```
#[must_use]
fn compute_levels_probabilities(level_mult: f32, dataset_len: f32) -> Vec<f32> {
    let mut probabs_levels = Vec::new();

    for level in 0.. {
        let proba = (-level as f32 / level_mult).exp() * (1.0 - (-1.0 / level_mult).exp());

        // Prune levels with expected number of assigned nodes below 1
        if proba < 1.0 / dataset_len {
            break;
        }
        probabs_levels.push(proba);
    }

    probabs_levels
}

/// This function generates a random level for a node in the HNSW graph.
///
/// # Description
///
/// The function begins by generating a random floating-point number `f` between 0.0 and 1.0.
/// The function then iterates over the `probabs_levels` vector, comparing `f` with the probability thresholds for
/// each level. If `f` is less than the current level's probability, that level is selected and returned as a `u8`.
/// If `f` is larger, the function reduces `f` by the threshold value and continues to the next level. If no level
/// is selected, the maximum level, which corresponds to the last index of `probabs_levels`, is returned.
///
/// # Parameters
///
/// - `probabs_levels`: A vector whose i-th entry represents the probability of selecting level `i` of the HNSW graph.
/// - `rng`: A mutable reference to a random number generator of type `StdRng`.
///
/// # Returns
///
/// - `u8`: The level selected for the node, ranging from 0 to the maximum level.
///
/// /// # Example
///
/// Assume `probabs_levels` contains `[0.6, 0.3, 0.1]` and the random value `f` is `0.65`.
/// After checking level 0 (0.6),`f` is decreased by 0.6 to become `0.05`. The function would then
/// return level 1, as `0.05` is less than the probability for level 1 (0.3).
#[must_use]
#[inline]
fn random_level(probabs_levels: &[f32], rng: &mut StdRng) -> u8 {
    let mut f: f32 = rng.gen_range(0.0..1.0);
    for (level, &prob) in probabs_levels.iter().enumerate() {
        if f < prob {
            return level as u8;
        }
        f -= prob;
    }
    // it returns the maximum level which is the size of the vector probabs_levels
    (probabs_levels.len() - 1) as u8
}

/// Assigns levels to each vector in the graph and updates the internal `offsets` and `neighbors` vectors.
///
/// # Arguments
///
/// - `default_probabs`: A vector of probabilities for each level, which is used to determine the level assignment for each vector.
/// - `num_vectors`: The number of vectors to which levels will be assigned.
///
/// # Description
///
/// This function assigns a level to each vector in the graph and computes the levels matrix which contains the IDs of vectors at each level.
/// It uses a random number generator to select a level based on the provided probabilities. Each vector is assigned to all levels up to and including its assigned level.
/// The function also keeps track of the maximum level assigned to any vector, that could be lower than the length of `default_probabs` in case no vector was assigned to a level.
/// Finally, it ensures that the levels vector does not contain any empty vectors, removing them if necessary.
///
/// # Returns
/// /// - A tuple containing:
///  - A vector of vectors, where each inner vector contains the IDs of vectors assigned to that level.
///  - The maximum level assigned to any vector.
///
#[must_use]
#[inline]
fn compute_levels(
    default_probabs: &[f32],
    num_vectors: usize,
) -> (Vec<Vec<usize>>, Vec<usize>, Vec<usize>, u8) {
    let mut rng = StdRng::seed_from_u64(523);

    // 1. Create a shuffled list of all node IDs. This is the single source of randomness.
    let mut all_ids: Vec<usize> = (0..num_vectors).collect();
    all_ids.shuffle(&mut rng);

    // 2. Assign a highest level to each node.
    // `ids_per_level[i]` will store nodes whose highest assigned level is `i`.
    let mut ids_per_level: Vec<Vec<usize>> = vec![Vec::new(); default_probabs.len() + 1];
    for &id in &all_ids {
        let level = random_level(default_probabs, &mut rng);
        ids_per_level[level as usize].push(id);
    }

    // 3. Find the actual maximum level that has any nodes assigned to it.
    let max_level = ids_per_level
        .iter()
        .rposition(|level_nodes| !level_nodes.is_empty())
        .unwrap_or(0) as u8;

    // 4. Create the final, sorted build order.
    // Candidates are ordered by level (highest to lowest). Because we populated `ids_per_level`
    // from a shuffled list, the nodes within each level block are already randomized.
    let mut ids_sorted_by_level: Vec<usize> = Vec::with_capacity(num_vectors);
    for i in (0..=max_level).rev() {
        ids_sorted_by_level.extend(&ids_per_level[i as usize]);
    }

    // 5. `cumulative_ids_per_level` tracks the number of nodes *at or above* a given HNSW level.
    // It's used to slice `ids_sorted_by_level` during the build loop.
    let mut cumulative_ids_per_level = Vec::with_capacity(max_level as usize + 2);
    cumulative_ids_per_level.push(0);
    let mut count = 0;
    for i in (0..=max_level).rev() {
        count += ids_per_level[i as usize].len();
        cumulative_ids_per_level.push(count);
    }

    // 6. `levels_mapping[i]` contains all global IDs present at HNSW level `i+1`.
    // A node at level L is also present at all levels < L. The mapping for each level
    // is now a consistent prefix of the final `ids_sorted_by_level` list.
    let mut levels_mapping: Vec<Vec<usize>> = Vec::with_capacity(max_level as usize);
    for i in 0..max_level as usize {
        // HNSW level `i+1` corresponds to `levels_mapping[i]`.
        // The nodes for this level are all nodes from the highest level down to level `i+1`.
        let num_nodes_at_this_level_or_above = cumulative_ids_per_level[max_level as usize - i];
        let mapping_for_this_level: Vec<usize> =
            ids_sorted_by_level[0..num_nodes_at_this_level_or_above].to_vec();
        levels_mapping.push(mapping_for_this_level);
    }

    (
        levels_mapping,
        ids_sorted_by_level,
        cumulative_ids_per_level,
        max_level,
    )
}

// --- Private Helper Methods for HNSW build process ---
impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync,
    <D::Encoder as VectorEncoder>::Distance: Ord + Copy,
    G: GraphTrait,
{
    fn insert_entry_point(
        growable_levels: &mut [GrowableGraph],
        entry_point_global_id: usize,
        max_level: u8,
        pb: &ProgressBar,
    ) {
        for (i, graph) in growable_levels.iter_mut().enumerate() {
            if i < max_level as usize {
                // Is an upper level
                graph.push_with_precomputed_reverse_links(Some(entry_point_global_id), &[], 0, &[]);
            } else {
                // Is the ground level
                graph.push_with_precomputed_reverse_links(None, &[], entry_point_global_id, &[]);
            }
        }
        pb.inc(1); // Increment for the entry point

        // After inserting the entry point, we must advance the counter on all upper levels.
        for graph in growable_levels.iter_mut().take(max_level as usize) {
            graph.advance_inserted_nodes(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_level_sequentially(
        nodes_to_insert_slice: &[usize],
        level: u8,
        max_level: u8,
        m: usize,
        growable_levels: &mut [GrowableGraph],
        source_dataset: &D,
        build_params: &HNSWBuildConfiguration,
        entry_point_local_id: usize,
        level1_to_level0_mapping: &[usize],
        ids_sorted_by_level: &[usize],
        pb: &ProgressBar,
    ) where
        <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    {
        let entry_point_global_id = ids_sorted_by_level[0];
        for &global_id in nodes_to_insert_slice {
            let query_eval = source_dataset
                .encoder()
                .vector_evaluator(source_dataset.get(global_id as VectorId));
            let entry_distance =
                query_eval.compute_distance(source_dataset.get(entry_point_global_id as VectorId));
            let mut entry_node = ScoredItemGeneric {
                distance: entry_distance,
                vector: entry_point_local_id,
            };

            if level > 0 {
                for current_level in ((level + 1)..=max_level).rev() {
                    let graph_idx = max_level as usize - current_level as usize;
                    entry_node = growable_levels[graph_idx].greedy_search_nearest(
                        source_dataset,
                        &query_eval,
                        entry_node,
                    );
                }
                for current_level in (1..=level).rev() {
                    let graph_idx = max_level as usize - current_level as usize;
                    let graph = &mut growable_levels[graph_idx];
                    let local_id = graph.inserted_nodes();

                    let (forward, reverse, new_entry) = graph.find_and_prune_neighbors(
                        source_dataset,
                        &query_eval,
                        entry_node,
                        build_params.ef_construction,
                        m,
                        local_id,
                    );

                    graph.push_with_precomputed_reverse_links(
                        Some(global_id),
                        &forward,
                        local_id,
                        &reverse,
                    );
                    graph.advance_inserted_nodes(1);
                    entry_node = new_entry;
                }
            }

            let ground_graph = &mut growable_levels[max_level as usize];
            let ground_entry_global_id = if max_level > 0 {
                level1_to_level0_mapping[entry_node.vector]
            } else {
                ids_sorted_by_level[0]
            };
            let dist =
                query_eval.compute_distance(source_dataset.get(ground_entry_global_id as VectorId));
            let ground_entry_node = ScoredItemGeneric {
                distance: dist,
                vector: ground_entry_global_id,
            };

            let (ground_neighbors, ground_reverse_links, _) = ground_graph
                .find_and_prune_neighbors(
                    source_dataset,
                    &query_eval,
                    ground_entry_node,
                    build_params.ef_construction,
                    2 * m,
                    global_id,
                );

            ground_graph.push_with_precomputed_reverse_links(
                None,
                &ground_neighbors,
                global_id,
                &ground_reverse_links,
            );
            pb.inc(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_level_parallelly(
        nodes_to_insert_slice: &[usize],
        level: u8,
        max_level: u8,
        m: usize,
        growable_levels: &mut [GrowableGraph],
        source_dataset: &D,
        build_params: &HNSWBuildConfiguration,
        entry_point_local_id: usize,
        level1_to_level0_mapping: &[usize],
        ids_sorted_by_level: &[usize],
        pb: &ProgressBar,
    ) where
        <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
        D: Sync,
    {
        // (global_id, forward links per upper level, ground forward links).
        // The upper-level entry `j` is HNSW level `level - j` (levels are visited
        // top-down), matching the order they are pushed in below.
        type InsertionEntry = (usize, Vec<Vec<usize>>, Vec<usize>);

        let level_start_local_ids: Vec<usize> =
            growable_levels.iter().map(|g| g.inserted_nodes()).collect();
        let total_nodes = nodes_to_insert_slice.len();
        let entry_point_global_id = ids_sorted_by_level[0];
        // ── Phase 1 ─────────────────────────────────────────────────────────────────
        // Parallel graph search against the frozen graph, returning pruned forward links.
        // Reverse links are not computed here: they are grouped by target and merged in
        // Phase 2, so that a target hit by several nodes of the batch is pruned once with
        // all of its incoming edges present.
        let run_phase1 = |batch: &[usize], shared: &[GrowableGraph]| -> Vec<InsertionEntry> {
            batch
                .par_iter()
                .map(|&global_id| {
                    let query_eval = source_dataset
                        .encoder()
                        .vector_evaluator(source_dataset.get(global_id as VectorId));
                    let entry_distance = query_eval
                        .compute_distance(source_dataset.get(entry_point_global_id as VectorId));
                    let mut entry_node = ScoredItemGeneric {
                        distance: entry_distance,
                        vector: entry_point_local_id,
                    };
                    let mut upper_level_data = Vec::new();

                    if level > 0 {
                        for current_level in ((level + 1)..=max_level).rev() {
                            let graph_idx = max_level as usize - current_level as usize;
                            entry_node = shared[graph_idx].greedy_search_nearest(
                                source_dataset,
                                &query_eval,
                                entry_node,
                            );
                        }
                        for current_level in (1..=level).rev() {
                            let graph_idx = max_level as usize - current_level as usize;
                            let (forward, new_entry) = shared[graph_idx].search_and_prune_forward(
                                source_dataset,
                                &query_eval,
                                entry_node,
                                build_params.ef_construction,
                                m,
                            );
                            upper_level_data.push(forward);
                            entry_node = new_entry;
                        }
                    }

                    let ground_entry_global_id = if max_level > 0 {
                        level1_to_level0_mapping[entry_node.vector]
                    } else {
                        ids_sorted_by_level[0]
                    };
                    let dist = query_eval
                        .compute_distance(source_dataset.get(ground_entry_global_id as VectorId));
                    let ground_entry_node = ScoredItemGeneric {
                        distance: dist,
                        vector: ground_entry_global_id,
                    };
                    let (ground_neighbors, _) = shared[max_level as usize]
                        .search_and_prune_forward(
                            source_dataset,
                            &query_eval,
                            ground_entry_node,
                            build_params.ef_construction,
                            2 * m,
                        );

                    (global_id, upper_level_data, ground_neighbors)
                })
                .collect()
        };

        {
            // The local id of batch entry `i` in the level graph `graph_idx`, and the
            // forward list it computed there. Ground-level local ids are global ids.
            fn entry_in_graph<'a>(
                entry: &'a InsertionEntry,
                i: usize,
                batch_start: usize,
                graph_idx: usize,
                max_level: usize,
                level: usize,
                level_start_local_ids: &[usize],
            ) -> (usize, &'a [usize]) {
                let hnsw_level = max_level - graph_idx;
                if hnsw_level == 0 {
                    (entry.0, &entry.2)
                } else {
                    (
                        level_start_local_ids[graph_idx] + batch_start + i,
                        &entry.1[level - hnsw_level],
                    )
                }
            }

            // ── Phase 2: grouped reverse-edge merge, then forward links ─────────────
            //
            // 2a. Collect this batch's `(target, new_node)` reverse pairs and sort them,
            //     which groups each target's incoming edges into one contiguous run.
            // 2b. Merge in parallel, one prune per distinct target with all of its
            //     incoming edges present. Targets are distinct across tasks and are
            //     always already-inserted nodes, so no task can observe another's work.
            // 2c. Commit: the merges are pure reads, so all writes happen afterwards.
            let apply_phase2 = |data: &[InsertionEntry],
                                batch_start: usize,
                                levels: &mut [GrowableGraph]| {
                let mut pairs: Vec<(u32, u32)> = Vec::new();
                let mut groups: Vec<(usize, usize)> = Vec::new();

                // The index is needed for `level_start_local_ids` and to derive the HNSW
                // level, and the merge borrows the graph immutably then mutably, so an
                // iterator over `levels` does not fit.
                #[allow(clippy::needless_range_loop)]
                for graph_idx in (max_level as usize - level as usize)..=max_level as usize {
                    // 2a — group.
                    pairs.clear();
                    for (i, entry) in data.iter().enumerate() {
                        let (local_id, forward) = entry_in_graph(
                            entry,
                            i,
                            batch_start,
                            graph_idx,
                            max_level as usize,
                            level as usize,
                            &level_start_local_ids,
                        );
                        pairs.extend(forward.iter().map(|&v| (v as u32, local_id as u32)));
                    }
                    pairs.par_sort_unstable();

                    groups.clear();
                    let mut start = 0;
                    while start < pairs.len() {
                        let mut end = start + 1;
                        while end < pairs.len() && pairs[end].0 == pairs[start].0 {
                            end += 1;
                        }
                        groups.push((start, end - start));
                        start = end;
                    }

                    // 2b — merge, in parallel over distinct targets.
                    let graph = &levels[graph_idx];
                    let merges: Vec<_> = groups
                        .par_iter()
                        .map(|&(start, len)| {
                            let run = &pairs[start..start + len];
                            graph.merge_reverse_links(source_dataset, run[0].0 as usize, run)
                        })
                        .collect();

                    // 2c — commit the merges.
                    let graph = &mut levels[graph_idx];
                    for (&(start, len), (old_degree, replacement)) in groups.iter().zip(merges) {
                        let run = &pairs[start..start + len];
                        let target = run[0].0 as usize;
                        match replacement {
                            None => {
                                graph.append_links(target, old_degree, run.iter().map(|&(_, u)| u))
                            }
                            Some(list) => graph.replace_links(target, &list, old_degree),
                        }
                    }
                }

                // Forward links: each new node writes its own slots, disjoint by
                // construction from every reverse target (which is always an
                // already-inserted node).
                for (i, entry) in data.iter().enumerate() {
                    #[allow(clippy::needless_range_loop)] // see the merge loop above
                    for graph_idx in (max_level as usize - level as usize)..=max_level as usize {
                        let (local_id, forward) = entry_in_graph(
                            entry,
                            i,
                            batch_start,
                            graph_idx,
                            max_level as usize,
                            level as usize,
                            &level_start_local_ids,
                        );
                        levels[graph_idx].write_links(forward, local_id);
                    }
                }
            };

            let mut current_batch_size = BUILD_BATCH_INITIAL;
            let mut batch_start = 0usize;

            while batch_start < total_nodes {
                let batch_size = current_batch_size.min(total_nodes - batch_start);
                let batch = &nodes_to_insert_slice[batch_start..batch_start + batch_size];

                // Phase 1: parallel search against the frozen graph.
                let data = run_phase1(batch, growable_levels);

                // ids_mapping needs &mut, so it is written before the &self link writes.
                for (i, (global_id, _, _)) in data.iter().enumerate() {
                    for level_idx in 0..level as usize {
                        let hnsw_level = level_idx + 1;
                        let graph_idx = max_level as usize - hnsw_level;
                        let local_id = level_start_local_ids[graph_idx] + batch_start + i;
                        growable_levels[graph_idx].write_id_mapping(local_id, Some(*global_id));
                    }
                    growable_levels[max_level as usize].write_id_mapping(*global_id, None);
                }

                // Forward-edge counts.
                let upper_edge_counts: Vec<usize> = (0..level as usize)
                    .map(|level_idx| {
                        let upper_data_idx = level as usize - 1 - level_idx;
                        data.iter()
                            .map(|(_, upper, _)| upper[upper_data_idx].len())
                            .sum()
                    })
                    .collect();
                let ground_edge_count: usize = data.iter().map(|(_, _, f)| f.len()).sum();

                // Phase 2: grouped reverse merge + forward commit.
                apply_phase2(&data, batch_start, growable_levels);

                Self::finish_batch(
                    growable_levels,
                    level,
                    max_level,
                    batch_size,
                    upper_edge_counts,
                    ground_edge_count,
                );
                pb.inc(batch_size as u64);

                batch_start += batch_size;
                let effective_max = effective_batch_max(batch_start);
                if current_batch_size < effective_max {
                    current_batch_size = (current_batch_size * 2).min(effective_max);
                }
            }
        }
    }

    /// Update n_edges and advance inserted-node counters after a batch completes.
    fn finish_batch(
        growable_levels: &mut [GrowableGraph],
        level: u8,
        max_level: u8,
        batch_size: usize,
        upper_edge_counts: Vec<usize>,
        ground_edge_count: usize,
    ) {
        for (level_idx, count) in upper_edge_counts.into_iter().enumerate() {
            let hnsw_level = level_idx + 1;
            let graph_idx = max_level as usize - hnsw_level;
            growable_levels[graph_idx].add_n_edges(count);
        }
        growable_levels[max_level as usize].add_n_edges(ground_edge_count);
        for current_level in (1..=level).rev() {
            let graph_idx = max_level as usize - current_level as usize;
            growable_levels[graph_idx].advance_inserted_nodes(batch_size);
        }
    }
}

#[cfg(test)]
mod convert_dataset_tests {
    use super::*;
    use crate::graph::Graph;
    use vectorium::{
        BridgedDotPacking8U32ScalarU8Dataset, ComponentType, DatasetGrowable,
        DotPacking8U16FixedU8Dataset, DotProduct, FixedU8Q, FixedU16Q, IndexStats,
        PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer, ScalarSparseDataset,
        ScalarU8SparseDataset, SparseVectorView,
    };

    fn build_test_hnsw() -> HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph> {
        build_test_hnsw_with::<u16>()
    }

    fn build_test_hnsw_with<C>() -> HNSW<PlainSparseDataset<C, f32, DotProduct>, Graph>
    where
        C: ComponentType + num_traits::FromPrimitive + SpaceUsage,
    {
        let encoder = PlainSparseQuantizer::<C, f32, DotProduct>::new(20, 20);
        let mut growable: PlainSparseDatasetGrowable<C, f32, DotProduct> =
            PlainSparseDatasetGrowable::new(encoder);

        for i in 0u16..30 {
            let components: Vec<u16> = (0..5).map(|j: u16| (i * 3 + j) % 20).collect();
            let mut components = components;
            components.sort();
            components.dedup();
            let values: Vec<f32> = components.iter().map(|&c| (c as f32 + 1.0) * 0.1).collect();
            let components: Vec<C> = components
                .iter()
                .map(|&c| C::from_u16(c).unwrap())
                .collect();
            growable.push(SparseVectorView::new(&components, &values));
        }

        let dataset: PlainSparseDataset<C, f32, DotProduct> = growable.into();

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(4)
            .with_ef_construction(20);

        HNSW::build_index(dataset, &config)
    }

    #[test]
    fn test_convert_dataset_into_dotpacking8_fixedu8() {
        let plain_hnsw = build_test_hnsw();
        let n = plain_hnsw.n_elements();

        let hnsw: HNSW<DotPacking8U16FixedU8Dataset, Graph> = plain_hnsw.convert_dataset_into(());

        assert_eq!(hnsw.n_elements(), n);
    }

    #[test]
    fn test_convert_dataset_into_fixedu8() {
        let plain_hnsw = build_test_hnsw();
        let n = plain_hnsw.n_elements();

        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> =
            plain_hnsw.convert_dataset_into(());

        assert_eq!(hnsw.n_elements(), n);
    }

    #[test]
    fn test_convert_dataset_into_fixedu16() {
        let plain_hnsw = build_test_hnsw();
        let n = plain_hnsw.n_elements();

        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> =
            plain_hnsw.convert_dataset_into(());

        assert_eq!(hnsw.n_elements(), n);
    }

    #[test]
    fn test_dotpacking8_fixedu8_search_returns_results() {
        let hnsw: HNSW<DotPacking8U16FixedU8Dataset, Graph> =
            build_test_hnsw().convert_dataset_into(());

        let query_components: Vec<u16> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_fixedu8_search_returns_results() {
        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> =
            build_test_hnsw().convert_dataset_into(());

        let query_components: Vec<u16> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_fixedu16_search_returns_results() {
        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> =
            build_test_hnsw().convert_dataset_into(());

        let query_components: Vec<u16> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_scalaru8_search_returns_results() {
        let hnsw: HNSW<ScalarU8SparseDataset<u16, DotProduct>, Graph> =
            build_test_hnsw().convert_dataset_into(());

        let query_components: Vec<u16> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_bridged_dotpacking8_u32_search_returns_results() {
        let hnsw: HNSW<BridgedDotPacking8U32ScalarU8Dataset, Graph> =
            build_test_hnsw_with::<u32>().convert_dataset_into(());

        let query_components: Vec<u32> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }
}

#[cfg(test)]
mod permute_compress_tests {
    use super::*;
    use crate::graph::egb;
    use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
    use vectorium::DenseDataset;
    use vectorium::core::vector::DenseVectorView;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;

    type TestDataset = DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>;
    type PlainHnsw = HNSW<TestDataset, Graph<PlainNeighbors>>;

    fn build_test_hnsw(n: usize, dim: usize) -> PlainHnsw {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
        // Deterministic pseudo-random values, no external RNG dependency needed.
        let flat: Vec<f32> = (0..n * dim)
            .map(|i| (((i * 2654435761u64 as usize) % 1000) as f32) / 1000.0)
            .collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(40);

        HNSW::build_index(dataset, &config)
    }

    /// Same dataset and parameters as [`build_test_hnsw`], built over the fixed-stride backend.
    fn build_fixed_test_hnsw<N>(n: usize, dim: usize) -> HNSW<TestDataset, Graph<N>>
    where
        N: Neighbors + From<NeighborData>,
    {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
        let flat: Vec<f32> = (0..n * dim)
            .map(|i| (((i * 2654435761u64 as usize) % 1000) as f32) / 1000.0)
            .collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(40);
        HNSW::build_index(dataset, &config)
    }

    /// Runs a fixed set of queries and returns their `(distance, external_id)` pairs, so
    /// different graph representations can be compared directly.
    fn run_queries<G>(
        hnsw: &HNSW<DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>, G>,
        dim: usize,
        k: usize,
    ) -> Vec<Vec<(SquaredEuclideanDistance, usize)>>
    where
        G: GraphTrait,
    {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        (0..10)
            .map(|q| {
                let query_val: Vec<f32> = (0..dim)
                    .map(|j| (((q * 97 + j * 13) % 1000) as f32) / 1000.0)
                    .collect();
                let query = DenseVectorView::new(&query_val);
                // Sort by (distance, id): heap order is not stable among exact ties, but the
                // set of results must match regardless of node ordering or compression.
                let mut results: Vec<(SquaredEuclideanDistance, usize)> = hnsw
                    .search(query, k, &search_config)
                    .into_iter()
                    .map(|sv| (sv.distance, sv.vector as usize))
                    .collect();
                results.sort_by(|a, b| a.partial_cmp(b).unwrap());
                results
            })
            .collect()
    }

    /// Asserts two result sets are equivalent *as answers*: identical distance at every rank,
    /// with the identity at that rank free to differ.
    ///
    /// Comparing `(distance, id)` pairs instead would assert something reordering does not
    /// promise. When two database vectors are exactly equidistant from the query, which one
    /// enters the top-k is decided by the order the search reaches them, and that order follows
    /// the node numbering — which reordering exists to change. The answer is equally good either
    /// way, so the distance sequence is the invariant and the identities are not.
    fn assert_results_equivalent(
        tag: &str,
        baseline: &[Vec<(SquaredEuclideanDistance, usize)>],
        other: &[Vec<(SquaredEuclideanDistance, usize)>],
    ) {
        assert_eq!(baseline.len(), other.len(), "{tag}: query count differs");
        for (q, (b, o)) in baseline.iter().zip(other).enumerate() {
            assert_eq!(b.len(), o.len(), "{tag}: query {q} returned a different k");
            let (b_dists, o_dists): (Vec<f32>, Vec<f32>) = (
                b.iter().map(|(d, _)| d.distance()).collect(),
                o.iter().map(|(d, _)| d.distance()).collect(),
            );
            assert_eq!(
                b_dists, o_dists,
                "{tag}: query {q} returned different distances"
            );
        }
    }

    /// The `permuted` graph type must return equivalent answers to the baseline: reordering
    /// changes the internal node order, never the quality of the result.
    #[test]
    fn permuted_graph_matches_baseline_search_results() {
        let n = 200;
        let dim = 8;
        let plain = build_test_hnsw(n, dim);
        let baseline = run_queries(&plain, dim, 5);

        let permuted: HNSW<TestDataset, Graph<PlainNeighbors>> =
            plain.permute_and_encode::<PlainNeighbors>();

        assert_results_equivalent("permuted", &baseline, &run_queries(&permuted, dim, 5));

        // Sanity: the permutation actually reordered nodes, so the check above is not vacuous.
        // Recomputed here because `permute_and_encode` runs EGB internally and does not return it.
        let perm = egb::compute_permutation(plain.get_ground_level());
        let identity_count = perm.iter().enumerate().filter(|&(i, &p)| i == p).count();
        assert!(
            identity_count < n,
            "EGB permutation should reorder at least some nodes"
        );
    }

    /// The original-ID mapping must stay bit-packed: one `usize` per node silently
    /// inflates the index and the reported bits/edge (~1.4 bits/edge on SIFT1M).
    #[test]
    fn original_ids_are_bit_packed() {
        let n = 4000;
        let dim = 8;
        let plain = build_test_hnsw(n, dim);

        let permuted: HNSW<
            DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>,
            Graph<PlainNeighbors>,
        > = plain.permute_and_encode::<PlainNeighbors>();

        let inv = permuted
            .original_ids
            .as_ref()
            .expect("a permuted index must carry an inverse permutation");

        // Read through the packed representation, the mapping is still a permutation.
        let mut seen: Vec<usize> = (0..n)
            .map(|local| permuted.get_original_id(local))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>());

        // 4000 IDs need 12 bits each, so the packed form must be far below 8 bytes/entry.
        let unpacked = n * std::mem::size_of::<usize>();
        assert!(
            inv.space_usage_bytes() < unpacked / 3,
            "inverse permutation not packed: {} bytes vs {} unpacked",
            inv.space_usage_bytes(),
            unpacked
        );
    }

    /// Every backend must survive a save/load round trip with its search output intact. This
    /// covers the serialized layout of each `Neighbors` implementation, including the
    /// fixed-stride one, whose on-disk representation changed when it became a backend.
    #[test]
    fn every_backend_survives_serialization_round_trip() {
        use crate::graph::FixedDegreeNeighbors;
        use vectorium::IndexSerializer;

        let n = 200;
        let dim = 8;
        let plain = build_test_hnsw(n, dim);
        let baseline = run_queries(&plain, dim, 5);

        let dir = std::env::temp_dir();

        // Each backend is compared against *itself* before the round trip, so this test
        // isolates serialization: reordering-induced tie-break differences, which the
        // `permute_and_encode` path can produce, are not in scope here.
        let p = dir
            .join("kannolo_rt_plain.idx")
            .to_str()
            .unwrap()
            .to_string();
        plain.save_index(&p).unwrap();
        let loaded: PlainHnsw = PlainHnsw::load_index(&p).unwrap();
        assert_eq!(
            baseline,
            run_queries(&loaded, dim, 5),
            "plain: round trip changed search results"
        );
        std::fs::remove_file(&p).ok();

        let p = dir
            .join("kannolo_rt_fixed.idx")
            .to_str()
            .unwrap()
            .to_string();
        let fixed: HNSW<TestDataset, Graph<FixedDegreeNeighbors>> = build_fixed_test_hnsw(n, dim);
        let fixed_before = run_queries(&fixed, dim, 5);
        assert_eq!(
            baseline, fixed_before,
            "fixed backend must reproduce the plain build's results"
        );
        fixed.save_index(&p).unwrap();
        let loaded: HNSW<TestDataset, Graph<FixedDegreeNeighbors>> =
            IndexSerializer::load_index(&p).unwrap();
        assert_eq!(
            fixed_before,
            run_queries(&loaded, dim, 5),
            "fixed: round trip changed search results"
        );
        std::fs::remove_file(&p).ok();

        let p = dir.join("kannolo_rt_svb.idx").to_str().unwrap().to_string();
        let svb: HNSW<TestDataset, Graph<StreamVByteNeighbors>> =
            plain.permute_and_encode::<StreamVByteNeighbors>();
        let svb_before = run_queries(&svb, dim, 5);
        svb.save_index(&p).unwrap();
        let loaded: HNSW<TestDataset, Graph<StreamVByteNeighbors>> =
            IndexSerializer::load_index(&p).unwrap();
        assert_eq!(
            svb_before,
            run_queries(&loaded, dim, 5),
            "streamvbyte: round trip changed search results"
        );
        std::fs::remove_file(&p).ok();
    }

    /// A delta-coded backend is reachable straight from a build, without the reordering pass:
    /// the generic `From<GrowableGraph>` sorts the lists because `REQUIRES_SORTED` asks it to.
    /// The resulting adjacency must be the plain one, sorted and deduplicated.
    #[test]
    fn build_index_into_streamvbyte_backend_matches_sorted_plain_lists() {
        let n = 200;
        let dim = 8;
        let plain = build_test_hnsw(n, dim);

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
        let flat: Vec<f32> = (0..n * dim)
            .map(|i| (((i * 2654435761u64 as usize) % 1000) as f32) / 1000.0)
            .collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(40);
        let compressed: HNSW<
            DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>,
            Graph<StreamVByteNeighbors>,
        > = HNSW::build_index(dataset, &config);

        let plain_ground = plain.get_ground_level();
        let svb_ground = compressed.get_ground_level();
        assert_eq!(plain_ground.n_nodes(), svb_ground.n_nodes());

        let mut a = Vec::new();
        let mut b = Vec::new();
        for v in 0..plain_ground.n_nodes() {
            let mut expected: Vec<u32> = plain_ground.neighbors(v, &mut a).to_vec();
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(
                expected,
                svb_ground.neighbors(v, &mut b).to_vec(),
                "node {v}: compressed backend must hold the sorted, deduplicated plain list"
            );
        }
    }

    /// Same as above for the `streamvbyte` graph type: compression must be lossless
    /// with respect to the search output.
    #[test]
    fn streamvbyte_graph_matches_baseline_search_results() {
        let n = 200;
        let dim = 8;
        let plain = build_test_hnsw(n, dim);
        let baseline = run_queries(&plain, dim, 5);

        let compressed: HNSW<TestDataset, Graph<StreamVByteNeighbors>> =
            plain.permute_and_encode::<StreamVByteNeighbors>();

        assert_results_equivalent("streamvbyte", &baseline, &run_queries(&compressed, dim, 5));

        // Compression must be lossless on the adjacency itself, which is the property exact
        // result equality was really reaching for: every list decodes back to the permuted
        // plain list, so any difference above can only come from scoring, not from the graph.
        let permuted: HNSW<TestDataset, Graph<PlainNeighbors>> =
            plain.permute_and_encode::<PlainNeighbors>();
        let (uncompressed_ground, compressed_ground) =
            (permuted.get_ground_level(), compressed.get_ground_level());
        assert_eq!(uncompressed_ground.n_nodes(), compressed_ground.n_nodes());
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for v in 0..uncompressed_ground.n_nodes() {
            assert_eq!(
                uncompressed_ground.neighbors(v, &mut a),
                compressed_ground.neighbors(v, &mut b),
                "node {v}: StreamVByte decoding must reproduce the uncompressed list"
            );
        }
        assert!(
            compressed_ground.get_space_usage_bytes() < uncompressed_ground.get_space_usage_bytes(),
            "the compressed backend should use less space than the plain one"
        );
    }

    /// A permuted index must carry the *same* encoded dataset as the unpermuted one, only in a
    /// different row order. Encoders that are fitted to the data must therefore be trained
    /// before the permutation, never after — which is why `build_permuted_and_save` in
    /// `src/bin/hnsw_build.rs` runs `convert` before `permute_and_encode`.
    ///
    /// The DotPacking8 conversion trains its component order and value steps on every vector, so
    /// this test locks the contract rather than a known failure: it was introduced for the
    /// removed DotVByte encoder, which trained on a prefix sample, and whose `standard` and
    /// `permuted` indexes differed by 47.8 MB on the real 8.8M-document collection.
    #[test]
    fn permuting_preserves_the_encoded_dotpacking8_dataset() {
        use vectorium::{
            DatasetGrowable, DotPacking8U16FixedU8Dataset, DotProduct, PlainSparseDataset,
            PlainSparseDatasetGrowable, PlainSparseQuantizer, SparseVectorView,
        };

        let n = 200;
        let dim = 64;

        let quantizer = PlainSparseQuantizer::<u16, f32, DotProduct>::new(dim, dim);
        let mut growable = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(quantizer);
        for i in 0..n {
            // Deterministic and strictly increasing: the j*5 offsets stay distinct modulo 64.
            let mut components: Vec<u16> = (0..8).map(|j| ((i * 7 + j * 5) % dim) as u16).collect();
            components.sort_unstable();
            let values: Vec<f32> = (0..components.len())
                .map(|j| (((i + j) % 97) as f32) / 97.0 + 0.01)
                .collect();
            growable.push(SparseVectorView::new(&components, &values));
        }
        let dataset: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(40);
        let plain: HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph<PlainNeighbors>> =
            HNSW::build_index(dataset, &config);

        // Same order as `build_permuted_and_save`: encode first, permute second.
        let standard: HNSW<DotPacking8U16FixedU8Dataset, Graph<PlainNeighbors>> =
            plain.convert_dataset_into(());
        let permuted = standard.permute_and_encode::<PlainNeighbors>();

        assert_eq!(
            permuted.dataset.space_usage_bytes(),
            standard.dataset.space_usage_bytes(),
            "permuting rows must not change the encoded dataset size"
        );

        for new_id in 0..n {
            let old_id = permuted.get_original_id(new_id);
            assert_eq!(
                permuted.dataset.get(new_id as VectorId).data(),
                standard.dataset.get(old_id as VectorId).data(),
                "row {new_id} (originally {old_id}) was re-encoded instead of copied"
            );
        }
    }
}

#[cfg(test)]
mod acorn_search_tests {
    use super::*;
    use crate::graph::Graph;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
    use vectorium::vector::DenseVectorView;
    use vectorium::{DenseDataset, PlainDenseDataset};

    /// Build a small 1-D HNSW for testing.
    /// Vectors are [0.0], [1.0], ..., [(n-1).0].
    fn build_1d_hnsw(n: usize) -> HNSW<PlainDenseDataset<f32, SquaredEuclideanDistance>, Graph> {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(50);
        HNSW::build_index(dataset, &config)
    }

    /// Every result of `search_filtered` must pass the predicate.
    #[test]
    fn search_filtered_all_results_pass_predicate() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);

        let query_val = [50.0f32];
        let query = DenseVectorView::new(&query_val);

        // Only even IDs allowed.
        let results = hnsw.search_filtered(query, 10, &search_config, |id| id % 2 == 0);

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(
                r.vector % 2,
                0,
                "result {} does not pass even predicate",
                r.vector
            );
        }
    }

    /// The nearest filtered result should be the closest vector satisfying the predicate.
    ///
    /// Query = 50.5.  Predicate: id divisible by 3.
    /// Nearest divisible-by-3 IDs to 50.5 are 51 (d=0.25), 48 (d=6.25), 54 (d=12.25) …
    #[test]
    fn search_filtered_finds_nearest_predicate_passing_neighbors() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(100);

        let query_val = [50.5f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered(query, 5, &search_config, |id| id % 3 == 0);

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.vector % 3, 0, "result {} is not divisible by 3", r.vector);
        }
        // The closest divisible-by-3 vector to 50.5 is 51.
        assert_eq!(
            results[0].vector, 51,
            "expected nearest filtered result 51, got {}",
            results[0].vector
        );
    }

    /// With a predicate that accepts every vector, filtered search must return at
    /// most `k` results and the nearest neighbor must match the unfiltered search.
    #[test]
    fn search_filtered_full_predicate_matches_unfiltered_nearest() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let k = 5;

        let query_val = [30.0f32];
        let query_filtered = DenseVectorView::new(&query_val);
        let query_plain = DenseVectorView::new(&query_val);

        let filtered = hnsw.search_filtered(query_filtered, k, &search_config, |_| true);
        let plain = hnsw.search(query_plain, k, &search_config);

        assert_eq!(filtered.len(), plain.len());
        // Both searches must agree on the nearest neighbor.
        assert_eq!(
            filtered[0].vector, plain[0].vector,
            "nearest neighbor mismatch: filtered={}, plain={}",
            filtered[0].vector, plain[0].vector
        );
    }

    /// When the predicate is very selective (only 1 vector passes), filtered search
    /// must still return exactly that vector — provided the query is placed near it
    /// so the HNSW entry point lands in the same neighbourhood.
    #[test]
    fn search_filtered_single_eligible_vector() {
        let hnsw = build_1d_hnsw(50);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);

        // Query near 42 so the HNSW navigates to that neighbourhood.
        // The two-hop expansion from nearby nodes will reach node 42.
        let query_val = [42.0f32];
        let query = DenseVectorView::new(&query_val);

        // Only vector 42 passes the predicate.
        let results = hnsw.search_filtered(query, 5, &search_config, |id| id == 42);

        assert_eq!(
            results.len(),
            1,
            "expected exactly 1 result, got {}",
            results.len()
        );
        assert_eq!(results[0].vector, 42);
    }

    /// When no vector satisfies the predicate, the result must be empty.
    #[test]
    fn search_filtered_no_eligible_vectors_returns_empty() {
        let hnsw = build_1d_hnsw(50);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);

        let query_val = [25.0f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered(query, 5, &search_config, |_| false);
        assert!(
            results.is_empty(),
            "expected empty results when predicate always returns false"
        );
    }
}

#[cfg(test)]
mod acorn_gamma_search_tests {
    use super::*;
    use crate::graph::Graph;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
    use vectorium::vector::DenseVectorView;
    use vectorium::{DenseDataset, PlainDenseDataset};

    fn build_1d_hnsw(n: usize) -> HNSW<PlainDenseDataset<f32, SquaredEuclideanDistance>, Graph> {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(50);
        HNSW::build_index(dataset, &config)
    }

    /// `build_acorn_gamma_neighbors` with gamma=2 must produce at least as many
    /// neighbors per node as the original ground graph (two-hop union is a superset).
    #[test]
    fn build_acorn_gamma_neighbors_expands_neighbor_lists() {
        let hnsw = build_1d_hnsw(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(2);

        let n = 50usize;
        assert_eq!(acorn_gamma.n_nodes(), n);

        let ground = &hnsw.levels[hnsw.levels.len() - 1];
        for v in 0..n {
            let mut scratch = Vec::new();
            let orig_deg = ground.neighbors(v, &mut scratch).len();
            let mut gamma_scratch = Vec::new();
            let expanded_deg = acorn_gamma.neighbors(v, &mut gamma_scratch).len();
            // Two-hop union is at least as large as one-hop.
            assert!(
                expanded_deg >= orig_deg,
                "node {v}: expanded {expanded_deg} < original {orig_deg}"
            );
        }
    }

    /// gamma = 1 is the case the `2M` floor exists for: `gamma * M` alone would give every node
    /// half the ground level's degree budget, turning the expansion into a contraction.
    #[test]
    fn build_acorn_gamma_neighbors_never_shrinks_below_the_ground_level() {
        let hnsw = build_1d_hnsw(50);
        let ground = &hnsw.levels[hnsw.levels.len() - 1];

        for gamma in [1usize, 2] {
            let acorn_gamma = hnsw.build_acorn_gamma_neighbors(gamma);
            assert!(
                acorn_gamma.max_degree() >= ground.max_degree(),
                "gamma {gamma}: budget {} < ground level {}",
                acorn_gamma.max_degree(),
                ground.max_degree()
            );

            for v in 0..50usize {
                let mut scratch = Vec::new();
                let orig_deg = ground.neighbors(v, &mut scratch).len();
                let mut gamma_scratch = Vec::new();
                let expanded_deg = acorn_gamma.neighbors(v, &mut gamma_scratch).len();
                assert!(
                    expanded_deg >= orig_deg,
                    "gamma {gamma}, node {v}: expanded {expanded_deg} < original {orig_deg}"
                );
            }
        }
    }

    /// Every result of `search_filtered_gamma` must pass the predicate.
    #[test]
    fn search_filtered_gamma_all_results_pass_predicate() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(4);

        let query_val = [50.0f32];
        let query = DenseVectorView::new(&query_val);

        let results =
            hnsw.search_filtered_gamma(query, 10, &search_config, &acorn_gamma, |id| id % 2 == 0);

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.vector % 2, 0, "node {} fails even predicate", r.vector);
        }
    }

    /// With an all-pass predicate, `search_filtered_gamma` must find the true nearest.
    #[test]
    fn search_filtered_gamma_full_predicate_finds_nearest() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(4);

        let query_val = [37.0f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered_gamma(query, 1, &search_config, &acorn_gamma, |_| true);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector, 37, "nearest to 37.0 should be node 37");
    }

    /// When no vector satisfies the predicate, the result must be empty.
    #[test]
    fn search_filtered_gamma_no_eligible_vectors_returns_empty() {
        let hnsw = build_1d_hnsw(50);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(2);

        let query_val = [25.0f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered_gamma(query, 5, &search_config, &acorn_gamma, |_| false);
        assert!(results.is_empty());
    }
}

/// RaBitQ is the first encoder kANNolo uses that takes *query-side* parameters, so these
/// tests cover the whole path: the graph is built over plain f32 vectors, the dataset is
/// re-encoded into RaBitQ codes, and search forwards
/// [`HNSWSearchConfiguration::query_params`] to the encoder.
#[cfg(test)]
mod rabitq_tests {
    use super::*;
    use crate::graph::Graph;
    use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
    use vectorium::vector::DenseVectorView;
    use vectorium::{
        DenseDataset, IndexStats, PlainDenseDataset, RabitqConfig, RabitqExtConfig,
        RabitqExtQuantizer, RabitqQuantizer, RabitqQueryParams,
    };

    /// RaBitQ packs codes into 64-bit words, so the dimension must be a multiple of 64.
    const DIM: usize = 64;
    const N: usize = 256;

    /// Deterministic pseudo-random vectors — a plain LCG, so the tests do not depend on
    /// `rand`'s generator staying fixed across versions.
    fn plain_dataset<D: vectorium::ScalarDenseSupportedDistance>() -> PlainDenseDataset<f32, D> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut values = Vec::with_capacity(N * DIM);
        for _ in 0..N * DIM {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Top 24 bits, mapped to [-1, 1).
            values.push(((state >> 40) as f32 / (1u32 << 23) as f32) - 1.0);
        }
        DenseDataset::from_raw(
            values.into_boxed_slice(),
            N,
            PlainDenseQuantizer::<f32, D>::new(DIM),
        )
    }

    fn build_plain<D>() -> HNSW<PlainDenseDataset<f32, D>, Graph>
    where
        D: vectorium::ScalarDenseSupportedDistance + Distance,
    {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(16)
            .with_ef_construction(100);
        HNSW::build_index(plain_dataset::<D>(), &config)
    }

    /// A dataset vector used verbatim as its own query: whatever the estimator's error,
    /// the exact match should still land in the top-k.
    fn query_vector<D: vectorium::ScalarDenseSupportedDistance>(id: usize) -> Vec<f32> {
        plain_dataset::<D>().get(id as VectorId).values().to_vec()
    }

    #[test]
    fn rabitq_conversion_preserves_element_count() {
        let plain = build_plain::<SquaredEuclideanDistance>();
        let n = plain.n_elements();

        let hnsw: HNSW<DenseDataset<RabitqQuantizer<SquaredEuclideanDistance>>, Graph> =
            plain.convert_dataset_into_ref(RabitqConfig::default());

        assert_eq!(hnsw.n_elements(), n);
    }

    #[test]
    fn rabitq_search_finds_the_exact_match() {
        let hnsw: HNSW<DenseDataset<RabitqQuantizer<SquaredEuclideanDistance>>, Graph> =
            build_plain::<SquaredEuclideanDistance>()
                .convert_dataset_into_ref(RabitqConfig::default());

        let query_values = query_vector::<SquaredEuclideanDistance>(37);
        let query = DenseVectorView::new(&query_values);
        let config = HNSWSearchConfiguration::default()
            .with_ef_search(100)
            .with_query_params(RabitqQueryParams::new(8));

        let results = hnsw.search(query, 10, &config);

        assert_eq!(results.len(), 10);
        assert!(
            results.iter().any(|r| r.vector == 37),
            "the query is dataset vector 37 verbatim, but it is not in the top-10: {:?}",
            results.iter().map(|r| r.vector).collect::<Vec<_>>()
        );
    }

    #[test]
    fn rabitq_search_works_under_dot_product() {
        let hnsw: HNSW<DenseDataset<RabitqQuantizer<DotProduct>>, Graph> =
            build_plain::<DotProduct>().convert_dataset_into_ref(RabitqConfig::default());

        let query_values = query_vector::<DotProduct>(101);
        let query = DenseVectorView::new(&query_values);
        let config = HNSWSearchConfiguration::default()
            .with_ef_search(100)
            .with_query_params(RabitqQueryParams::new(8));

        let results = hnsw.search(query, 10, &config);
        assert_eq!(results.len(), 10);
    }

    /// Every `query_bits` setting must be accepted by the same index: the parameter
    /// describes the query, not the stored data.
    #[test]
    fn one_index_serves_every_query_bits_setting() {
        let hnsw: HNSW<DenseDataset<RabitqQuantizer<SquaredEuclideanDistance>>, Graph> =
            build_plain::<SquaredEuclideanDistance>()
                .convert_dataset_into_ref(RabitqConfig::default());

        let query_values = query_vector::<SquaredEuclideanDistance>(5);
        let query = DenseVectorView::new(&query_values);

        for query_bits in 1..=8 {
            let config = HNSWSearchConfiguration::default()
                .with_ef_search(64)
                .with_query_params(RabitqQueryParams::new(query_bits));
            let results = hnsw.search(query, 10, &config);
            assert_eq!(results.len(), 10, "query_bits={query_bits}");
        }
    }

    /// The sharp plumbing test: `query_bits` is validated inside
    /// `RabitqQuantizer::query_evaluator`, so an out-of-range value can only panic if the
    /// configuration actually reached the encoder. Built as a literal to bypass the same
    /// check in `RabitqQueryParams::new`.
    #[test]
    #[should_panic(expected = "RaBitQ requires query_bits in 1..=8")]
    fn query_params_reach_the_encoder() {
        let hnsw: HNSW<DenseDataset<RabitqQuantizer<SquaredEuclideanDistance>>, Graph> =
            build_plain::<SquaredEuclideanDistance>()
                .convert_dataset_into_ref(RabitqConfig::default());

        let query_values = query_vector::<SquaredEuclideanDistance>(0);
        let query = DenseVectorView::new(&query_values);
        let config = HNSWSearchConfiguration {
            ef_search: 32,
            early_termination: EarlyTerminationStrategy::None,
            query_params: RabitqQueryParams { query_bits: 9 },
        };

        hnsw.search(query, 10, &config);
    }

    #[test]
    fn rabitq_ext_search_finds_the_exact_match() {
        let config = RabitqExtConfig {
            total_bits: 4,
            ..RabitqExtConfig::default()
        };
        let hnsw: HNSW<DenseDataset<RabitqExtQuantizer<SquaredEuclideanDistance>>, Graph> =
            build_plain::<SquaredEuclideanDistance>().convert_dataset_into_ref(config);

        let query_values = query_vector::<SquaredEuclideanDistance>(37);
        let query = DenseVectorView::new(&query_values);
        // RaBitQ-ext takes no query-side parameters, so the default `()` config is enough.
        let search_config = HNSWSearchConfiguration::default().with_ef_search(100);

        let results = hnsw.search(query, 10, &search_config);

        assert_eq!(results.len(), 10);
        assert!(
            results.iter().any(|r| r.vector == 37),
            "the query is dataset vector 37 verbatim, but it is not in the top-10: {:?}",
            results.iter().map(|r| r.vector).collect::<Vec<_>>()
        );
    }
}
