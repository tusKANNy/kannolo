use std::cmp::Reverse;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};
use vectorium::core::dataset::ScoredItemGeneric;
use vectorium::distances::Distance;
use vectorium::vector_encoder::{QueryEvaluator, VectorEncoder};
use vectorium::{Dataset, VectorId};

use crate::graph::neighbors::{NeighborData, Neighbors};
use crate::hnsw_utils::from_max_heap_to_min_heap;
use crate::utils::{CompactArray, invert_mapping, validate_permutation};
use crate::visited_set::create_visited_set;

/// A trait that defines the common interface for different graph implementations.
///
/// This allows graph indexes to be generic over the specific graph storage strategy.
/// Graph construction is handled through concrete type constructors and `Default`.
pub trait GraphTrait {
    /// Returns the local IDs of the neighbors of node `u`.
    ///
    /// `scratch` is a work buffer: graphs that can hand out their storage directly ignore it,
    /// the others fill it. **Callers must read only the returned slice**, never `scratch`, whose
    /// length and contents afterwards are unspecified. Returning a slice rather than an iterator
    /// lets callers walk a node's neighbors more than once without paying to produce them twice.
    fn neighbors<'a>(&'a self, u: usize, scratch: &'a mut Vec<u32>) -> &'a [u32];

    /// Returns the number of nodes in the graph.
    #[must_use]
    fn n_nodes(&self) -> usize;

    /// Returns true if the graph is empty, false otherwise.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.n_nodes() == 0
    }

    /// Returns the number of edges in the graph.
    #[must_use]
    fn n_edges(&self) -> usize;

    /// Average out-degree, `n_edges / n_nodes`.
    #[must_use]
    fn avg_degree(&self) -> usize {
        self.n_edges() / self.n_nodes().max(1)
    }

    /// Returns the maximum degree of any node in the graph.
    #[must_use]
    fn max_degree(&self) -> usize;

    /// Returns the external (original dataset) ID of a node given its local graph ID.
    /// If the graph has no external ID mapping, this function returns the local ID itself.
    #[must_use]
    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        id
    }

    /// Returns the memory space used by the graph structure in bytes.
    #[must_use]
    fn get_space_usage_bytes(&self) -> usize;

    /// Greedily searches for the single nearest neighbor to a query, starting from an `entry_point`.
    ///
    /// # Arguments
    /// * `dataset`: The dataset containing the vectors.
    /// * `query_evaluator`: An evaluator that can compute the distance from the query to any vector in the dataset.
    /// * `entry_point`: The candidate (`distance`, `id`) from which the search begins.
    ///
    /// # Returns
    /// The best `ScoredItemGeneric` found during the search.
    #[must_use]
    fn greedy_search_nearest<'e, D>(
        &self,
        dataset: &D,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        entry_point: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
    ) -> ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>
    where
        D: Dataset,
    {
        let mut nearest_id = entry_point.vector;
        let mut nearest_distance = entry_point.distance;
        let mut updated = true;

        let mut scratch: Vec<u32> = Vec::new();

        while updated {
            updated = false;

            for &neighbor in self.neighbors(nearest_id, &mut scratch) {
                let neighbor = neighbor as usize;
                let external_id = self.get_external_id(neighbor);
                let distance_neighbor =
                    query_evaluator.compute_distance(dataset.get(external_id as VectorId));

                if distance_neighbor < nearest_distance {
                    nearest_distance = distance_neighbor;
                    nearest_id = neighbor;
                    updated = true;
                }
            }
        }

        ScoredItemGeneric {
            distance: nearest_distance,
            vector: nearest_id,
        }
    }

    /// Performs a greedy search on the graph to find the top `k` nearest neighbors.
    /// It uses a beam search-like approach, maintaining a list of candidates to visit (`ef`)
    /// and returning the `k` best results found.
    ///
    /// # Arguments
    /// * `dataset`: The dataset containing the vectors.
    /// * `starting_node`: The candidate from which the search begins.
    /// * `query_evaluator`: An evaluator that can compute distances to the query.
    /// * `k`: The number of nearest neighbors to return.
    /// * `ef`: The size of the dynamic candidate list during the search.
    /// * `lambda`: Relaxation parameter used for adaptive early stopping/admission.
    ///
    /// # Returns
    /// A `Vec` containing tuples of `(distance, id)` for the `k` nearest neighbors.
    #[must_use]
    fn greedy_search_topk<'e, D>(
        &self,
        dataset: &'e D,
        starting_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        k: usize,
        ef: usize,
        lambda: f32,
    ) -> Vec<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        let top_candidates = self.search_candidates_for_query(
            dataset,
            starting_node,
            query_evaluator,
            ef,
            k,
            lambda,
        );

        let mut top_k = top_candidates.into_sorted_vec();
        top_k.truncate(k);
        top_k
    }

    /// Search candidates for a query (uses efSearch and top-k pruning).
    #[must_use]
    fn search_candidates_for_query<'e, D>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef_search: usize,
        k: usize,
        lambda: f32,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        self.search_candidates_impl(
            dataset,
            entry_node,
            query_evaluator,
            ef_search,
            Some(k),
            lambda,
        )
    }

    /// Search candidates for insertion (uses efConstruction, no top-k pruning).
    #[must_use]
    fn search_candidates_for_insert<'e, D>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef_construction: usize,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        self.search_candidates_impl(
            dataset,
            entry_node,
            query_evaluator,
            ef_construction,
            None,
            0.0,
        )
    }

    /// Shared implementation for candidate search.
    #[must_use]
    fn search_candidates_impl<'e, D>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef: usize,
        k: Option<usize>,
        lambda: f32,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        // max-heap: We want to substitute worst result with a better one
        let mut top_candidates: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        // min-heap: We want to extract best candidate first to visit it
        let mut candidates: BinaryHeap<
            Reverse<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>,
        > = BinaryHeap::with_capacity(ef);

        let mut visited_table = create_visited_set(ef, lambda, self.avg_degree(), self.n_nodes());

        // Scratch for `neighbors`, reused for every expanded node. The slice it backs is walked
        // twice, to prefetch and to score: producing the list twice would make the compressed
        // backend decode the StreamVByte block again, interleaved with the dataset accesses.
        let mut neighbors_buf: Vec<u32> = Vec::new();

        top_candidates.push(entry_node);
        candidates.push(Reverse(entry_node));

        visited_table.insert(entry_node.vector);

        while let Some(Reverse(node)) = candidates.pop() {
            let id_candidate = node.vector;
            let distance_candidate = node.distance;

            if top_candidates.len() >= ef.max(k.unwrap_or(0)) {
                // Standard HNSW termination: stop when the best remaining candidate
                // is worse than the worst result collected so far.
                let worst_top = &top_candidates.peek().unwrap().distance;
                if !distance_candidate.is_within_relaxation(worst_top, lambda) {
                    break;
                }
            }

            // TODO: This is fine with dense vectors but sub optimal for sparse ones. This is because
            // dataset.range_from_id(id) needs to do a random access to the offsets vector to get the
            // start and end of the vector. It would be better to store the offsets of the neighbors in
            // the graph structure to allow for more efficient prefetching.
            let neighbors = self.neighbors(id_candidate, &mut neighbors_buf);

            for &neighbor_local_id in neighbors {
                let range = dataset
                    .range_from_id(self.get_external_id(neighbor_local_id as usize) as VectorId);
                // 256 bytes: higher values may evict lines the search still needs, 256 is enough to trigger hardware prefetching.
                dataset.prefetch_with_range(range, 256);
            }

            // Buffer up to 6 unvisited neighbours for batch distance computation.
            let mut buf_local: [usize; 6] = [0; 6];
            let mut buf_ext: [VectorId; 6] = [0; 6];
            let mut count = 0usize;

            // Admit a pre-scored candidate into both heaps.
            let mut admit = |local_id: usize, distance: <D::Encoder as VectorEncoder>::Distance| {
                let candidate = ScoredItemGeneric {
                    distance,
                    vector: local_id,
                };
                let should_add = if top_candidates.len() < ef {
                    true
                } else if let Some(top_node) = top_candidates.peek() {
                    candidate
                        .distance
                        .is_within_relaxation(&top_node.distance, lambda)
                } else {
                    false
                };
                if should_add {
                    candidates.push(Reverse(candidate));
                    top_candidates.push(candidate);
                }
                if top_candidates.len() > ef {
                    top_candidates.pop();
                }
            };

            for &neighbor_local_id in neighbors {
                let neighbor_local_id = neighbor_local_id as usize;
                if visited_table.insert(neighbor_local_id) {
                    buf_local[count] = neighbor_local_id;
                    buf_ext[count] = self.get_external_id(neighbor_local_id) as VectorId;
                    count += 1;
                    if count == 6 {
                        let dists = query_evaluator.compute_distances_batch6([
                            dataset.get(buf_ext[0]),
                            dataset.get(buf_ext[1]),
                            dataset.get(buf_ext[2]),
                            dataset.get(buf_ext[3]),
                            dataset.get(buf_ext[4]),
                            dataset.get(buf_ext[5]),
                        ]);
                        for i in 0..6 {
                            admit(buf_local[i], dists[i]);
                        }
                        count = 0;
                    }
                }
            }
            // Flush remaining neighbours (fewer than 6).
            for i in 0..count {
                let d = query_evaluator.compute_distance(dataset.get(buf_ext[i]));
                admit(buf_local[i], d);
            }
        }
        top_candidates
    }

    /// Performs ACORN-1 filtered approximate nearest-neighbor search.
    ///
    /// Only vectors satisfying `predicate(external_id) == true` are returned.
    /// To maintain connectivity in sparse predicate sub-graphs, the search performs a
    /// two-hop neighbor expansion: when a direct neighbor does not satisfy the predicate,
    /// its own neighbors are also inspected ("jumping over" non-matching nodes).
    ///
    /// This expansion is *conditional*, where ACORN-1 as published expands every visited node's
    /// list in full — all one-hop and two-hop neighbors — and truncates the filtered result to
    /// `M`. A passing neighbor enters the frontier and is expanded on its own turn, so it is a
    /// failing neighbor, never expanded, whose connectivity has to be restored here. The two are
    /// not equivalent: a passing neighbor the beam never pops has its own neighbors reached by
    /// the published form and not by this one. Nothing is truncated to `M` either, so the beam
    /// width alone bounds the work.
    ///
    /// The index underneath also differs. Published ACORN-1 is ACORN-γ's construction at
    /// `gamma = 1, M_beta = M`, i.e. each node keeps its `M` nearest candidates and HNSW's
    /// RNG-approximation prune is not applied — ACORN argues that prune is unsound under a
    /// predicate, since it drops `v–b` on the strength of a path `v→a→b` that disappears when
    /// `a` fails the filter. This runs over an ordinary robust-pruned index instead.
    ///
    /// # Arguments
    /// * `dataset` – Dataset containing the raw vectors.
    /// * `entry_node` – `(distance, local_id)` starting point for the search.
    /// * `query_evaluator` – Computes distances from the query to any vector.
    /// * `ef` – Dynamic candidate list size (controls recall vs. speed).
    /// * `k` – Number of results requested (used for early-termination threshold).
    /// * `lambda` – Relaxation parameter for adaptive early stopping (`0.0` = standard HNSW).
    /// * `predicate` – Called with the **external** vector ID; returns `true` for eligible vectors.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    fn acorn_search_candidates_filtered<'e, D, F>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef: usize,
        k: usize,
        lambda: f32,
        predicate: &F,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
        F: Fn(usize) -> bool,
    {
        // max-heap: predicate-satisfying results (worst-first for eviction).
        let mut top_candidates: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        // min-heap: traversal candidates (best-first for greedy exploration).
        let mut candidates: BinaryHeap<
            Reverse<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>,
        > = BinaryHeap::with_capacity(ef);

        let mut visited = create_visited_set(ef, lambda, self.avg_degree(), self.n_nodes());

        // Always add the entry to the traversal set so we can expand from it even
        // when it does not satisfy the predicate.
        visited.insert(entry_node.vector);
        candidates.push(Reverse(entry_node));

        // Only add the entry to the result set if it passes the predicate.
        if predicate(self.get_external_id(entry_node.vector)) {
            top_candidates.push(entry_node);
        }

        // Reusable buffer for non-predicate direct neighbors used in the two-hop phase;
        // allocated once and cleared per iteration to avoid repeated heap allocations.
        let mut non_pred_direct: Vec<usize> = Vec::new();
        // Scratch for `neighbors`, shared by both phases.
        let mut scratch: Vec<u32> = Vec::new();

        while let Some(Reverse(node)) = candidates.pop() {
            // Standard HNSW termination: stop when the best remaining candidate
            // cannot improve the worst result in the result set.
            if top_candidates.len() >= ef.max(k) {
                let worst_top = top_candidates.peek().unwrap().distance;
                if !node.distance.is_within_relaxation(&worst_top, lambda) {
                    break;
                }
            }

            non_pred_direct.clear();

            // --- Phase 1: direct neighbors ---
            // Predicate-satisfying neighbors are admitted to the candidate/result heaps.
            // Non-predicate neighbors are collected for the two-hop phase below.
            for &neighbor_local in self.neighbors(node.vector, &mut scratch) {
                let neighbor_local = neighbor_local as usize;
                if !visited.insert(neighbor_local) {
                    continue;
                }

                let ext = self.get_external_id(neighbor_local);
                if predicate(ext) {
                    let d = query_evaluator.compute_distance(dataset.get(ext as VectorId));
                    let cand = ScoredItemGeneric {
                        distance: d,
                        vector: neighbor_local,
                    };
                    let should_add = if top_candidates.len() < ef {
                        true
                    } else if let Some(top) = top_candidates.peek() {
                        cand.distance.is_within_relaxation(&top.distance, lambda)
                    } else {
                        false
                    };
                    if should_add {
                        candidates.push(Reverse(cand));
                        top_candidates.push(cand);
                    }
                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }
                } else {
                    non_pred_direct.push(neighbor_local);
                }
            }

            // --- Phase 2: two-hop expansion (ACORN-1 core) ---
            // For each non-predicate direct neighbor, inspect its neighbors.
            // This compensates for sparse connectivity in the predicate sub-graph.
            // The phase-1 slice is dead by now, so the same scratch can be reused.
            for &mid_local in &non_pred_direct {
                for &neighbor_local in self.neighbors(mid_local, &mut scratch) {
                    let neighbor_local = neighbor_local as usize;
                    if !visited.insert(neighbor_local) {
                        continue;
                    }

                    let ext = self.get_external_id(neighbor_local);
                    if predicate(ext) {
                        let d = query_evaluator.compute_distance(dataset.get(ext as VectorId));
                        let cand = ScoredItemGeneric {
                            distance: d,
                            vector: neighbor_local,
                        };
                        let should_add = if top_candidates.len() < ef {
                            true
                        } else if let Some(top) = top_candidates.peek() {
                            cand.distance.is_within_relaxation(&top.distance, lambda)
                        } else {
                            false
                        };
                        if should_add {
                            candidates.push(Reverse(cand));
                            top_candidates.push(cand);
                        }
                        if top_candidates.len() > ef {
                            top_candidates.pop();
                        }
                    }
                    // Non-predicate two-hop nodes are not expanded further (no three-hop).
                }
            }
        }

        top_candidates
    }

    /// ACORN-γ filtered search on a pre-expanded neighbor graph.
    ///
    /// Unlike `acorn_search_candidates_filtered`, this method performs **no two-hop
    /// expansion** at query time. It is designed for the pre-expanded adjacency built by
    /// [`crate::hnsw::HNSW::build_acorn_gamma_neighbors`], whose `neighbors()` already returns
    /// γ·M candidates (two-hop union, pruned by distance). Predicate-failing nodes are simply
    /// skipped without further expansion — connectivity is guaranteed by the pre-built lists.
    ///
    /// # Arguments
    /// Same as `acorn_search_candidates_filtered`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    fn acorn_gamma_search_filtered<'e, D, F>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef: usize,
        k: usize,
        lambda: f32,
        predicate: &F,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
        F: Fn(usize) -> bool,
    {
        let mut top_candidates: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        let mut candidates: BinaryHeap<
            Reverse<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>,
        > = BinaryHeap::with_capacity(ef);

        let mut visited = create_visited_set(ef, lambda, self.avg_degree(), self.n_nodes());

        // Always add the entry to C (traversal); add to W (results) only if predicate passes.
        visited.insert(entry_node.vector);
        candidates.push(Reverse(entry_node));
        if predicate(self.get_external_id(entry_node.vector)) {
            top_candidates.push(entry_node);
        }

        let mut scratch: Vec<u32> = Vec::new();

        while let Some(Reverse(node)) = candidates.pop() {
            if top_candidates.len() >= ef.max(k) {
                let worst_top = top_candidates.peek().unwrap().distance;
                if !node.distance.is_within_relaxation(&worst_top, lambda) {
                    break;
                }
            }

            for &neighbor_local in self.neighbors(node.vector, &mut scratch) {
                let neighbor_local = neighbor_local as usize;
                if !visited.insert(neighbor_local) {
                    continue;
                }

                let ext = self.get_external_id(neighbor_local);
                let d = query_evaluator.compute_distance(dataset.get(ext as VectorId));
                let cand = ScoredItemGeneric {
                    distance: d,
                    vector: neighbor_local,
                };

                // Gate on W's current frontier — same threshold as search_candidates_impl.
                let should_add = if top_candidates.len() < ef {
                    true
                } else if let Some(top) = top_candidates.peek() {
                    cand.distance.is_within_relaxation(&top.distance, lambda)
                } else {
                    false
                };

                if should_add {
                    // Always add to C (traversal), even if the node fails the predicate —
                    // non-predicate nodes act as stepping stones through the expanded graph.
                    candidates.push(Reverse(cand));

                    // Only add to W (results) if the predicate passes.
                    if predicate(ext) {
                        top_candidates.push(cand);
                        if top_candidates.len() > ef {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        top_candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorium::DenseDataset;
    use vectorium::core::dataset::ScoredItemGeneric;
    use vectorium::core::vector::DenseVectorView;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;

    /// Build a line graph of `n` nodes (each connected to i-1 and i+1).
    fn build_line_graph(n: usize, max_degree: usize) -> GrowableGraph {
        let mut g = GrowableGraph::with_max_degree(max_degree);
        g.reserve(n);
        g.advance_inserted_nodes(n);
        for i in 0..n {
            let mut nbrs: Vec<usize> = Vec::new();
            if i > 0 {
                nbrs.push(i - 1);
            }
            if i + 1 < n {
                nbrs.push(i + 1);
            }
            g.push_with_precomputed_reverse_links(None, &nbrs, i, &[]);
        }
        g
    }

    /// Build a graph on `n` nodes where node `v` has exactly `nbrs` as neighbors.
    fn build_graph_with(n: usize, max_degree: usize, v: usize, nbrs: &[usize]) -> GrowableGraph {
        let mut g = GrowableGraph::with_max_degree(max_degree);
        g.reserve(n);
        g.advance_inserted_nodes(n);
        g.push_with_precomputed_reverse_links(None, nbrs, v, &[]);
        g
    }

    fn line_dataset(n: usize) -> DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>> {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder)
    }

    /// When the target has room, every incoming reverse edge is appended and the list is
    /// not rewritten.
    #[test]
    fn merge_reverse_links_appends_all_incoming_when_there_is_room() {
        let dataset = line_dataset(20);
        let mut graph = build_graph_with(20, 8, 5, &[4, 6]);

        let incoming = [(5u32, 1u32), (5u32, 2u32)];
        let (old_degree, replacement) = graph.merge_reverse_links(&dataset, 5, &incoming);

        assert_eq!(old_degree, 2);
        assert!(
            replacement.is_none(),
            "2 existing + 2 incoming fits in max_degree 8, so it must append"
        );

        graph.append_links(5, old_degree, incoming.iter().map(|&(_, u)| u));
        let mut scratch: Vec<u32> = Vec::new();
        let neighbors: Vec<usize> = graph
            .neighbors(5, &mut scratch)
            .iter()
            .map(|&n| n as usize)
            .collect();
        assert_eq!(
            neighbors,
            vec![4, 6, 1, 2],
            "both incoming edges must survive the merge"
        );
    }

    /// When the target is full, every incoming edge is still present in the single prune.
    #[test]
    fn merge_reverse_links_prunes_once_with_every_incoming_edge_present() {
        let dataset = line_dataset(20);
        // Node 5's current neighbors are all far away; the two incoming nodes are its
        // nearest possible neighbors, so a correct merge must retain both.
        let mut graph = build_graph_with(20, 4, 5, &[15, 16, 17]);

        let incoming = [(5u32, 4u32), (5u32, 6u32)];
        let (old_degree, replacement) = graph.merge_reverse_links(&dataset, 5, &incoming);

        assert_eq!(old_degree, 3);
        let replacement =
            replacement.expect("3 existing + 2 incoming exceeds max_degree 4, so it must prune");
        assert!(replacement.len() <= 4);
        assert!(
            replacement.contains(&4) && replacement.contains(&6),
            "both incoming edges were in the prune input and are the closest \
             candidates, so both must be kept; got {replacement:?}"
        );

        graph.replace_links(5, &replacement, old_degree);
        let mut scratch: Vec<u32> = Vec::new();
        let neighbors: Vec<usize> = graph
            .neighbors(5, &mut scratch)
            .iter()
            .map(|&n| n as usize)
            .collect();
        assert!(neighbors.contains(&4) && neighbors.contains(&6));
        assert!(neighbors.len() <= 4);
    }

    /// Regression test for the 2026-03-06 early-termination bug.
    ///
    /// `search_candidates_for_insert` passes `k=None` to `search_candidates_impl`.
    /// Before the fix, `if let Some(k_limit) = k && ...` never fired for k=None,
    /// so the loop drained the entire candidate queue. The fix uses `k.unwrap_or(0)`,
    /// making termination fire as soon as `top_candidates.len() >= ef`.
    ///
    /// This test verifies:
    /// 1. The returned heap is bounded by ef.
    /// 2. The true nearest neighbour is found (search is still correct after the fix).
    #[test]
    fn search_candidates_for_insert_bounded_by_ef_and_finds_nearest() {
        let n = 20usize;
        let ef = 5usize;

        // 1-D dataset: vectors [0.0], [1.0], ..., [19.0].
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        // Query at 10.0 — true nearest neighbour is node 10 (distance 0).
        let query_val = [10.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let entry_dist = evaluator.compute_distance(dataset.get(0));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 0usize,
        };

        let top = graph.search_candidates_for_insert(&dataset, entry, &evaluator, ef);

        assert!(top.len() <= ef, "heap size {} exceeds ef={}", top.len(), ef);

        let best = top.into_sorted_vec().into_iter().next().unwrap();
        assert_eq!(
            best.vector, 10,
            "expected nearest node 10, got {}",
            best.vector
        );
        assert_eq!(best.distance, SquaredEuclideanDistance::from(0.0));
    }

    /// Verify the `k=Some` path (`search_candidates_for_query`) is also bounded and correct.
    #[test]
    fn search_candidates_for_query_bounded_and_correct() {
        let n = 20usize;
        let ef = 6usize;
        let k = 3usize;

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        let query_val = [5.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let entry_dist = evaluator.compute_distance(dataset.get(0));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 0usize,
        };

        let top_heap = graph.search_candidates_for_query(&dataset, entry, &evaluator, ef, k, 0.0);

        assert!(top_heap.len() <= ef);
        let mut results = top_heap.into_sorted_vec();
        results.truncate(k);

        assert_eq!(
            results[0].vector, 5,
            "expected nearest node 5, got {}",
            results[0].vector
        );
        assert_eq!(results[0].distance, SquaredEuclideanDistance::from(0.0));
    }

    /// All results from `acorn_search_candidates_filtered` must satisfy the predicate.
    ///
    /// Dataset: 1-D vectors [0.0 … 19.0] on a line graph.
    /// Query: 10.0.  Predicate: only even IDs.
    /// The nearest even node is 10 (distance 0); the next nearest are 8 and 12.
    #[test]
    fn acorn_filtered_results_all_pass_predicate() {
        let n = 20usize;
        let k = 5usize;
        let ef = 10usize;

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        let query_val = [10.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let entry_dist = evaluator.compute_distance(dataset.get(0));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 0usize,
        };

        let predicate = |id: usize| id.is_multiple_of(2);
        let top_heap = graph
            .acorn_search_candidates_filtered(&dataset, entry, &evaluator, ef, k, 0.0, &predicate);

        assert!(!top_heap.is_empty(), "expected at least one result");
        for result in &top_heap {
            assert_eq!(
                result.vector % 2,
                0,
                "node {} fails even predicate",
                result.vector
            );
        }

        // The nearest even node to 10.0 is node 10 itself (distance 0).
        let best = top_heap.into_sorted_vec().into_iter().next().unwrap();
        assert_eq!(
            best.vector, 10,
            "expected nearest even node 10, got {}",
            best.vector
        );
        assert_eq!(best.distance, SquaredEuclideanDistance::from(0.0));
    }

    /// When the entry point does not satisfy the predicate the search must still
    /// find predicate-passing nodes via two-hop expansion.
    ///
    /// Line graph [0 … 19].  Entry = node 0 (fails `id >= 2`).
    /// - Phase 1: neighbor 1 also fails → collected for two-hop.
    /// - Phase 2: neighbors of 1 are {0, 2}; 0 is already visited; 2 passes → admitted.
    ///
    /// So the two-hop expansion must discover node 2 even though neither entry
    /// nor its direct neighbor satisfy the predicate.
    #[test]
    fn acorn_filtered_entry_not_in_predicate_still_finds_results() {
        let n = 20usize;
        let k = 3usize;
        let ef = 10usize;

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        // Entry at node 0 — does NOT satisfy `id >= 2`.
        // Node 1 (direct neighbor) also does not satisfy it.
        // Node 2 (two hops away) does satisfy it and must be found.
        let query_val = [3.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let entry_dist = evaluator.compute_distance(dataset.get(0));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 0usize,
        };

        let predicate = |id: usize| id >= 2;
        let top_heap = graph
            .acorn_search_candidates_filtered(&dataset, entry, &evaluator, ef, k, 0.0, &predicate);

        assert!(
            !top_heap.is_empty(),
            "expected results even when entry and its direct neighbors fail predicate"
        );
        for result in &top_heap {
            assert!(
                result.vector >= 2,
                "node {} fails id>=2 predicate",
                result.vector
            );
        }
    }

    /// With a predicate that accepts every node, filtered search must return the same
    /// nearest neighbor as the standard unfiltered search.
    #[test]
    fn acorn_filtered_all_pass_matches_unfiltered_nearest() {
        let n = 20usize;
        let ef = 8usize;
        let k = 3usize;

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        let query_val = [7.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let entry_dist = evaluator.compute_distance(dataset.get(0));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 0usize,
        };

        let all_pass = |_: usize| true;
        let top_heap = graph
            .acorn_search_candidates_filtered(&dataset, entry, &evaluator, ef, k, 0.0, &all_pass);

        let mut results = top_heap.into_sorted_vec();
        results.truncate(k);

        // Node 7 is the exact nearest neighbor.
        assert_eq!(
            results[0].vector, 7,
            "expected nearest node 7, got {}",
            results[0].vector
        );
        assert_eq!(results[0].distance, SquaredEuclideanDistance::from(0.0));
    }
}

#[cfg(test)]
mod acorn_gamma_tests {
    use super::*;
    use vectorium::DenseDataset;
    use vectorium::core::dataset::ScoredItemGeneric;
    use vectorium::core::vector::DenseVectorView;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;

    fn build_line_graph(n: usize, max_degree: usize) -> GrowableGraph {
        let mut g = GrowableGraph::with_max_degree(max_degree);
        g.reserve(n);
        g.advance_inserted_nodes(n);
        for i in 0..n {
            let mut nbrs: Vec<usize> = Vec::new();
            if i > 0 {
                nbrs.push(i - 1);
            }
            if i + 1 < n {
                nbrs.push(i + 1);
            }
            g.push_with_precomputed_reverse_links(None, &nbrs, i, &[]);
        }
        g
    }

    /// All results from `acorn_gamma_search_filtered` must satisfy the predicate.
    ///
    /// On a standard (non-pre-expanded) graph, the method behaves like a predicate-aware
    /// beam search that simply skips non-matching nodes — no two-hop expansion.
    /// With a line graph and even-IDs predicate, nearest even node to 10.0 is node 10.
    #[test]
    fn acorn_gamma_filtered_results_all_pass_predicate() {
        let n = 20usize;
        let k = 5usize;
        let ef = 10usize;

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        let query_val = [10.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let entry_dist = evaluator.compute_distance(dataset.get(10));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 10usize,
        };

        let predicate = |id: usize| id.is_multiple_of(2);
        let top_heap =
            graph.acorn_gamma_search_filtered(&dataset, entry, &evaluator, ef, k, 0.0, &predicate);

        assert!(!top_heap.is_empty(), "expected at least one result");
        for result in &top_heap {
            assert_eq!(
                result.vector % 2,
                0,
                "node {} fails even predicate",
                result.vector
            );
        }
        let best = top_heap.into_sorted_vec().into_iter().next().unwrap();
        assert_eq!(
            best.vector, 10,
            "expected nearest even node 10, got {}",
            best.vector
        );
        assert_eq!(best.distance, SquaredEuclideanDistance::from(0.0));
    }

    /// With an all-pass predicate, `acorn_gamma_search_filtered` must return the same
    /// nearest neighbor as the standard unfiltered search.
    #[test]
    fn acorn_gamma_filtered_all_pass_matches_unfiltered_nearest() {
        let n = 20usize;
        let ef = 8usize;
        let k = 3usize;

        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let graph = build_line_graph(n, 4);

        let query_val = [7.0f32];
        let query = DenseVectorView::new(&query_val);
        let evaluator = dataset.encoder().query_evaluator(query, &());
        let entry_dist = evaluator.compute_distance(dataset.get(7));
        let entry = ScoredItemGeneric {
            distance: entry_dist,
            vector: 7usize,
        };

        let all_pass = |_: usize| true;
        let top_heap =
            graph.acorn_gamma_search_filtered(&dataset, entry, &evaluator, ef, k, 0.0, &all_pass);

        let mut results = top_heap.into_sorted_vec();
        results.truncate(k);

        assert_eq!(
            results[0].vector, 7,
            "expected nearest node 7, got {}",
            results[0].vector
        );
        assert_eq!(results[0].distance, SquaredEuclideanDistance::from(0.0));
    }
}

/// A representation of a graph whose adjacency lists are stored through a pluggable
/// [`Neighbors`] backend (e.g. plain `u32`s, or delta+StreamVByte compressed).
/// Node ids are represented as `u32` internally but returned as `usize`.
///
/// # Fields
/// - `neighbors`: The backend holding every node's adjacency list.
/// - `ids_mapping`: Optional local-id -> external-id mapping, used when the graph's node
///   order has been permuted relative to the dataset it was built from.
#[derive(Serialize, Deserialize)]
pub struct Graph<N: Neighbors> {
    neighbors: N,
    ids_mapping: Option<CompactArray>,
    max_degree: usize,
    n_nodes: usize,
}

impl<N: Neighbors + Default> Default for Graph<N> {
    fn default() -> Self {
        Graph {
            neighbors: N::default(),
            ids_mapping: None,
            max_degree: 0,
            n_nodes: 0,
        }
    }
}

impl<N: Neighbors> GraphTrait for Graph<N> {
    #[inline]
    fn neighbors<'a>(&'a self, id: usize, scratch: &'a mut Vec<u32>) -> &'a [u32] {
        self.neighbors.get(id, scratch)
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.neighbors.len()
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            if id >= mapping.len() {
                panic!("ID out of bounds: {}", id);
            }
            mapping.get(id)
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.space_usage_bytes();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, CompactArray::space_usage_bytes);

        neighbors_size + ids_mapping_size
    }
}

impl<N: Neighbors + From<NeighborData>> From<GrowableGraph> for Graph<N> {
    /// Converts a `GrowableGraph` into a compact `Graph` over any backend, by removing padding.
    ///
    /// Neighbor order within each node is left untouched unless the target backend demands
    /// otherwise (`N::REQUIRES_SORTED`), so the `standard` build path keeps producing exactly the
    /// same graph as before the backend became pluggable. For a delta-coded backend the lists are
    /// sorted and deduplicated here, which is what makes such a backend reachable straight from a
    /// build rather than only through the reordering pass.
    fn from(growable_graph: GrowableGraph) -> Self {
        let n_nodes = growable_graph.n_nodes();
        let max_degree = growable_graph.max_degree();

        let mut neighbors: Vec<u32> = Vec::with_capacity(growable_graph.neighbors.len());
        let mut offsets = Vec::with_capacity(n_nodes + 1);

        offsets.push(0);
        for v in 0..n_nodes {
            let start = v * max_degree;
            let end = start + max_degree;
            let slots = &growable_graph.neighbors[start..end];
            let len = slots
                .iter()
                .position(|&v| v == u32::MAX)
                .unwrap_or(max_degree);
            let list_start = neighbors.len();
            neighbors.extend_from_slice(&slots[..len]);
            if N::REQUIRES_SORTED {
                let list = &mut neighbors[list_start..];
                list.sort_unstable();
                let kept = {
                    let mut w = list_start;
                    for i in list_start..neighbors.len() {
                        if i == list_start || neighbors[i] != neighbors[i - 1] {
                            neighbors[w] = neighbors[i];
                            w += 1;
                        }
                    }
                    w
                };
                neighbors.truncate(kept);
            }
            offsets.push(neighbors.len());
        }

        let final_mapping = growable_graph.ids_mapping.map(CompactArray::from);

        Graph {
            neighbors: N::from(NeighborData {
                data: neighbors.into_boxed_slice(),
                offsets: offsets.into_boxed_slice(),
            }),
            ids_mapping: final_mapping,
            max_degree,
            n_nodes,
        }
    }
}

impl<N> Graph<N>
where
    N: Neighbors + From<NeighborData>,
{
    /// Builds a graph directly from adjacency lists.
    ///
    /// This is the entry point for passes that produce their own neighbor lists rather than
    /// deriving them from a [`GrowableGraph`] or from another `Graph` — the ACORN-γ expansion of
    /// [`crate::hnsw::HNSW::build_acorn_gamma_neighbors`] is one such pass.
    ///
    /// `max_degree` is the degree bound the caller wants reported; it is not enforced here.
    /// `ids_mapping` is the optional local-to-external ID translation (`None` when the two ID
    /// spaces coincide, which is the case for a ground level); it is bit-packed on the way in,
    /// exactly as the [`GrowableGraph`] conversion does.
    ///
    /// Note that delta-coded backends such as [`crate::graph::neighbors::StreamVByteNeighbors`]
    /// require each list to be sorted ascending; a caller handing over lists in any other order
    /// must use a backend that stores them verbatim.
    pub fn from_neighbor_data(
        data: NeighborData,
        max_degree: usize,
        ids_mapping: Option<Vec<usize>>,
    ) -> Self {
        let n_nodes = data.offsets.len().saturating_sub(1);
        Graph {
            neighbors: N::from(data),
            ids_mapping: ids_mapping.map(CompactArray::from),
            max_degree,
            n_nodes,
        }
    }
}

impl<Nsrc> Graph<Nsrc>
where
    Nsrc: Neighbors,
{
    /// Applies `mapping` (`old_external_id -> new_external_id`, e.g. from
    /// [`crate::graph::egb::permute_graph_bisection_from_graph`]) to one level, rebuilding its
    /// adjacency lists into a fresh `Ndst` backend.
    ///
    /// How the relabelling lands depends on how the level stores its external IDs:
    ///
    /// - **With an `ids_mapping`** (the upper levels): only that array is rewritten, so every node
    ///   keeps the local slot it already occupies. Local IDs are independent of the dataset, so
    ///   leaving them alone keeps the neighbor lists — which hold local IDs — valid untouched, and
    ///   preserves the nested-prefix relationship between levels for free.
    /// - **Without one** (the ground level): local IDs *are* external IDs, so the same relabelling
    ///   can only be expressed by reordering the nodes and remapping every neighbor value. Only
    ///   then is the inverse permutation (`new_id -> old_id`) meaningful, and only then is it
    ///   returned — the caller stores it to translate search results back to original dataset IDs.
    pub fn permute_level<Ndst>(&self, mapping: &[usize]) -> (Graph<Ndst>, Option<Vec<usize>>)
    where
        Ndst: Neighbors + From<NeighborData>,
    {
        validate_permutation(mapping).unwrap_or_else(|e| panic!("{e}"));

        let n = self.n_nodes;

        // `inv` being `Some` is what marks this level as the reordered one, for the loop below and
        // for the caller alike.
        let inv = if self.ids_mapping.is_none() {
            assert_eq!(
                mapping.len(),
                n,
                "invalid graph permutation length: expected {}, got {}",
                n,
                mapping.len()
            );
            Some(invert_mapping(mapping))
        } else {
            None
        };

        let mut all_neighbors = Vec::with_capacity(self.n_edges());
        let mut new_offsets = Vec::with_capacity(n + 1);
        let mut new_external_ids = Vec::with_capacity(if inv.is_some() { 0 } else { n });
        let mut neigh = Vec::new();
        let mut scratch: Vec<u32> = Vec::new();
        new_offsets.push(0);

        for new_local in 0..n {
            neigh.clear();

            match &inv {
                // Reordered: slot `new_local` takes over the node that used to sit at `inv[..]`,
                // and its neighbor values are local IDs that the reorder moves too, so each one
                // goes through `mapping`.
                Some(inv) => neigh.extend(
                    self.neighbors(inv[new_local], &mut scratch)
                        .iter()
                        .map(|&v| mapping[v as usize] as u32),
                ),
                // Left in place: the neighbor values stay valid as they are, and only the node's
                // external ID is relabelled so it points at the co-permuted dataset.
                None => {
                    neigh.extend_from_slice(self.neighbors(new_local, &mut scratch));

                    let old_global = self.get_external_id(new_local);
                    assert!(
                        old_global < mapping.len(),
                        "invalid upper-level external id {} for global permutation length {}",
                        old_global,
                        mapping.len()
                    );
                    new_external_ids.push(mapping[old_global]);
                }
            }

            // Ascending order is a prerequisite of delta coding, not a preference: the decoder
            // reconstructs by prefix sum. The build emits each list in arbitrary order.
            neigh.sort_unstable();
            neigh.dedup();

            all_neighbors.extend_from_slice(&neigh);
            new_offsets.push(all_neighbors.len());
        }

        let graph = Graph {
            neighbors: Ndst::from(NeighborData {
                data: all_neighbors.into_boxed_slice(),
                offsets: new_offsets.into_boxed_slice(),
            }),
            ids_mapping: inv.is_none().then(|| CompactArray::from(new_external_ids)),
            max_degree: self.max_degree,
            n_nodes: n,
        };
        (graph, inv)
    }
}

// GrowableGraph is build-only and never serialized.
#[derive(Default)]
pub struct GrowableGraph {
    // `u32::MAX` marks an empty slot, the same sentinel the fixed-stride backend uses. Storing
    // plain `u32` rather than an optional wrapper is what lets `neighbors` hand out a subslice
    // of this buffer instead of copying into the caller's scratch on every node expansion,
    // which the construction search does tens of millions of times.
    neighbors: Vec<u32>,
    ids_mapping: Option<Vec<usize>>,
    max_degree: usize,
    n_edges: usize,
    n_nodes: usize,
    inserted_nodes: usize,
}

impl GraphTrait for GrowableGraph {
    /// Hands out a subslice of the stored buffer; `scratch` is left untouched, as the
    /// [`Neighbors`] contract allows for backends that can return their storage directly.
    #[inline]
    fn neighbors<'a>(&'a self, u: usize, _scratch: &'a mut Vec<u32>) -> &'a [u32] {
        let start = u * self.max_degree;
        let end = start + self.max_degree;
        let slots = &self.neighbors[start..end];
        // Neighbors occupy a contiguous prefix; the first sentinel ends the list.
        let len = slots
            .iter()
            .position(|&v| v == u32::MAX)
            .unwrap_or(self.max_degree);
        &slots[..len]
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            if id >= mapping.len() {
                panic!("ID out of bounds: {}", id);
            }
            mapping[id]
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<u32>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        neighbors_size + ids_mapping_size
    }
}

impl<N: Neighbors> From<Graph<N>> for GrowableGraph {
    fn from(graph: Graph<N>) -> Self {
        let max_degree = graph.max_degree;
        let n_nodes = graph.n_nodes;
        let n_edges = graph.neighbors.len();
        let mut neighbors: Vec<u32> = Vec::with_capacity(n_nodes * max_degree);

        let mut scratch: Vec<u32> = Vec::new();
        for v in 0..n_nodes {
            let list = graph.neighbors.get(v, &mut scratch);
            neighbors.extend_from_slice(list);
            let pad = max_degree.saturating_sub(list.len());
            neighbors.extend((0..pad).map(|_| u32::MAX));
        }

        let ids_mapping = graph.ids_mapping.map(|mapping| mapping.to_vec());

        GrowableGraph {
            neighbors,
            ids_mapping,
            max_degree,
            n_edges,
            n_nodes,
            inserted_nodes: n_nodes,
        }
    }
}

impl GrowableGraph {
    /// Creates a new `GrowableGraph` with the specified maximum degree.
    #[must_use]
    pub fn with_max_degree(max_degree: usize) -> Self {
        GrowableGraph {
            neighbors: Vec::new(),
            ids_mapping: None, // No mapping by default
            max_degree,
            n_edges: 0,
            n_nodes: 0,
            inserted_nodes: 0, // No nodes inserted yet
        }
    }

    /// Returns the number of nodes that have been inserted into the graph.
    #[must_use]
    #[inline]
    pub fn inserted_nodes(&self) -> usize {
        self.inserted_nodes
    }

    /// Advances the count of inserted nodes by a given amount.
    /// This is used by the parallel builder to update the state after a batch is processed.
    pub fn advance_inserted_nodes(&mut self, count: usize) {
        self.inserted_nodes += count;
    }

    /// Pre-allocates space for a fixed number of nodes.
    pub fn reserve(&mut self, n_expected_nodes: usize) {
        self.neighbors = vec![u32::MAX; n_expected_nodes * self.max_degree];
        self.n_nodes = n_expected_nodes;
        self.ids_mapping = None;
    }

    /// Sets the ID mapping for the graph, converting local IDs to external/original IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the mapping length does not match the number of nodes in the graph.
    pub fn set_mapping(&mut self, mapping: Vec<usize>) -> Result<(), String> {
        if mapping.len() != self.n_nodes {
            return Err(format!(
                "Mapping length mismatch: got {}, expected {}",
                mapping.len(),
                self.n_nodes
            ));
        }
        self.ids_mapping = Some(mapping);
        Ok(())
    }

    /// A version of push for the parallel builder that accepts pre-computed reverse links.
    pub fn push_with_precomputed_reverse_links(
        &mut self,
        external_id: Option<usize>,
        neighbors: &[usize],
        local_id: usize,
        reverse_links: &[(usize, Vec<usize>)], // (neighbor_id, new_neighbor_list_for_it)
    ) {
        let new_node_local_id = local_id;

        // Add forward links
        let start = new_node_local_id * self.max_degree;
        for (i, &neighbor) in neighbors.iter().enumerate() {
            self.neighbors[start + i] = neighbor as u32;
        }
        self.n_edges += neighbors.len();

        if let Some(vec_id) = external_id {
            if let Some(mapping) = self.ids_mapping.as_mut() {
                if new_node_local_id >= mapping.len() {
                    panic!(
                        "Attempted to write to local_id {} but ids_mapping len is {}",
                        new_node_local_id,
                        mapping.len()
                    );
                }
                mapping[new_node_local_id] = vec_id;
            } else {
                panic!("Attempted to set external ID for a graph without an ID mapping.");
            }
        } else {
            // If no external ID is provided, we assume the local ID is the external ID
            if let Some(mapping) = self.ids_mapping.as_mut() {
                if new_node_local_id >= mapping.len() {
                    panic!(
                        "Attempted to write to local_id {} but ids_mapping len is {}",
                        new_node_local_id,
                        mapping.len()
                    );
                }
                mapping[new_node_local_id] = new_node_local_id;
            }
        }

        // Add pre-computed reverse links
        for (neighbor_id, new_neighbor_list) in reverse_links {
            let start = *neighbor_id * self.max_degree;
            for (i, &n) in new_neighbor_list.iter().enumerate() {
                self.neighbors[start + i] = n as u32;
            }
            for i in new_neighbor_list.len()..self.max_degree {
                self.neighbors[start + i] = u32::MAX;
            }
        }
    }

    /// Write the forward links of one newly inserted node.
    /// Does NOT update `n_edges` or `ids_mapping` — the caller handles those separately.
    pub fn write_links(&mut self, forward: &[usize], local_id: usize) {
        let start = local_id * self.max_degree;
        for (i, &nbr) in forward.iter().enumerate() {
            self.neighbors[start + i] = nbr as u32;
        }
        // Forward-link slots beyond forward.len() stay None (set by reserve()), no padding needed.
    }

    /// Number of neighbors currently stored for node `u`.
    #[must_use]
    #[inline]
    pub fn degree(&self, u: usize) -> usize {
        let start = u * self.max_degree;
        self.neighbors[start..start + self.max_degree]
            .iter()
            .take_while(|&&v| v != u32::MAX)
            .count()
    }

    /// Append `ids` to `node`'s neighbor list, starting at slot `at`.
    /// The caller guarantees `at` is the node's current degree and that the
    /// appended entries fit within `max_degree`.
    pub fn append_links(&mut self, node: usize, at: usize, ids: impl Iterator<Item = u32>) {
        let start = node * self.max_degree + at;
        for (i, id) in ids.enumerate() {
            self.neighbors[start + i] = id;
        }
    }

    /// Replace `node`'s neighbor list with `ids`, clearing the tail left over
    /// from the previous list of length `old_degree`.
    pub fn replace_links(&mut self, node: usize, ids: &[usize], old_degree: usize) {
        let start = node * self.max_degree;
        for (i, &id) in ids.iter().enumerate() {
            self.neighbors[start + i] = id as u32;
        }
        for i in ids.len()..old_degree {
            self.neighbors[start + i] = u32::MAX;
        }
    }

    /// Write a single ids_mapping entry. Called before the batch's links are committed, so
    /// that reverse-link writes can resolve external IDs of nodes being inserted.
    pub fn write_id_mapping(&mut self, local_id: usize, external_id: Option<usize>) {
        if let Some(mapping) = self.ids_mapping.as_mut() {
            mapping[local_id] = external_id.unwrap_or(local_id);
        }
    }

    /// Add to the edge count after a batch completes.
    pub fn add_n_edges(&mut self, count: usize) {
        self.n_edges += count;
    }

    pub fn precompute_reverse_links<D>(
        &self,
        dataset: &D,
        node_to_insert_local_id: usize,
        forward_neighbors: &[usize],
    ) -> Vec<(usize, Vec<usize>)>
    // (neighbor_local_id, new_neighbor_list_for_it)
    where
        D: Dataset + Sync,
    {
        let mut reverse_links_data = Vec::with_capacity(forward_neighbors.len());
        let mut scratch: Vec<u32> = Vec::new();

        for &neighbor_local_id in forward_neighbors {
            let neighbor_external_id = self.get_external_id(neighbor_local_id) as VectorId;

            // 1. Build a max-heap containing the neighbor's current neighbors and the new node.
            //    The distances are all relative to `neighbor_external_id`.
            let mut closest_vectors = BinaryHeap::<
                ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
            >::new();

            // Add its current neighbors
            for &local_id in self.neighbors(neighbor_local_id, &mut scratch) {
                let local_id = local_id as usize;
                let external_id = self.get_external_id(local_id) as VectorId;
                let dist = dataset.encoder().compute_distance_between(
                    dataset.get(neighbor_external_id),
                    dataset.get(external_id),
                );
                closest_vectors.push(ScoredItemGeneric {
                    distance: dist,
                    vector: local_id,
                });
            }

            // Add the new reverse link (the node we are inserting)
            let node_to_insert_external_id =
                self.get_external_id(node_to_insert_local_id) as VectorId;
            let dist_to_inserted_node = dataset.encoder().compute_distance_between(
                dataset.get(neighbor_external_id),
                dataset.get(node_to_insert_external_id),
            );
            closest_vectors.push(ScoredItemGeneric {
                distance: dist_to_inserted_node,
                vector: node_to_insert_local_id,
            });

            // 2. Use the robust `shrink_neighbor_list` heuristic to prune the list.
            let new_neighbor_list =
                self.shrink_neighbor_list(dataset, &mut closest_vectors, self.max_degree);

            reverse_links_data.push((neighbor_local_id, new_neighbor_list));
        }
        reverse_links_data
    }

    /// Merge one batch's incoming reverse edges into `target`'s neighbor list.
    ///
    /// `incoming` is the contiguous run of `(target, new_node)` pairs that selected
    /// `target` as a forward neighbor, sorted by the caller. Every incoming edge is
    /// present in the prune input, so nothing is silently discarded, and the sort is what
    /// makes that input a canonical sequence for a given batch — independent of thread
    /// scheduling, hence deterministic.
    ///
    /// The new nodes of a batch are never neighbors of an already-inserted `target`
    /// (they are not in the graph the batch searched), so `incoming` never duplicates
    /// an existing entry.
    ///
    /// Returns `target`'s degree before the merge, and either the pruned replacement for
    /// the whole list or `None` when every incoming edge fits and can just be appended
    /// at that degree.
    #[must_use]
    pub fn merge_reverse_links<D>(
        &self,
        dataset: &D,
        target: usize,
        incoming: &[(u32, u32)],
    ) -> (usize, Option<Vec<usize>>)
    where
        D: Dataset + Sync,
    {
        let old_degree = self.degree(target);

        // `shrink_neighbor_list` returns its input unchanged when it already fits, so
        // when there is room for every incoming edge the merged list is exactly
        // `neighbors(target) ++ incoming`. Appending it avoids both the distance
        // computations and the full-list rewrite.
        if old_degree + incoming.len() <= self.max_degree {
            return (old_degree, None);
        }

        let target_external = self.get_external_id(target) as VectorId;
        let mut closest_vectors =
            BinaryHeap::<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>::new();

        let mut scratch: Vec<u32> = Vec::new();
        let existing = self.neighbors(target, &mut scratch);

        for local_id in existing
            .iter()
            .map(|&n| n as usize)
            .chain(incoming.iter().map(|&(_, u)| u as usize))
        {
            let external_id = self.get_external_id(local_id) as VectorId;
            let dist = dataset
                .encoder()
                .compute_distance_between(dataset.get(target_external), dataset.get(external_id));
            closest_vectors.push(ScoredItemGeneric {
                distance: dist,
                vector: local_id,
            });
        }

        (
            old_degree,
            Some(self.shrink_neighbor_list(dataset, &mut closest_vectors, self.max_degree)),
        )
    }

    pub fn shrink_neighbor_list<D>(
        &self,
        dataset: &D,
        closest_vectors: &mut BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        >,
        max_size: usize,
    ) -> Vec<usize>
    where
        D: Dataset + Sync,
    {
        if closest_vectors.len() <= max_size {
            return closest_vectors
                .iter()
                .map(|candidate| candidate.vector)
                .collect();
        }

        let mut min_heap = from_max_heap_to_min_heap(closest_vectors);
        let mut new_closest_vectors: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        while let Some(node) = min_heap.pop() {
            let node1 = node.0;
            let mut keep_node_1 = true;

            // The robust pruning heuristic from the paper.
            // For each candidate, check if it is closer to the query than it is to any
            // other candidate already in the result set.
            for node2 in new_closest_vectors.iter() {
                let node1_external = self.get_external_id(node1.vector) as VectorId;
                let node2_external = self.get_external_id(node2.vector) as VectorId;
                let dist_node_1_node2 = dataset.encoder().compute_distance_between(
                    dataset.get(node1_external),
                    dataset.get(node2_external),
                );
                if dist_node_1_node2 < node1.distance {
                    keep_node_1 = false;
                    break;
                }
            }

            if keep_node_1 {
                new_closest_vectors.push(node1);
                if new_closest_vectors.len() >= max_size {
                    return new_closest_vectors.iter().map(|c| c.vector).collect();
                }
            }
        }

        // Return the IDs of the closest vectors
        new_closest_vectors
            .iter()
            .map(|candidate| candidate.vector)
            .collect()
    }

    /// Finds and prunes neighbors for a new node and computes the necessary reverse links.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `Vec<usize>`: The pruned forward neighbors for the new node.
    /// - `Vec<(usize, Vec<usize>)>`: The pre-computed reverse links for existing neighbors.
    /// - `ScoredItemGeneric`: The best candidate found, to be used as the entry point for the next lower level.
    #[must_use]
    // The three components are documented above; a single-use alias would only move the
    // type away from the signature that explains it.
    #[allow(clippy::type_complexity)]
    pub fn find_and_prune_neighbors<'e, D>(
        &self,
        dataset: &'e D,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        ef_construction: usize,
        m: usize,
        future_local_id: usize,
    ) -> (
        Vec<usize>,
        Vec<(usize, Vec<usize>)>,
        ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
    )
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        let (forward_neighbors, new_entry_node) =
            self.search_and_prune_forward(dataset, query_evaluator, entry_node, ef_construction, m);

        // 3. Compute reverse links with the PRUNED list
        let reverse_links =
            self.precompute_reverse_links(dataset, future_local_id, &forward_neighbors);

        (forward_neighbors, reverse_links, new_entry_node)
    }

    /// Searches the graph for a new node's neighbors and prunes them to `m`.
    ///
    /// The batched builder collects reverse edges across the whole
    /// batch and merges them per target instead.
    ///
    /// # Returns
    /// - `Vec<usize>`: the pruned forward neighbors for the new node.
    /// - `ScoredItemGeneric`: the entry point for the next lower level.
    #[must_use]
    pub fn search_and_prune_forward<'e, D>(
        &self,
        dataset: &'e D,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        ef_construction: usize,
        m: usize,
    ) -> (
        Vec<usize>,
        ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
    )
    where
        D: Dataset + Sync,
        <D::Encoder as VectorEncoder>::Distance: Distance,
    {
        // 1. Get candidate neighbors
        let mut neighbors_nodes = self.search_candidates_for_insert(
            dataset,
            entry_node,
            query_evaluator,
            ef_construction,
        );

        // The new entry point for the next level is the best candidate we found.
        let new_entry_node = *neighbors_nodes.peek().unwrap();

        // 2. Prune with heuristic
        let forward_neighbors = self.shrink_neighbor_list(dataset, &mut neighbors_nodes, m);

        (forward_neighbors, new_entry_node)
    }
}
