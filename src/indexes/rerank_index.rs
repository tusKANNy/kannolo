//! An implementation of a two-stage retrieval index.
//!
//! ## Architecture
//!
//! The two-stage approach combines a first-stage index, that usually employs lighter vector
//! representations for fast candidates retrieval, with a usually heavier/slower but higher-quality
//! representation for candidates re-scoring and final selection.
//!
//! 1. **First Stage**: Uses an ANN index (e.g., HNSW from kANNolo) to retrieve k_candidates
//!    from a potentially lower-precision or quantized representation.
//!
//! 2. **Second Stage**: Reranks the retrieved candidates using a higher-quality dataset
//!    (e.g., full-precision vectors or multi-vector representations) to compute exact scores
//!    and return the top k_final results.
//!
//! By default the second-stage score *replaces* the first-stage one. The optional residual mode
//! (the `residuals` flag of [`RerankIndex::search`]) instead *sums* the two, for stages that split
//! one score between them.

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::fmt;
use vectorium::Dataset;
use vectorium::IndexSerializer;
use vectorium::QueryEvaluator;
use vectorium::VectorId;
use vectorium::core::dataset::ScoredVector;
use vectorium::core::index::{Index, IndexStats};
use vectorium::core::vector_encoder::VectorEncoder;
use vectorium::distances::Distance;

/// Sums a first-stage score into a rerank score, for residual reranking.
///
/// `first_stage` is `None` when residual reranking is off, in which case the rerank score passes
/// through untouched. See [`RerankIndex::search`] for what the caller must guarantee about the two
/// scores being summable.
#[inline]
fn add_first_stage_score<D: Distance + From<f32>>(rerank: D, first_stage: Option<f32>) -> D {
    match first_stage {
        Some(score) => (rerank.distance() + score).into(),
        None => rerank,
    }
}

/// A reranking index that combines a first-stage index for candidate retrieval
/// with a higher-quality dataset for reranking.
///
/// # Type Parameters
/// - `FirstStageIndex`: The index type (e.g., HNSW from kANNolo) that implements Index
/// - `RerankDataset`: The dataset type used for reranking (e.g., MultiVectorDataset, DenseDataset)
///
/// # Constraints
/// Both the first-stage index and rerank dataset must contain the same number of items
/// with the same vector IDs (same semantic content, different representations).
#[derive(Serialize, Deserialize)]
pub struct RerankIndex<FirstStageIndex, RerankDataset>
where
    FirstStageIndex: Index,
    RerankDataset: Dataset,
{
    first_stage_index: FirstStageIndex,
    rerank_dataset: RerankDataset,
}

impl<FirstStageIndex, RerankDataset> RerankIndex<FirstStageIndex, RerankDataset>
where
    FirstStageIndex: Index,
    RerankDataset: Dataset,
{
    /// Creates a new rerank index from a first-stage index and a rerank dataset.
    ///
    /// # Arguments
    /// - `first_stage_index`: The first-stage index for candidate retrieval
    /// - `rerank_dataset`: The dataset for reranking candidates
    ///
    /// # Note
    /// The caller should ensure that both the first-stage index and rerank dataset
    /// have the same number of items with matching vector IDs.
    pub fn new(first_stage_index: FirstStageIndex, rerank_dataset: RerankDataset) -> Self {
        Self {
            first_stage_index,
            rerank_dataset,
        }
    }

    /// Performs a complete two-stage search: candidate retrieval followed by reranking.
    ///
    /// This is the main search method that:
    /// 1. Retrieves k_candidates from the first-stage index
    /// 2. Reranks all candidates using the rerank dataset  
    /// 3. Returns the top k_final results along with timing information
    ///
    /// # Arguments
    /// - `first_stage_query`: Query vector for the first-stage index
    /// - `rerank_query`: Query vector for reranking
    /// - `k_candidates`: Number of candidates to retrieve from first-stage index
    /// - `k_final`: Number of final results to return after reranking
    /// - `first_stage_search_params`: Search parameters for the first-stage index
    /// - `alpha`: Optional Candidates Pruning (CP) threshold parameter (None for no pruning)
    /// - `beta`: Optional early-exit (EE) parameter for reranking (None for no early exit)
    /// - `residuals`: When `true`, score each candidate as the *sum* of its first-stage and rerank
    ///   scores instead of the rerank score alone (see below). Pass `false` for the plain two-stage
    ///   behaviour.
    ///
    /// # Residual reranking
    /// With `residuals = true` the returned score of a candidate is
    /// `first_stage_score + rerank_score`, and that combined value drives both the final ordering
    /// and the early-exit comparisons. This is meant for setups where the rerank dataset holds the
    /// *residual* part of a decomposed representation (e.g. a first stage scoring centroids and a
    /// rerank dataset scoring what the centroids left out), so the two scores add up to the full
    /// score.
    ///
    /// **The caller is responsible for the two scores being summable.** Nothing here checks that
    /// they share a metric, a scale, or even a sign convention: summing an unrelated first-stage
    /// score into the rerank score produces a well-ordered but meaningless ranking. In particular
    /// the sum is a genuine decomposition only for an additive metric such as dot product — adding
    /// two squared Euclidean distances is not the distance to anything.
    ///
    /// # Returns
    /// - results: Vector of scored vectors, sorted by distance using the Distance type's Ord implementation
    #[allow(clippy::too_many_arguments)]
    pub fn search<'q>(
        &self,
        first_stage_query: <FirstStageIndex as Index>::Query<'q>,
        rerank_query: <RerankDataset::Encoder as VectorEncoder>::QueryVector<'q>,
        k_candidates: usize,
        k_final: usize,
        first_stage_search_params: &FirstStageIndex::SearchParams,
        rerank_search_params: &<RerankDataset::Encoder as VectorEncoder>::QueryParams,
        alpha: Option<f32>,
        beta: Option<usize>,
        residuals: bool,
    ) -> Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>>
    where
        <RerankDataset::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    {
        // Stage 1: Get candidates from first stage index using its own search method
        let first_stage_results = self.first_stage_index.search(
            first_stage_query,
            k_candidates,
            first_stage_search_params,
        );

        // Alpha-based candidate pruning (CP): keep only candidates whose first-stage score is
        // within a relative slack `alpha` of the k-th best one.
        // The comparison goes through [`Distance::is_within_relaxation`].
        let threshold = match alpha {
            Some(alpha_val)
                if k_final > 0
                    && !first_stage_results.is_empty()
                    && k_final <= first_stage_results.len() =>
            {
                Some((first_stage_results[k_final - 1].distance, alpha_val))
            }
            _ => None,
        };

        // Extract candidate IDs from first-stage search results, keeping their first-stage scores
        // alongside when residual reranking needs them.
        let mut candidates: Vec<VectorId> = Vec::with_capacity(first_stage_results.len());
        let mut scores: Vec<f32> = Vec::with_capacity(if residuals {
            first_stage_results.len()
        } else {
            0
        });
        for result in &first_stage_results {
            let score = result.distance.distance();
            if threshold.is_some_and(|(t, a)| !result.distance.is_within_relaxation(&t, a)) {
                continue;
            }
            candidates.push(result.vector);
            if residuals {
                scores.push(score);
            }
        }
        let first_stage_scores = residuals.then_some(scores.as_slice());

        // Stage 2: Rerank candidates using the rerank dataset
        if let Some(beta_val) = beta {
            self.rerank_candidates_with_early_exit(
                rerank_query,
                rerank_search_params,
                &candidates,
                k_final,
                beta_val,
                first_stage_scores,
            )
        } else {
            self.rerank_candidates(
                rerank_query,
                rerank_search_params,
                &candidates,
                k_final,
                first_stage_scores,
            )
        }
    }

    /// Reranks a given list of candidates using the rerank dataset.
    ///
    /// This method takes candidate IDs (typically from a graph index search) and
    /// reranks them using the high-quality rerank dataset.
    ///
    /// # Arguments
    /// - `rerank_query`: Query vector for reranking
    /// - `candidates`: List of vector IDs from initial search
    /// - `k_final`: Number of final results to return
    /// - `first_stage_scores`: First-stage scores of `candidates`, in the same order, when residual
    ///   reranking is on; `None` to score by the rerank dataset alone
    ///
    /// # Returns
    /// Vector of scored vectors, sorted by distance using the Distance type's Ord implementation
    fn rerank_candidates<'q>(
        &self,
        rerank_query: <RerankDataset::Encoder as VectorEncoder>::QueryVector<'q>,
        rerank_search_params: &<RerankDataset::Encoder as VectorEncoder>::QueryParams,
        candidates: &[VectorId],
        k_final: usize,
        first_stage_scores: Option<&[f32]>,
    ) -> Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>>
    where
        <RerankDataset::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    {
        debug_assert!(first_stage_scores.is_none_or(|s| s.len() == candidates.len()));

        // Create query evaluator for the rerank dataset
        let encoder = self.rerank_dataset.encoder();
        let query_evaluator = encoder.query_evaluator(rerank_query, rerank_search_params);

        // Rerank candidates by computing exact distances
        let mut reranked: Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>> =
            candidates
                .iter()
                .enumerate()
                .map(|(i, vector_id)| {
                    let encoded_vector = self.rerank_dataset.get(*vector_id);
                    let distance = query_evaluator.compute_distance(encoded_vector);
                    ScoredVector {
                        distance: add_first_stage_score(distance, first_stage_scores.map(|s| s[i])),
                        vector: *vector_id,
                    }
                })
                .collect();

        // Sort by distance using the Distance type's Ord implementation and take top k_final
        reranked.sort_by_key(|b| b.distance);
        reranked.truncate(k_final);

        reranked
    }

    fn rerank_candidates_with_early_exit<'q>(
        &self,
        rerank_query: <RerankDataset::Encoder as VectorEncoder>::QueryVector<'q>,
        rerank_search_params: &<RerankDataset::Encoder as VectorEncoder>::QueryParams,
        candidates: &[VectorId],
        k_final: usize,
        beta: usize,
        first_stage_scores: Option<&[f32]>,
    ) -> Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>>
    where
        <RerankDataset::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    {
        debug_assert!(first_stage_scores.is_none_or(|s| s.len() == candidates.len()));

        // Ensure there are enough candidates
        if candidates.len() < k_final {
            return Vec::new();
        }

        let encoder = self.rerank_dataset.encoder();
        let query_evaluator = encoder.query_evaluator(rerank_query, rerank_search_params);

        // Rerank first k_final candidates
        let first_reranked: Vec<ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>> =
            candidates[..k_final]
                .iter()
                .enumerate()
                .map(|(i, vector_id)| {
                    let encoded_vector = self.rerank_dataset.get(*vector_id);
                    let distance = query_evaluator.compute_distance(encoded_vector);
                    ScoredVector {
                        distance: add_first_stage_score(distance, first_stage_scores.map(|s| s[i])),
                        vector: *vector_id,
                    }
                })
                .collect();

        // Create a max heap to keep track of the top k_final candidates
        let mut heap: BinaryHeap<
            ScoredVector<<RerankDataset::Encoder as VectorEncoder>::Distance>,
        > = BinaryHeap::from(first_reranked);

        let mut n_stalls = 0;
        for (i, vector_id) in candidates[k_final..].iter().enumerate() {
            let encoded_vector = self.rerank_dataset.get(*vector_id);
            let distance = query_evaluator.compute_distance(encoded_vector);
            let candidate = ScoredVector {
                distance: add_first_stage_score(
                    distance,
                    first_stage_scores.map(|s| s[k_final + i]),
                ),
                vector: *vector_id,
            };

            if let Some(mut worst) = heap.peek_mut() {
                if candidate.distance < worst.distance {
                    *worst = candidate;
                    n_stalls = 0;
                } else {
                    n_stalls += 1;
                    if n_stalls >= beta {
                        break;
                    }
                }
            }
        }

        // Extract results from heap
        heap.into_sorted_vec()
    }

    /// Returns a reference to the underlying first-stage index.
    pub fn first_stage_index(&self) -> &FirstStageIndex {
        &self.first_stage_index
    }

    /// Returns a reference to the rerank dataset.
    pub fn rerank_dataset(&self) -> &RerankDataset {
        &self.rerank_dataset
    }

    /// Returns the number of items in the index.
    pub fn len(&self) -> usize {
        self.rerank_dataset.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.rerank_dataset.is_empty()
    }
}

/// Search parameters of a [`RerankIndex`] used through the [`Index`] trait.
///
/// `first_stage` and `rerank` are the parameters of the first-stage index and of the rerank
/// dataset's encoder. See [`RerankIndex::search`] for `k_candidates`, `alpha`, `beta` and
/// `residuals`.
#[derive(Clone, Debug)]
pub struct RerankSearchParams<FirstStageParams, RerankParams> {
    /// Candidates retrieved from the first stage and reranked.
    pub k_candidates: usize,
    pub first_stage: FirstStageParams,
    pub rerank: RerankParams,
    /// Candidate pruning on the first-stage score; `None` for no pruning.
    pub alpha: Option<f32>,
    /// Early exit during reranking; `None` for no early exit.
    pub beta: Option<usize>,
    /// Sum the first-stage and rerank scores instead of taking the rerank score alone.
    pub residuals: bool,
}

impl<FirstStageParams, RerankParams> RerankSearchParams<FirstStageParams, RerankParams> {
    /// Parameters with no candidate pruning, no early exit and no residual reranking.
    pub fn new(k_candidates: usize, first_stage: FirstStageParams, rerank: RerankParams) -> Self {
        Self {
            k_candidates,
            first_stage,
            rerank,
            alpha: None,
            beta: None,
            residuals: false,
        }
    }
}

/// A [`RerankIndex`] is an [`Index`] whose query is the pair (first-stage query, rerank query).
///
/// The two are often the same vector; they are separate because the stages can use different
/// representations, such as a sparse first stage reranked against multi-vectors.
impl<FirstStageIndex, RerankDataset> Index for RerankIndex<FirstStageIndex, RerankDataset>
where
    FirstStageIndex: Index,
    RerankDataset: Dataset,
    <RerankDataset::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
{
    type Query<'q> = (
        <FirstStageIndex as Index>::Query<'q>,
        <RerankDataset::Encoder as VectorEncoder>::QueryVector<'q>,
    );
    type Distance = <RerankDataset::Encoder as VectorEncoder>::Distance;
    type SearchParams = RerankSearchParams<
        <FirstStageIndex as Index>::SearchParams,
        <RerankDataset::Encoder as VectorEncoder>::QueryParams,
    >;

    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<Self::Distance>> {
        let (first_stage_query, rerank_query) = query;
        RerankIndex::search(
            self,
            first_stage_query,
            rerank_query,
            search_params.k_candidates,
            k,
            &search_params.first_stage,
            &search_params.rerank,
            search_params.alpha,
            search_params.beta,
            search_params.residuals,
        )
    }
}

/// Element count and dimensionality of the rerank dataset.
impl<FirstStageIndex, RerankDataset> IndexStats for RerankIndex<FirstStageIndex, RerankDataset>
where
    FirstStageIndex: Index,
    RerankDataset: Dataset,
{
    fn n_elements(&self) -> usize {
        self.rerank_dataset.len()
    }

    fn dim(&self) -> usize {
        self.rerank_dataset.input_dim()
    }
}

impl<FirstStageIndex, RerankDataset> fmt::Debug for RerankIndex<FirstStageIndex, RerankDataset>
where
    FirstStageIndex: Index + fmt::Debug,
    RerankDataset: Dataset + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankIndex")
            .field("first_stage_index", &self.first_stage_index)
            .field("rerank_dataset", &self.rerank_dataset)
            .finish()
    }
}

/// A `RerankIndex` is not a [`Dataset`], so vectorium's blanket impl does not reach it, but it
/// derives `Serialize`/`Deserialize` and therefore serializes through the same single-file bincode
/// format as every other index.
impl<FirstStageIndex, RerankDataset> IndexSerializer for RerankIndex<FirstStageIndex, RerankDataset>
where
    FirstStageIndex: Index,
    RerankDataset: Dataset,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorium::core::flat_index::FlatIndex;
    use vectorium::core::vector::DenseVectorView;
    use vectorium::datasets::dense_dataset::DenseDatasetGrowable;
    use vectorium::distances::DotProduct;
    use vectorium::{DatasetGrowable, DenseDataset, PlainDenseQuantizer};

    type Enc = PlainDenseQuantizer<f32, DotProduct>;

    fn dataset(vectors: &[[f32; 2]]) -> DenseDataset<Enc> {
        let mut growable = DenseDatasetGrowable::new(Enc::new(2));
        for vector in vectors {
            growable.push(DenseVectorView::new(vector));
        }
        growable.into()
    }

    /// First stage scores the "centroid" part of each vector, the rerank dataset the "residual"
    /// part, so under dot product the two scores sum to the score of the full vector.
    ///
    /// For the query `[1, 0]` used by the tests below, first-stage scores are `[3, 0, 0.5]` and
    /// rerank scores `[0, 2, 1]` — the two rank the candidates in opposite orders, and their sums
    /// `[3, 2, 1.5]` in a third one.
    fn index() -> RerankIndex<FlatIndex<DenseDataset<Enc>>, DenseDataset<Enc>> {
        let centroids = dataset(&[[3.0, 0.0], [0.0, 0.0], [0.5, 0.0]]);
        let residuals = dataset(&[[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]]);
        RerankIndex::new(FlatIndex::from(centroids), residuals)
    }

    fn scored(results: &[ScoredVector<DotProduct>]) -> Vec<(VectorId, f32)> {
        results
            .iter()
            .map(|r| (r.vector, r.distance.distance()))
            .collect()
    }

    #[test]
    fn residuals_off_scores_by_rerank_dataset_alone() {
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 0.0]);

        let results = index.search(query, query, 3, 3, &(), &(), None, None, false);

        assert_eq!(scored(&results), vec![(1, 2.0), (2, 1.0), (0, 0.0)]);
    }

    #[test]
    fn residuals_on_sums_both_stages() {
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 0.0]);

        let results = index.search(query, query, 3, 3, &(), &(), None, None, true);

        assert_eq!(scored(&results), vec![(0, 3.0), (1, 2.0), (2, 1.5)]);
    }

    #[test]
    fn residuals_on_with_early_exit_sums_both_stages() {
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 0.0]);

        // Only one candidate is left to scan past the initial k_final, so beta = 3 can never
        // trigger: this checks that the early-exit path ranks on the combined score just like the
        // plain path, with no candidate cut short.
        let results = index.search(query, query, 3, 2, &(), &(), None, Some(3), true);

        assert_eq!(scored(&results), vec![(0, 3.0), (1, 2.0)]);
    }

    /// `k_final = 0` used to underflow while computing the alpha threshold, which indexes
    /// `first_stage_results[k_final - 1]`.
    #[test]
    fn alpha_with_zero_k_final_returns_empty() {
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 0.0]);

        for beta in [None, Some(1)] {
            for residuals in [false, true] {
                let results =
                    index.search(query, query, 3, 0, &(), &(), Some(0.1), beta, residuals);
                assert!(results.is_empty(), "beta {beta:?}, residuals {residuals}");
            }
        }
    }

    /// The mismatched-stage case: a dense PQ first stage and a sparse second stage, i.e. two
    /// different encoders over two different representations, whose only common ground is the dot
    /// product convention and the vector ids. The scores here are not hand-predictable — the PQ
    /// score is an estimate — so the test asserts the relation between the two runs instead: every
    /// combined score must be the residuals-off score plus the first-stage score of that same id.
    #[test]
    fn sum_a_pq_first_stage_into_a_sparse_rerank() {
        use std::collections::HashMap;
        use vectorium::core::vector::SparseVectorView;
        use vectorium::encoders::pq::ProductQuantizer;
        use vectorium::{PlainSparseDataset, PlainSparseDatasetGrowable, PlainSparseQuantizer};

        const M: usize = 4;
        const DENSE_DIM: usize = 8;
        const SPARSE_DIM: usize = 16;
        const KSUB: usize = 256;

        // Pretrained centroids rather than k-means: instant, deterministic, and the PQ's accuracy
        // is irrelevant here. Layout is [M × KSUB × dsub].
        let dsub = DENSE_DIM / M;
        let centroids: Vec<f32> = (0..M * KSUB * dsub)
            .map(|i| (i % 97) as f32 * 0.05 - 2.0)
            .collect();
        let pq = ProductQuantizer::<M, DotProduct>::from_pretrained(DENSE_DIM, centroids);

        let dense_docs = [
            [0.9f32, -0.3, 0.5, 0.1, -0.7, 0.2, 0.4, -0.1],
            [-0.2f32, 0.8, -0.4, 0.6, 0.3, -0.9, 0.1, 0.5],
            [0.4f32, 0.4, 0.9, -0.5, 0.2, 0.7, -0.3, 0.8],
        ];
        let mut first_stage = DenseDatasetGrowable::new(pq);
        for doc in &dense_docs {
            first_stage.push(DenseVectorView::new(doc));
        }
        let first_stage: DenseDataset<ProductQuantizer<M, DotProduct>> = first_stage.into();

        let sparse_docs: [(&[u16], &[f32]); 3] = [
            (&[0, 3, 9], &[1.0, -2.0, 0.5]),
            (&[1, 3, 12], &[2.0, 1.5, -1.0]),
            (&[0, 7, 15], &[-0.5, 3.0, 2.0]),
        ];
        let mut rerank = PlainSparseDatasetGrowable::<u16, f32, DotProduct>::new(
            PlainSparseQuantizer::new(SPARSE_DIM, SPARSE_DIM),
        );
        for (components, values) in &sparse_docs {
            rerank.push(SparseVectorView::new(components, values));
        }
        let rerank: PlainSparseDataset<u16, f32, DotProduct> = rerank.into();

        let index = RerankIndex::new(FlatIndex::from(first_stage), rerank);

        let dense_query = [0.6f32, 0.2, -0.5, 0.9, 0.1, 0.3, -0.8, 0.4];
        let dense_query = DenseVectorView::new(&dense_query);
        let sparse_query_components: Vec<u16> = (0..SPARSE_DIM as u16).collect();
        let sparse_query_values: Vec<f32> =
            (0..SPARSE_DIM).map(|i| 0.25 * (i as f32) - 1.0).collect();
        let sparse_query = SparseVectorView::new(&sparse_query_components, &sparse_query_values);

        let first_stage_scores: HashMap<VectorId, f32> = index
            .first_stage_index()
            .search(dense_query, 3, &())
            .iter()
            .map(|r| (r.vector, r.distance.distance()))
            .collect();

        // Without this the two runs would agree and the test would prove nothing.
        assert!(
            first_stage_scores.values().any(|s| s.abs() > 1e-3),
            "first-stage scores are all ~0: {first_stage_scores:?}"
        );

        let rerank_only =
            index.search(dense_query, sparse_query, 3, 3, &(), &(), None, None, false);
        let combined = index.search(dense_query, sparse_query, 3, 3, &(), &(), None, None, true);
        assert_eq!(rerank_only.len(), 3);
        assert_eq!(combined.len(), 3);

        for result in &combined {
            let rerank_score = rerank_only
                .iter()
                .find(|r| r.vector == result.vector)
                .expect("id missing from the residuals-off run")
                .distance
                .distance();
            let expected = rerank_score + first_stage_scores[&result.vector];
            assert!(
                (result.distance.distance() - expected).abs() <= 1e-4,
                "id {}: combined {} != rerank {} + first stage {}",
                result.vector,
                result.distance.distance(),
                rerank_score,
                first_stage_scores[&result.vector]
            );
        }
    }

    #[test]
    fn residuals_stay_aligned_with_alpha_pruned_candidates() {
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 0.0]);

        // First-stage order is 0 (3.0), 2 (0.5), 1 (0.0); the threshold 0.5 × (1 − 0.1) drops
        // candidate 1. Candidate 2 must keep its own first-stage score, giving 0.5 + 1.0.
        let results = index.search(query, query, 3, 2, &(), &(), Some(0.1), None, true);

        assert_eq!(scored(&results), vec![(0, 3.0), (2, 1.5)]);
    }

    /// Candidate pruning must keep the *nearest* candidates under a minimize metric.
    ///
    /// Written as `score >= d_k * (1 - alpha)` the rule is correct for dot product and exactly
    /// inverted for squared Euclidean, where it drops the closest candidates — the exact match
    /// first. Here the first stage ranks 2 (d=0), 1 (d=1), 0 (d=9); with `k_final = 2` the
    /// threshold is candidate 1's distance, so pruning may only ever remove candidate 0.
    #[test]
    fn alpha_pruning_keeps_the_nearest_candidates_under_a_minimize_metric() {
        use vectorium::distances::SquaredEuclideanDistance;

        type L2Enc = PlainDenseQuantizer<f32, SquaredEuclideanDistance>;
        fn l2_dataset(vectors: &[[f32; 2]]) -> DenseDataset<L2Enc> {
            let mut growable = DenseDatasetGrowable::new(L2Enc::new(2));
            for vector in vectors {
                growable.push(DenseVectorView::new(vector));
            }
            growable.into()
        }

        let points = l2_dataset(&[[3.0, 0.0], [1.0, 0.0], [0.0, 0.0]]);
        let rerank = l2_dataset(&[[3.0, 0.0], [1.0, 0.0], [0.0, 0.0]]);
        let index = RerankIndex::new(FlatIndex::from(points), rerank);
        let query = DenseVectorView::new(&[0.0f32, 0.0]);

        for alpha in [None, Some(0.2), Some(0.45)] {
            let results = index.search(query, query, 3, 2, &(), &(), alpha, None, false);
            let ids: Vec<VectorId> = results.iter().map(|r| r.vector).collect();
            assert_eq!(
                ids,
                vec![2, 1],
                "alpha = {alpha:?} pruned a nearer candidate than the k-th"
            );
        }
    }

    /// The `Index` impl forwards every parameter to the inherent `search`.
    #[test]
    fn index_trait_matches_inherent_search() {
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 0.0]);

        for residuals in [false, true] {
            for beta in [None, Some(3)] {
                for alpha in [None, Some(0.5)] {
                    let mut params = RerankSearchParams::new(3, (), ());
                    params.alpha = alpha;
                    params.beta = beta;
                    params.residuals = residuals;
                    let via_trait = Index::search(&index, (query, query), 2, &params);
                    let inherent =
                        index.search(query, query, 3, 2, &(), &(), alpha, beta, residuals);
                    assert_eq!(scored(&via_trait), scored(&inherent));
                }
            }
        }
    }

    /// Generic code written against `Index` can search a `RerankIndex`.
    #[test]
    fn rerank_index_is_usable_as_a_generic_index() {
        fn top<I: Index>(index: &I, query: I::Query<'_>, params: &I::SearchParams) -> VectorId {
            index.search(query, 1, params)[0].vector
        }
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 0.0]);
        let mut params = RerankSearchParams::new(3, (), ());
        assert_eq!(top(&index, (query, query), &params), 1);
        params.residuals = true;
        assert_eq!(top(&index, (query, query), &params), 0);
    }

    #[test]
    fn index_stats_describe_the_rerank_dataset() {
        let index = index();
        assert_eq!(index.n_elements(), 3);
        assert_eq!(IndexStats::dim(&index), 2);
    }

    /// A `RerankIndex` round-trips as one file, carrying both stages.
    ///
    /// It is not a `Dataset`, so it reaches `IndexSerializer` through its own impl rather than
    /// vectorium's blanket one; this pins that the impl is present and that both stages survive.
    #[test]
    fn index_serializer_round_trip_rerank_index() {
        let index = index();
        let query = DenseVectorView::new(&[1.0f32, 1.0]);
        let expected = index.search(query, query, 3, 3, &(), &(), None, None, false);

        let mut path = std::env::temp_dir();
        path.push(format!(
            "kannolo_rerank_round_trip_{}.bin",
            std::process::id()
        ));
        index.save_index(path.to_str().unwrap()).unwrap();
        let loaded: RerankIndex<FlatIndex<DenseDataset<Enc>>, DenseDataset<Enc>> =
            RerankIndex::load_index(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.len(), index.len());
        let actual = loaded.search(query, query, 3, 3, &(), &(), None, None, false);
        let expected_ids: Vec<_> = expected.iter().map(|s| s.vector).collect();
        let actual_ids: Vec<_> = actual.iter().map(|s| s.vector).collect();
        assert_eq!(actual_ids, expected_ids);
    }
}
