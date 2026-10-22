//! Two-stage indexes: an HNSW first stage that retrieves candidates, and a rerank dataset that
//! rescores them.
//!
//! [`DenseRerankIndex`] reranks a compressed dense first stage (PQ, RaBitQ or RaBitQ-ext) against
//! plain `f16` vectors or RaBitQ-ext codes. [`SparseRerankIndex`] reranks a sparse first stage,
//! optionally L1-pruned and quantized, against `f16` vectors or the pruned-away residuals.
//!
//! Both are built from one collection and saved and loaded as one file. Both implement
//! [`vectorium::Index`], with [`RerankSearchParams`] as search parameters, and
//! [`vectorium::IndexStats`]. The dense index is generic over its metric, `DotProduct` or
//! `SquaredEuclideanDistance`. The Python classes `DenseRerankHNSW` and `SparseRerankHNSW` and the
//! `hnsw_rerank_{build,search}_{dense,sparse}` binaries are built on them.
//!
//! ```no_run
//! use kannolo::rerank::{
//!     DenseEncoder, DenseRerankConfig, DenseRerankIndex, RerankSearchParams, SparseRerankConfig,
//!     SparseRerankIndex,
//! };
//! use vectorium::distances::DotProduct;
//! use vectorium::readers::read_seismic_format;
//! use vectorium::vector::DenseVectorView;
//! use vectorium::{Index, PlainSparseDataset};
//!
//! // Dense: RaBitQ-ext 4-bit first stage, reranked on 8-bit RaBitQ-ext codes.
//! let (dim, data): (usize, Vec<f32>) = (768, vec![0.0; 768 * 1000]);
//! let mut cfg = DenseRerankConfig::new(DenseEncoder::RabitqExt { total_bits: 4 });
//! cfg.rerank_bits = 8;
//! let dense = DenseRerankIndex::<DotProduct>::build(&data, dim, &cfg).unwrap();
//! let query = DenseVectorView::new(&data[..dim]);
//! let top10 = dense.search(query, 10, &RerankSearchParams::new(100, 100));
//!
//! // Sparse: first stage L1-pruned to 60% of each vector's mass, exact residual rescoring.
//! let docs: PlainSparseDataset<u16, f32, DotProduct> =
//!     read_seismic_format("documents.bin").unwrap();
//! let sparse = SparseRerankIndex::build(docs, &SparseRerankConfig::l1_pruned(60.0)).unwrap();
//! sparse.save("index.bin").unwrap();
//! ```

pub mod dense;
pub mod sparse;

pub use dense::{DenseEncoder, DenseRerankConfig, DenseRerankIndex};
pub use sparse::{SparseFirstStage, SparseRerankConfig, SparseRerankIndex};

use crate::graph::neighbors::MAX_NEIGHBORS_PER_NODE;
use crate::hnsw::EarlyTerminationStrategy;

/// An invalid configuration, or a failure to read or write an index file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankError(pub String);

impl std::fmt::Display for RerankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RerankError {}

pub type Result<T> = std::result::Result<T, RerankError>;

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(RerankError(msg.into()))
}

/// How the first-stage graph is stored.
///
/// `Standard` and `Permuted` load as the same type: the permutation is stored in the index data.
/// `Compressed` stores the adjacency lists with StreamVByte.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GraphLayout {
    Standard,
    Permuted,
    Compressed,
}

/// Search parameters, shared by both index types.
#[derive(Copy, Clone, Debug)]
pub struct RerankSearchParams {
    /// First-stage candidates to rerank.
    pub k_candidates: usize,
    /// HNSW candidate list size.
    pub ef_search: usize,
    /// Candidate pruning on the first-stage score, off when `None`.
    pub alpha: Option<f32>,
    /// Early exit during reranking, off when `None`.
    pub beta: Option<usize>,
    pub early_termination: EarlyTerminationStrategy,
}

impl RerankSearchParams {
    /// Parameters with no candidate pruning, no early exit and no early termination.
    pub fn new(k_candidates: usize, ef_search: usize) -> Self {
        Self {
            k_candidates,
            ef_search,
            alpha: None,
            beta: None,
            early_termination: EarlyTerminationStrategy::None,
        }
    }
}

/// Rejects a graph layout that cannot store the ground level of a graph built with `m`.
pub(crate) fn check_graph_layout(graph: GraphLayout, m: usize) -> Result<()> {
    if graph == GraphLayout::Compressed && 2 * m > MAX_NEIGHBORS_PER_NODE {
        return invalid(format!(
            "the compressed graph layout stores at most {MAX_NEIGHBORS_PER_NODE} neighbors per \
             node, but m={m} gives 2 * m = {} on the ground level; use m <= {}",
            2 * m,
            MAX_NEIGHBORS_PER_NODE / 2
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
    use vectorium::vector::{DenseVectorView, SparseVectorView};
    use vectorium::{DatasetGrowable, Index, IndexStats, PlainSparseDataset};
    use vectorium::{PlainSparseDatasetGrowable, PlainSparseQuantizer};

    /// A deterministic pseudo-random stream in [-1, 1).
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn top1<I: Index>(index: &I, query: I::Query<'_>, params: &I::SearchParams) -> usize {
        index.search(query, 1, params)[0].vector as usize
    }

    #[test]
    fn dense_rerank_index_is_an_index() {
        let (n, dim) = (300, 64);
        let data = values(n * dim, 1);
        let mut cfg = DenseRerankConfig::new(DenseEncoder::RabitqExt { total_bits: 4 });
        cfg.m = 8;
        cfg.ef_construction = 50;
        let index = DenseRerankIndex::<SquaredEuclideanDistance>::build(&data, dim, &cfg).unwrap();

        assert_eq!(index.n_elements(), n);
        assert_eq!(IndexStats::dim(&index), dim);

        // With every vector a candidate, the f16 rerank stage finds a stored vector as its own
        // nearest neighbour.
        let params = RerankSearchParams::new(n, n);
        let id = 17;
        let query = DenseVectorView::new(&data[id * dim..(id + 1) * dim]);
        let results = index.search(query, 5, &params);
        assert_eq!(results.len(), 5);
        assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
        assert_eq!(top1(&index, query, &params), id);
    }

    #[test]
    fn sparse_rerank_index_is_an_index() {
        let (n, dim, nnz) = (200, 1000usize, 30);
        let noise = values(n * nnz * 2, 2);
        let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            DatasetGrowable::new(PlainSparseQuantizer::new(dim, dim));
        let mut vectors = Vec::new();
        for i in 0..n {
            let mut comps: Vec<u16> = (0..nnz)
                .map(|j| ((noise[i * nnz + j].abs() * dim as f32) as usize % dim) as u16)
                .collect();
            comps.sort_unstable();
            comps.dedup();
            let vals: Vec<f32> = (0..comps.len())
                .map(|j| noise[n * nnz + i * nnz + j].abs() + 0.1)
                .collect();
            growable.push(SparseVectorView::new(&comps, &vals));
            vectors.push((comps, vals));
        }
        let collection: PlainSparseDataset<u16, f32, DotProduct> = growable.into();
        for first_stage in [SparseFirstStage::Blocked, SparseFirstStage::F16] {
            let mut cfg = SparseRerankConfig::l1_pruned(60.0);
            cfg.first_stage = first_stage;
            cfg.m = 8;
            cfg.ef_construction = 50;
            let index = SparseRerankIndex::build(collection.clone(), &cfg).unwrap();

            assert_eq!(index.n_elements(), n);
            assert_eq!(IndexStats::dim(&index), dim);

            // Residual reranking scores every candidate exactly, so with every vector a candidate the
            // top result for a stored vector is the stored vector with the largest dot product.
            let params = RerankSearchParams::new(n, n);
            let (comps, vals) = &vectors[42];
            let query = SparseVectorView::new(comps.as_slice(), vals.as_slice());
            let exact = |(c, v): &(Vec<u16>, Vec<f32>)| -> f32 {
                c.iter()
                    .zip(v)
                    .filter_map(|(ci, vi)| comps.binary_search(ci).ok().map(|p| vi * vals[p]))
                    .sum()
            };
            let best = (0..n)
                .max_by(|&a, &b| exact(&vectors[a]).total_cmp(&exact(&vectors[b])))
                .unwrap();
            assert_eq!(top1(&index, query, &params), best);

            let path = std::env::temp_dir().join(format!(
                "kannolo_sparse_rerank_{}_{first_stage:?}.bin",
                std::process::id()
            ));
            index.save(path.to_str().unwrap()).unwrap();
            let loaded = SparseRerankIndex::load(
                path.to_str().unwrap(),
                first_stage,
                GraphLayout::Standard,
                true,
            )
            .unwrap();
            std::fs::remove_file(&path).unwrap();
            let ids = |i: &SparseRerankIndex| -> Vec<u64> {
                i.search(query, 10, &params)
                    .iter()
                    .map(|r| r.vector)
                    .collect()
            };
            assert_eq!(ids(&loaded), ids(&index));
        }
    }
}
