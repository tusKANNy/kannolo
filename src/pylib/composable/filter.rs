//! Filtered search: the filters accepted from Python, as bitsets, and ACORN-1 and ACORN-γ on
//! HNSW. Flat and IVF indexes reject filtered search for now.

// In a build without IVF or without one of the representations, part of this code is unused.
#![cfg_attr(
    not(all(
        feature = "dense",
        feature = "sparse",
        feature = "multivec",
        feature = "ivf"
    )),
    allow(dead_code, unused_imports)
)]

use vectorium::core::dataset::ScoredVector;
use vectorium::core::index::Index;
use vectorium::core::vector_encoder::VectorEncoder;
use vectorium::distances::Distance;
use vectorium::{Dataset, FlatIndex, SpaceUsage};

use crate::graph::{Graph, GraphTrait};
use crate::hnsw::HNSW;
use crate::ivf::IVF;

/// A set of original ids, as a bitset over the collection.
pub(crate) struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    fn empty(n: usize) -> Self {
        Self {
            words: vec![0; n.div_ceil(64)],
        }
    }

    /// 64 mask entries per word, without a branch per entry.
    pub(crate) fn from_mask(mask: &[bool]) -> Self {
        let mut b = Self::empty(mask.len());
        for (word, chunk) in b.words.iter_mut().zip(mask.chunks(64)) {
            *word = chunk
                .iter()
                .enumerate()
                .fold(0u64, |w, (bit, &m)| w | ((m as u64) << bit));
        }
        b
    }

    /// Ids at or past `n` are not in the collection and are left out.
    pub(crate) fn from_ids(ids: &[u64], n: usize) -> Self {
        let mut b = Self::empty(n);
        for &id in ids.iter().filter(|&&id| (id as usize) < n) {
            b.words[id as usize >> 6] |= 1 << (id & 63);
        }
        b
    }

    #[inline]
    fn contains(&self, id: usize) -> bool {
        self.words
            .get(id >> 6)
            .is_some_and(|w| (w >> (id & 63)) & 1 == 1)
    }
}

/// The error of filtered search on anything but a single-stage HNSW index.
pub(crate) const FILTER_UNSUPPORTED: &str =
    "filtered search applies to HNSW first stages (single-stage indexes, for now)";
pub(crate) const ACORN_GAMMA_UNSUPPORTED: &str =
    "ACORN-γ applies to HNSW first stages (single-stage indexes, for now)";

/// Which original ids a filtered search may return.
pub(crate) enum Filter<'a> {
    /// The same ids for every query: a mask over the collection.
    Shared(&'a [bool]),
    /// Ids per query, CSR-style: query `i` allows `ids[offsets[i]..offsets[i + 1]]`.
    PerQuery {
        ids: &'a [u64],
        offsets: &'a [usize],
    },
    /// One mask per query, row-major: query `i` allows the ids set in `masks[i * n..(i + 1) * n]`.
    PerQueryMasks { masks: &'a [bool], n: usize },
}

/// Filtered search for one index type: ACORN-1 and ACORN-γ on HNSW. Other index types report
/// `UNSUPPORTED`.
pub(crate) trait FilteredSearch: Index {
    /// Why the index has no filtered search, if it has none.
    const UNSUPPORTED: Option<&'static str> = None;

    /// The `k` nearest neighbors among `allowed`. With `gamma`, the ACORN-γ neighbor lists
    /// built by [`acorn_gamma_neighbors`](Self::acorn_gamma_neighbors).
    fn search_filtered_with<'q>(
        &'q self,
        query: Self::Query<'q>,
        k: usize,
        params: &Self::SearchParams,
        allowed: &Bitset,
        gamma: Option<&Graph>,
    ) -> Vec<ScoredVector<Self::Distance>>;

    /// ACORN-γ neighbor lists, for the indexes that use them.
    fn acorn_gamma_neighbors(&self, _gamma: usize) -> Option<Graph> {
        None
    }
}

impl<D, G> FilteredSearch for HNSW<D, G>
where
    D: Dataset + Sync + SpaceUsage,
    <D::Encoder as VectorEncoder>::Distance: Distance,
    G: GraphTrait,
{
    fn search_filtered_with<'q>(
        &'q self,
        query: Self::Query<'q>,
        k: usize,
        params: &Self::SearchParams,
        allowed: &Bitset,
        gamma: Option<&Graph>,
    ) -> Vec<ScoredVector<Self::Distance>> {
        // Both ACORN variants call the predicate with original ids, for any graph layout.
        let predicate = |id: usize| allowed.contains(id);
        match gamma {
            Some(lists) => self.search_filtered_gamma(query, k, params, lists, predicate),
            None => self.search_filtered(query, k, params, predicate),
        }
    }

    fn acorn_gamma_neighbors(&self, gamma: usize) -> Option<Graph> {
        Some(self.build_acorn_gamma_neighbors(gamma))
    }
}

impl<D> FilteredSearch for FlatIndex<D>
where
    D: Dataset,
    <D::Encoder as VectorEncoder>::Distance: Distance,
{
    const UNSUPPORTED: Option<&'static str> = Some(FILTER_UNSUPPORTED);

    fn search_filtered_with<'q>(
        &'q self,
        _: Self::Query<'q>,
        _: usize,
        _: &Self::SearchParams,
        _: &Bitset,
        _: Option<&Graph>,
    ) -> Vec<ScoredVector<Self::Distance>> {
        unreachable!("filtered search is rejected through UNSUPPORTED")
    }
}

impl<DQ, D, CI> FilteredSearch for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
    IVF<DQ, D, CI>: Index,
{
    const UNSUPPORTED: Option<&'static str> = Some(FILTER_UNSUPPORTED);

    fn search_filtered_with<'q>(
        &'q self,
        _: Self::Query<'q>,
        _: usize,
        _: &Self::SearchParams,
        _: &Bitset,
        _: Option<&Graph>,
    ) -> Vec<ScoredVector<Self::Distance>> {
        unreachable!("filtered search is rejected through UNSUPPORTED")
    }
}
