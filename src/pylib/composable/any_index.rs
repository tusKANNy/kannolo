//! One common interface, [`AnyIndex`], for all the index types that [`super::build_load`]
//! creates.
//!
//! Rust compiles each combination of representation, encoder and index structure into a
//! different type, and the Python class must hold any of them in one field. Each first stage is
//! therefore wrapped in a [`SingleStage`] and stored as a `dyn AnyIndex`. Each rescoring dataset
//! is wrapped in a [`Rescorer`] and stored as a `dyn AnyRescorer`. A [`TwoStage`] pairs a first
//! stage with a rescorer and is itself an `AnyIndex`, so a cascade is a `TwoStage` whose first
//! stage is a `TwoStage`.
//!
//! Calls through the common interface happen once per batch of queries and per stage. Inside
//! them, the search, the rescoring loop and the distance functions are compiled for the concrete
//! types, so the inner loops run the same code as in a Rust program that names the types.
//! Composing stages at run time keeps the compiled code linear in the number of components.
//! Compiling every pair of stages would make it grow with the number of pairs, which is
//! quadratic in the number of components.

// In a build without one of the representations, the code only that representation uses is
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

use std::io::{Read, Write};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use pyo3::PyResult;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;

use vectorium::core::index::{Index, IndexStats};
use vectorium::core::vector_encoder::VectorEncoder;
use vectorium::distances::Distance;
use vectorium::{Dataset, FlatIndex, RabitqQueryParams, SpaceUsage, VectorId};

use crate::graph::{Graph, GraphTrait};
use crate::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use crate::ivf::{IVF, IVFSearchParams};
use crate::rerank_index::{rerank_candidates, rerank_candidates_with_early_exit};

fn value_err(msg: impl Into<String>) -> pyo3::PyErr {
    PyValueError::new_err(msg.into())
}

use super::filter::{ACORN_GAMMA_UNSUPPORTED, Bitset, FILTER_UNSUPPORTED, Filter, FilteredSearch};
use super::queries::{Queries, QueryFormat};

// ----------------------------------------------------------------------------- search params

/// Keyword arguments of `search`. Each index takes what applies to it and rejects the rest.
#[derive(Clone, Default)]
pub(crate) struct SearchArgs {
    pub ef_search: Option<usize>,
    pub early_exit_threshold: Option<f32>,
    pub query_bits: Option<u32>,
    pub nprobe: Option<usize>,
    /// Filtered search: use the ACORN-γ neighbor lists (`Some(true)`), ACORN-1 (`Some(false)`),
    /// or, with `None`, ACORN-γ if the lists have been built and ACORN-1 if not.
    pub acorn_gamma: Option<bool>,
    /// One entry per rescoring stage, in order.
    pub rerank: Vec<RerankArgs>,
}

#[derive(Clone, Default)]
pub(crate) struct RerankArgs {
    pub k_candidates: usize,
    /// Candidate pruning: drop candidates whose first-stage score is not within a factor
    /// `alpha` of the `k`-th one.
    pub alpha: Option<f32>,
    /// Early exit: stop rescoring after `beta` consecutive candidates fail to enter the top `k`.
    pub beta: Option<usize>,
    pub query_bits: Option<u32>,
}

/// Builds an index's (or an encoder's) search parameters from the Python keyword arguments.
pub(crate) trait FromSearchArgs: Sized {
    fn from_args(args: &SearchArgs) -> PyResult<Self>;
}

impl FromSearchArgs for () {
    fn from_args(args: &SearchArgs) -> PyResult<()> {
        if args.query_bits.is_some() {
            return Err(value_err(
                "query_bits applies only to stages with a RaBitQ(bits=1) encoder",
            ));
        }
        Ok(())
    }
}

impl FromSearchArgs for RabitqQueryParams {
    fn from_args(args: &SearchArgs) -> PyResult<Self> {
        let bits = args.query_bits.unwrap_or(1);
        if !(1..=8).contains(&bits) {
            return Err(value_err(format!(
                "query_bits must be in 1..=8, got {bits}"
            )));
        }
        Ok(RabitqQueryParams::new(bits))
    }
}

const DEFAULT_EF_SEARCH: usize = 100;
const DEFAULT_NPROBE: usize = 16;

impl<P: FromSearchArgs> FromSearchArgs for HNSWSearchConfiguration<P> {
    fn from_args(args: &SearchArgs) -> PyResult<Self> {
        let ef_search = args.ef_search.unwrap_or(DEFAULT_EF_SEARCH);
        if ef_search == 0 {
            return Err(value_err("ef_search must be at least 1"));
        }
        let early_termination = match args.early_exit_threshold {
            None => EarlyTerminationStrategy::None,
            Some(lambda) => EarlyTerminationStrategy::DistanceAdaptive { lambda },
        };
        Ok(HNSWSearchConfiguration {
            ef_search,
            early_termination,
            query_params: P::from_args(args)?,
        })
    }
}

/// `query_bits` goes to the cluster encoder; the centroid index sees the other arguments.
impl<CI, P> FromSearchArgs for IVFSearchParams<CI, P>
where
    CI: Index,
    CI::SearchParams: FromSearchArgs,
    P: FromSearchArgs,
{
    fn from_args(args: &SearchArgs) -> PyResult<Self> {
        let nprobe = args.nprobe.unwrap_or(DEFAULT_NPROBE);
        if nprobe == 0 {
            return Err(value_err("nprobe must be at least 1"));
        }
        let centroid_args = SearchArgs {
            query_bits: None,
            ..args.clone()
        };
        Ok(IVFSearchParams {
            nprobe,
            centroid_search_params: CI::SearchParams::from_args(&centroid_args)?,
            cluster_query_params: P::from_args(args)?,
        })
    }
}

/// Which keyword arguments an index uses. Search rejects the other arguments, so that none is
/// silently ignored.
#[derive(Clone, Copy, Default)]
pub(crate) struct Accepts {
    pub ef_search: bool,
    pub early_exit_threshold: bool,
    pub nprobe: bool,
}

impl Accepts {
    fn check(&self, args: &SearchArgs) -> PyResult<()> {
        if args.ef_search.is_some() && !self.ef_search {
            return Err(value_err(
                "ef_search applies to HNSW first stages and IVF with HNSW centroids",
            ));
        }
        if args.early_exit_threshold.is_some() && !self.early_exit_threshold {
            return Err(value_err(
                "early_exit_threshold applies to HNSW first stages and IVF with HNSW centroids",
            ));
        }
        if args.nprobe.is_some() && !self.nprobe {
            return Err(value_err("nprobe applies to IVF first stages"));
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------- index trait

/// Total bytes held by an index, under the library's own `SpaceUsage` accounting.
pub(crate) trait IndexSpaceUsage {
    fn index_space_usage_bytes(&self) -> usize;
}

impl<D, G> IndexSpaceUsage for HNSW<D, G>
where
    D: Dataset + SpaceUsage,
    G: GraphTrait,
{
    fn index_space_usage_bytes(&self) -> usize {
        self.space_usage_bytes()
    }
}

impl<D: Dataset + SpaceUsage> IndexSpaceUsage for FlatIndex<D> {
    fn index_space_usage_bytes(&self) -> usize {
        self.dataset().space_usage_bytes()
    }
}

impl<DQ, D, CI> IndexSpaceUsage for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset + SpaceUsage,
    CI: Index + IndexSpaceUsage,
{
    fn index_space_usage_bytes(&self) -> usize {
        self.centroid_index().index_space_usage_bytes()
            + self.cluster_dataset().space_usage_bytes()
            + self.metadata_space_usage_bytes()
    }
}

/// Flattened search results: `n_queries * k` distances and ids, padded with `inf` / `-1`.
pub(crate) struct SearchResults {
    pub n_queries: usize,
    pub k: usize,
    pub distances: Vec<f32>,
    pub ids: Vec<i64>,
}

/// What the Python class needs from any index, for any type parameters.
pub(crate) trait AnyIndex: Send + Sync {
    fn search(
        &self,
        queries: &Queries<'_>,
        k: usize,
        args: &SearchArgs,
        num_threads: usize,
    ) -> PyResult<SearchResults>;
    /// The `k` nearest neighbors that `filter` allows (ACORN on HNSW, an exact scan on Flat).
    fn search_filtered(
        &self,
        queries: &Queries<'_>,
        k: usize,
        args: &SearchArgs,
        filter: &Filter<'_>,
        num_threads: usize,
    ) -> PyResult<SearchResults>;
    /// Builds (or rebuilds) the ACORN-γ neighbor lists of the first stage.
    fn build_acorn_gamma(&self, gamma: usize) -> PyResult<()>;
    fn has_acorn_gamma(&self) -> bool;
    /// Whether a score `d` of this index is within relaxation `alpha` of `threshold`, under
    /// the index's own metric (candidate pruning of the stage after it).
    fn within_relaxation(&self, d: f32, threshold: f32, alpha: f32) -> bool;
    fn n_elements(&self) -> usize;
    fn dim(&self) -> usize;
    fn space_usage_bytes(&self) -> usize;
    fn write_payload(&self, writer: &mut dyn Write) -> Result<(), String>;
}

/// The bincode configuration `vectorium::IndexSerializer` uses, so payloads stay compatible.
fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
}

pub(crate) fn read_payload<I: DeserializeOwned>(mut reader: &mut dyn Read) -> Result<I, String> {
    bincode::serde::decode_from_std_read(&mut reader, bincode_config()).map_err(|e| format!("{e}"))
}

fn write_payload<I: Serialize>(value: &I, mut writer: &mut dyn Write) -> Result<(), String> {
    bincode::serde::encode_into_std_write(value, &mut writer, bincode_config())
        .map(|_| ())
        .map_err(|e| format!("{e}"))
}

/// Runs `one` over `0..n`, serially, on the global pool, or on a pool of `num_threads`.
fn run_queries<T: Send>(
    n: usize,
    num_threads: usize,
    one: impl Fn(usize) -> T + Sync + Send,
) -> PyResult<Vec<T>> {
    Ok(match num_threads {
        _ if n <= 1 => (0..n).map(one).collect(),
        1 => (0..n).map(one).collect(),
        0 => (0..n).into_par_iter().map(one).collect(),
        t => rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .map_err(|e| value_err(format!("cannot build a {t}-thread pool: {e}")))?
            .install(|| (0..n).into_par_iter().map(one).collect()),
    })
}

/// Packs per-query `(score, id)` lists into padded flat arrays.
fn pack(per_query: Vec<Vec<(f32, VectorId)>>, k: usize) -> SearchResults {
    let n_queries = per_query.len();
    let mut out = SearchResults {
        n_queries,
        k,
        distances: Vec::with_capacity(n_queries * k),
        ids: Vec::with_capacity(n_queries * k),
    };
    for results in per_query {
        let found = results.len().min(k);
        for (d, id) in results.into_iter().take(k) {
            out.distances.push(d);
            out.ids.push(id as i64);
        }
        out.distances
            .extend(std::iter::repeat_n(f32::INFINITY, k - found));
        out.ids.extend(std::iter::repeat_n(-1, k - found));
    }
    out
}

/// A concrete single-stage index `I` whose queries have layout `L`.
pub(crate) struct SingleStage<I, L> {
    index: I,
    accepts: Accepts,
    /// ACORN-γ neighbor lists, once built. Kept in memory only: `save` does not write them.
    acorn_gamma: RwLock<Option<Graph>>,
    _layout: PhantomData<fn() -> L>,
}

impl<I, L> SingleStage<I, L> {
    pub(crate) fn new(index: I, accepts: Accepts) -> Self {
        Self {
            index,
            accepts,
            acorn_gamma: RwLock::new(None),
            _layout: PhantomData,
        }
    }
}

impl<I, L> AnyIndex for SingleStage<I, L>
where
    L: QueryFormat,
    I: for<'q> Index<Query<'q> = L::View<'q>>
        + IndexStats
        + IndexSpaceUsage
        + FilteredSearch
        + Serialize
        + Send
        + Sync,
    I::Distance: From<f32>,
    I::SearchParams: FromSearchArgs + Sync,
{
    fn search(
        &self,
        queries: &Queries<'_>,
        k: usize,
        args: &SearchArgs,
        num_threads: usize,
    ) -> PyResult<SearchResults> {
        if k == 0 {
            return Err(value_err("k must be at least 1"));
        }
        self.accepts.check(args)?;
        let params = I::SearchParams::from_args(args)?;
        let converted = L::convert(queries, self.index.dim())?;
        let per_query = run_queries(L::len(&converted), num_threads, |i| {
            self.index
                .search(L::view(&converted, i), k, &params)
                .into_iter()
                .map(|s| (s.distance.distance(), s.vector))
                .collect::<Vec<_>>()
        })?;
        Ok(pack(per_query, k))
    }

    fn search_filtered(
        &self,
        queries: &Queries<'_>,
        k: usize,
        args: &SearchArgs,
        filter: &Filter<'_>,
        num_threads: usize,
    ) -> PyResult<SearchResults> {
        if let Some(reason) = I::UNSUPPORTED {
            return Err(value_err(reason));
        }
        if k == 0 {
            return Err(value_err("k must be at least 1"));
        }
        self.accepts.check(args)?;
        let params = I::SearchParams::from_args(args)?;
        let converted = L::convert(queries, self.index.dim())?;
        let n_queries = L::len(&converted);
        let n = self.index.n_elements();

        let lists = self.acorn_gamma.read().expect("ACORN-γ lock poisoned");
        let gamma = match args.acorn_gamma {
            Some(true) if lists.is_none() => {
                return Err(value_err(
                    "acorn_gamma=True needs the ACORN-γ neighbor lists: call \
                     build_acorn_gamma(gamma) first",
                ));
            }
            Some(false) => None,
            _ => lists.as_ref(),
        };

        let shared = match *filter {
            Filter::Shared(mask) => {
                if mask.len() != n {
                    return Err(value_err(format!(
                        "the filter mask has {} entries, but the index has {n} vectors",
                        mask.len()
                    )));
                }
                Some(Bitset::from_mask(mask))
            }
            Filter::PerQuery { offsets, .. } => {
                if offsets.len() != n_queries + 1 {
                    return Err(value_err(format!(
                        "{} per-query filters for {n_queries} queries",
                        offsets.len().saturating_sub(1)
                    )));
                }
                None
            }
            Filter::PerQueryMasks { masks, n: width } => {
                if width != n || masks.len() != n_queries * n {
                    return Err(value_err(format!(
                        "per-query masks must have shape ({n_queries}, {n}), got {} rows of {width}",
                        masks.len() / width.max(1)
                    )));
                }
                None
            }
        };

        let per_query = run_queries(n_queries, num_threads, |i| {
            let own;
            let allowed = match (&shared, filter) {
                (Some(bits), _) => bits,
                (None, Filter::PerQuery { ids, offsets }) => {
                    own = Bitset::from_ids(&ids[offsets[i]..offsets[i + 1]], n);
                    &own
                }
                (None, Filter::PerQueryMasks { masks, .. }) => {
                    own = Bitset::from_mask(&masks[i * n..(i + 1) * n]);
                    &own
                }
                (None, Filter::Shared(_)) => unreachable!("a shared filter is built above"),
            };
            self.index
                .search_filtered_with(L::view(&converted, i), k, &params, allowed, gamma)
                .into_iter()
                .map(|s| (s.distance.distance(), s.vector))
                .collect::<Vec<_>>()
        })?;
        Ok(pack(per_query, k))
    }

    fn build_acorn_gamma(&self, gamma: usize) -> PyResult<()> {
        if gamma == 0 {
            return Err(value_err("gamma must be at least 1"));
        }
        let lists = self
            .index
            .acorn_gamma_neighbors(gamma)
            .ok_or_else(|| value_err(ACORN_GAMMA_UNSUPPORTED))?;
        *self.acorn_gamma.write().expect("ACORN-γ lock poisoned") = Some(lists);
        Ok(())
    }

    fn has_acorn_gamma(&self) -> bool {
        self.acorn_gamma
            .read()
            .expect("ACORN-γ lock poisoned")
            .is_some()
    }

    fn within_relaxation(&self, d: f32, threshold: f32, alpha: f32) -> bool {
        I::Distance::from(d).is_within_relaxation(&I::Distance::from(threshold), alpha)
    }

    fn n_elements(&self) -> usize {
        self.index.n_elements()
    }

    fn dim(&self) -> usize {
        self.index.dim()
    }

    fn space_usage_bytes(&self) -> usize {
        self.index.index_space_usage_bytes()
    }

    fn write_payload(&self, writer: &mut dyn Write) -> Result<(), String> {
        write_payload(&self.index, writer)
    }
}

// ----------------------------------------------------------------------------- rescoring

/// The candidates of one query for a rescoring stage, best first.
pub(crate) struct Candidates {
    pub ids: Vec<VectorId>,
    /// First-stage scores, in the same order, when the stage scores residuals.
    pub first_stage_scores: Option<Vec<f32>>,
}

/// A rescoring dataset of any type.
pub(crate) trait AnyRescorer: Send + Sync {
    fn rerank(
        &self,
        queries: &Queries<'_>,
        candidates: &[Candidates],
        k: usize,
        args: &RerankArgs,
        num_threads: usize,
    ) -> PyResult<SearchResults>;
    fn within_relaxation(&self, d: f32, threshold: f32, alpha: f32) -> bool;
    fn space_usage_bytes(&self) -> usize;
    fn write_payload(&self, writer: &mut dyn Write) -> Result<(), String>;
}

/// A concrete rescoring dataset `RD` whose queries have layout `L`.
pub(crate) struct Rescorer<RD, L> {
    dataset: RD,
    _layout: PhantomData<fn() -> L>,
}

impl<RD, L> Rescorer<RD, L> {
    pub(crate) fn new(dataset: RD) -> Self {
        Self {
            dataset,
            _layout: PhantomData,
        }
    }
}

impl<RD, L> AnyRescorer for Rescorer<RD, L>
where
    L: QueryFormat,
    RD: Dataset + SpaceUsage + Serialize + Send + Sync,
    RD::Encoder: for<'q> VectorEncoder<QueryVector<'q> = L::View<'q>>,
    <RD::Encoder as VectorEncoder>::Distance: Distance + From<f32>,
    <RD::Encoder as VectorEncoder>::QueryParams: FromSearchArgs + Sync,
{
    fn rerank(
        &self,
        queries: &Queries<'_>,
        candidates: &[Candidates],
        k: usize,
        args: &RerankArgs,
        num_threads: usize,
    ) -> PyResult<SearchResults> {
        let params = <<RD::Encoder as VectorEncoder>::QueryParams>::from_args(&SearchArgs {
            query_bits: args.query_bits,
            ..SearchArgs::default()
        })?;
        let converted = L::convert(queries, self.dataset.input_dim())?;
        if L::len(&converted) != candidates.len() {
            return Err(value_err(format!(
                "{} rescoring queries for {} first-stage queries",
                L::len(&converted),
                candidates.len()
            )));
        }
        let per_query = run_queries(candidates.len(), num_threads, |i| {
            let c = &candidates[i];
            let scores = c.first_stage_scores.as_deref();
            let view = L::view(&converted, i);
            let results = match args.beta {
                Some(beta) => rerank_candidates_with_early_exit(
                    &self.dataset,
                    view,
                    &params,
                    &c.ids,
                    k,
                    beta,
                    scores,
                ),
                None => rerank_candidates(&self.dataset, view, &params, &c.ids, k, scores),
            };
            results
                .into_iter()
                .map(|s| (s.distance.distance(), s.vector))
                .collect::<Vec<_>>()
        })?;
        Ok(pack(per_query, k))
    }

    fn within_relaxation(&self, d: f32, threshold: f32, alpha: f32) -> bool {
        type D<RD> = <<RD as Dataset>::Encoder as VectorEncoder>::Distance;
        D::<RD>::from(d).is_within_relaxation(&D::<RD>::from(threshold), alpha)
    }

    fn space_usage_bytes(&self) -> usize {
        self.dataset.space_usage_bytes()
    }

    fn write_payload(&self, writer: &mut dyn Write) -> Result<(), String> {
        write_payload(&self.dataset, writer)
    }
}

/// A first stage (any `AnyIndex`, including another `TwoStage`) followed by a rescoring
/// stage: the semantics of `kannolo::rerank_index::RerankIndex::search`, with the two stages
/// chosen at run time.
///
/// The first stage is shared (`Arc`): one built or loaded index can be the first stage of
/// several two-stage indexes without being copied.
pub(crate) struct TwoStage {
    pub first: Arc<dyn AnyIndex>,
    pub rerank: Box<dyn AnyRescorer>,
    /// Position of this rescoring stage (0 for the first one), which selects its arguments.
    pub level: usize,
    pub residual: bool,
}

impl TwoStage {
    /// Runs `first_stage` for the candidates, then rescores them.
    fn search_with(
        &self,
        queries: &Queries<'_>,
        k: usize,
        args: &SearchArgs,
        num_threads: usize,
        first_stage: impl FnOnce(usize) -> PyResult<SearchResults>,
    ) -> PyResult<SearchResults> {
        if k == 0 {
            return Err(value_err("k must be at least 1"));
        }
        // The number of per-stage argument sets is checked once, against the whole index, by
        // the Python class: an inner stage of a cascade sees the arguments of the outer ones.
        let rerank_args = args.rerank.get(self.level).cloned().unwrap_or_default();
        let k_candidates = if rerank_args.k_candidates == 0 {
            (10 * k).max(100)
        } else {
            rerank_args.k_candidates
        };
        if k_candidates < k {
            return Err(value_err(format!(
                "k_candidates ({k_candidates}) must be at least k ({k})"
            )));
        }
        if rerank_args.alpha.is_some_and(|a| !(0.0..=1.0).contains(&a)) {
            return Err(value_err("alpha must be in [0, 1]"));
        }
        if rerank_args.beta == Some(0) {
            return Err(value_err("beta must be at least 1"));
        }

        let first = first_stage(k_candidates)?;

        // Candidate lists, with candidate pruning applied. Whether a candidate is within `alpha`
        // of the k-th one follows the ordering and the metric of the first stage.
        let kc = first.k;
        let candidates: Vec<Candidates> = (0..first.n_queries)
            .map(|q| {
                let ids = &first.ids[q * kc..(q + 1) * kc];
                let dists = &first.distances[q * kc..(q + 1) * kc];
                let found = ids.iter().take_while(|&&id| id >= 0).count();
                let threshold = match rerank_args.alpha {
                    Some(alpha) if k <= found => Some((dists[k - 1], alpha)),
                    _ => None,
                };
                let keep: Vec<usize> = (0..found)
                    .filter(|&j| {
                        threshold.is_none_or(|(t, a)| self.first.within_relaxation(dists[j], t, a))
                    })
                    .collect();
                Candidates {
                    ids: keep.iter().map(|&j| ids[j] as VectorId).collect(),
                    first_stage_scores: self
                        .residual
                        .then(|| keep.iter().map(|&j| dists[j]).collect()),
                }
            })
            .collect();

        self.rerank
            .rerank(queries, &candidates, k, &rerank_args, num_threads)
    }
}

impl AnyIndex for TwoStage {
    fn search(
        &self,
        queries: &Queries<'_>,
        k: usize,
        args: &SearchArgs,
        num_threads: usize,
    ) -> PyResult<SearchResults> {
        self.search_with(queries, k, args, num_threads, |kc| {
            self.first.search(queries, kc, args, num_threads)
        })
    }

    fn search_filtered(
        &self,
        _: &Queries<'_>,
        _: usize,
        _: &SearchArgs,
        _: &Filter<'_>,
        _: usize,
    ) -> PyResult<SearchResults> {
        Err(value_err(FILTER_UNSUPPORTED))
    }

    fn build_acorn_gamma(&self, _: usize) -> PyResult<()> {
        Err(value_err(ACORN_GAMMA_UNSUPPORTED))
    }

    fn has_acorn_gamma(&self) -> bool {
        false
    }

    fn within_relaxation(&self, d: f32, threshold: f32, alpha: f32) -> bool {
        self.rerank.within_relaxation(d, threshold, alpha)
    }

    fn n_elements(&self) -> usize {
        self.first.n_elements()
    }

    fn dim(&self) -> usize {
        self.first.dim()
    }

    fn space_usage_bytes(&self) -> usize {
        self.first.space_usage_bytes() + self.rerank.space_usage_bytes()
    }

    fn write_payload(&self, writer: &mut dyn Write) -> Result<(), String> {
        self.first.write_payload(writer)?;
        self.rerank.write_payload(writer)
    }
}
