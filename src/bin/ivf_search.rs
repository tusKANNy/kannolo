/// Search an IVF index over dense (`.npy`) or sparse (Seismic binary format) vectors.
use std::fs::File;
use std::io::Write as IoWrite;
use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;

use kannolo::graph::Graph;
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use kannolo::ivf::{IVF, IVFSearchParams};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::vector_encoder::VectorEncoder;
use vectorium::{
    Dataset, DenseDataset, Distance, FlatIndex, PlainDenseDataset, RabitqExtQuantizer,
    RabitqQuantizer, RabitqQueryParams, ScalarDenseDataset, ScalarSparseDataset, SpaceUsage,
};

/// Dense centroids are stored in f16, under the reported metric.
#[allow(non_camel_case_types)]
type DQ_SE = ScalarDenseDataset<f32, f16, SquaredEuclideanDistance>;
#[allow(non_camel_case_types)]
type DQ_DP = ScalarDenseDataset<f32, f16, DotProduct>;
type DenseQueries = PlainDenseDataset<f32, SquaredEuclideanDistance>;
/// Sparse queries and centroids: f32, u16 components, dot product.
type Sparse = vectorium::PlainSparseDataset<u16, f32, DotProduct>;
type DensePlainDP = PlainDenseDataset<f32, DotProduct>;

#[derive(Debug, Clone, ValueEnum, Default)]
enum DatasetType {
    #[default]
    Dense,
    Sparse,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum ValueTypeArg {
    #[default]
    F32,
    F16,
}

/// Cluster storage encoder. Dense: plain, pq (with --m-pq), rabitq, rabitq-ext. Sparse: plain.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum EncoderType {
    #[default]
    Plain,
    Pq,
    Rabitq,
    RabitqExt,
}

#[derive(Debug, Clone, ValueEnum)]
enum MetricArg {
    Euclidean,
    Dotproduct,
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    #[clap(short, long)]
    index_file: String,
    #[clap(short, long)]
    query_file: String,
    #[clap(short, long)]
    output_path: Option<String>,
    /// Dense vectors (`.npy`) or sparse vectors (Seismic binary format).
    #[clap(long, value_enum, default_value_t = DatasetType::Dense)]
    dataset_type: DatasetType,
    #[clap(short, long, default_value_t = 10)]
    k: usize,
    #[clap(long, default_value_t = 32)]
    n_probe: usize,
    /// Reported metric. Default: euclidean for dense, dotproduct for sparse (the only one
    /// sparse IVF supports).
    #[clap(long, value_enum)]
    distance: Option<MetricArg>,
    #[clap(long, value_enum, default_value_t = ValueTypeArg::F32)]
    value_type: ValueTypeArg,
    #[clap(long, default_value_t = false)]
    hnsw: bool,
    /// Cluster storage encoder, as built (dense only for anything but plain). `--m-pq` alone
    /// implies pq.
    #[clap(long, value_enum, default_value_t = EncoderType::Plain)]
    encoder: EncoderType,
    /// PQ subspaces (dense only).
    #[clap(long)]
    m_pq: Option<usize>,
    /// Bits per component of the quantized query, 1..=8 (rabitq only). Use the value the
    /// index was built with; 1 is only valid for plain dot-product indexes.
    #[clap(long, default_value_t = 4)]
    rabitq_query_bits: u32,
    #[clap(long, default_value_t = 40)]
    ef_search: usize,
    #[clap(long, default_value_t = 0.0)]
    lambda: f32,
    #[clap(long, default_value_t = 1)]
    num_runs: usize,
}

/// Dense: dispatch on the cluster storage. Residual-vs-plain is read from the index.
macro_rules! dense {
    ($dq:ty, $args:expr, $queries:expr) => {
        match ($args.encoder, $args.m_pq, &$args.value_type) {
            (EncoderType::Rabitq, _, _) => {
                centroid_index::<$dq, DenseDataset<RabitqQuantizer<DotProduct>>, _>($args, $queries)
            }
            (EncoderType::RabitqExt, _, _) => {
                centroid_index::<$dq, DenseDataset<RabitqExtQuantizer<DotProduct>>, _>(
                    $args, $queries,
                )
            }
            (_, None, ValueTypeArg::F32) => centroid_index::<$dq, DensePlainDP, _>($args, $queries),
            (_, None, ValueTypeArg::F16) => {
                centroid_index::<$dq, ScalarDenseDataset<f32, f16, DotProduct>, _>($args, $queries)
            }
            (_, Some(m), _) => dense!(@pq $dq, $args, $queries, m; 4, 8, 16, 32, 48, 64, 96, 128, 192),
        }
    };
    (@pq $dq:ty, $args:expr, $queries:expr, $m:expr; $($pq:literal),*) => {
        match $m {
            $($pq => centroid_index::<$dq, DenseDataset<ProductQuantizer<$pq, DotProduct>>, _>(
                $args, $queries,
            ),)*
            m => fail(&format!("unsupported --m-pq {m}")),
        }
    };
}

fn fail(msg: &str) -> ! {
    eprintln!("Error: {msg}");
    process::exit(1);
}

fn main() {
    let args = Args::parse();
    match (args.encoder, args.m_pq) {
        (EncoderType::Pq, None) => fail("--encoder pq needs --m-pq."),
        (EncoderType::Rabitq | EncoderType::RabitqExt, Some(_)) => {
            fail("--m-pq is for --encoder pq only.")
        }
        _ => {}
    }
    match args.dataset_type {
        DatasetType::Dense => {
            let queries: DenseQueries = read_npy_f32(&args.query_file)
                .unwrap_or_else(|e| fail(&format!("reading queries: {e:?}")));
            match args.distance.as_ref().unwrap_or(&MetricArg::Euclidean) {
                MetricArg::Euclidean => dense!(DQ_SE, &args, &queries),
                MetricArg::Dotproduct => dense!(DQ_DP, &args, &queries),
            }
        }
        DatasetType::Sparse => {
            if args.m_pq.is_some() || !matches!(args.encoder, EncoderType::Plain) {
                fail("sparse IVF supports --encoder plain only.");
            }
            if matches!(args.distance, Some(MetricArg::Euclidean)) {
                fail("sparse IVF reports dot product only.");
            }
            let queries: Sparse = read_seismic_format(&args.query_file)
                .unwrap_or_else(|e| fail(&format!("reading queries: {e:?}")));
            match args.value_type {
                ValueTypeArg::F32 => {
                    centroid_index::<Sparse, ScalarSparseDataset<u16, f32, f32, DotProduct>, _>(
                        &args, &queries,
                    )
                }
                ValueTypeArg::F16 => {
                    centroid_index::<Sparse, ScalarSparseDataset<u16, f32, f16, DotProduct>, _>(
                        &args, &queries,
                    )
                }
            }
        }
    }
}

/// Dispatch on the centroid index: flat or HNSW.
fn centroid_index<DQ, D, Q>(args: &Args, queries: &Q)
where
    DQ: Dataset + Sync + SpaceUsage,
    <DQ::Encoder as VectorEncoder>::Distance: Distance,
    D: Dataset,
    IVF<DQ, D, FlatIndex<DQ>>: Searchable<Q, CentroidParams = ()>,
    IVF<DQ, D, HNSW<DQ, Graph>>: Searchable<Q, CentroidParams = HNSWSearchConfiguration>,
{
    if args.hnsw {
        let cfg = HNSWSearchConfiguration::default()
            .with_ef_search(args.ef_search.max(args.n_probe))
            .with_early_termination(if args.lambda > 0.0 {
                EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: args.lambda,
                }
            } else {
                EarlyTerminationStrategy::None
            });
        IVF::<DQ, D, HNSW<DQ, Graph>>::run(args, queries, cfg);
    } else {
        IVF::<DQ, D, FlatIndex<DQ>>::run(args, queries, ());
    }
}

/// An IVF that can be loaded and searched with queries from the dataset `Q`.
trait Searchable<Q> {
    type CentroidParams;
    fn run(args: &Args, queries: &Q, centroid_search_params: Self::CentroidParams);
}

/// Query-side parameters of the cluster storage encoder, from the command line.
trait ClusterQueryParams {
    fn from_args(args: &Args) -> Self;
}

impl ClusterQueryParams for () {
    fn from_args(_args: &Args) -> Self {}
}

impl ClusterQueryParams for RabitqQueryParams {
    fn from_args(args: &Args) -> Self {
        RabitqQueryParams {
            query_bits: args.rabitq_query_bits,
        }
    }
}

impl<DQ, D, CI, Q> Searchable<Q> for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    <D::Encoder as VectorEncoder>::QueryParams: ClusterQueryParams,
    CI: Index,
    Q: Dataset,
    IVF<DQ, D, CI>: IndexSerializer
        + Index<SearchParams = IVFSearchParams<CI, <D::Encoder as VectorEncoder>::QueryParams>>
        + for<'de> serde::Deserialize<'de>,
    for<'q> <Q::Encoder as VectorEncoder>::EncodedVector<'q>:
        Into<<IVF<DQ, D, CI> as Index>::Query<'q>>,
{
    type CentroidParams = CI::SearchParams;

    fn run(args: &Args, queries: &Q, centroid_search_params: CI::SearchParams) {
        let index: Self = IndexSerializer::load_index(&args.index_file)
            .unwrap_or_else(|e| fail(&format!("loading index: {e:?}")));
        index.print_space_usage_bytes();
        // A 1-bit RaBitQ query carries a per-query scale, which breaks adding exact terms (the
        // L2 correction, a residual's centroid term) to its estimate.
        let reports_l2 = matches!(args.dataset_type, DatasetType::Dense)
            && !matches!(args.distance, Some(MetricArg::Dotproduct));
        if matches!(args.encoder, EncoderType::Rabitq)
            && args.rabitq_query_bits == 1
            && (reports_l2 || index.is_residual())
        {
            fail("--rabitq-query-bits 1 is only valid for plain dot-product RaBitQ indexes.");
        }
        let params = IVFSearchParams {
            nprobe: args.n_probe,
            centroid_search_params,
            cluster_query_params: ClusterQueryParams::from_args(args),
        };

        let nq = queries.len();
        let mut results = Vec::<(f32, usize)>::with_capacity(nq * args.k);
        let mut total_us = 0u128;
        for _ in 0..args.num_runs {
            results.clear();
            for query in queries.iter() {
                let t = Instant::now();
                let res = Index::search(&index, query.into(), args.k, &params);
                total_us += t.elapsed().as_micros();
                results.extend(
                    res.iter()
                        .map(|s| (s.distance.distance(), s.vector as usize)),
                );
            }
        }
        let avg = total_us / (nq * args.num_runs) as u128;
        println!("[######] Average Query Time: {avg} μs");
        if let Some(path) = &args.output_path {
            write_results(path, &results, args.k);
        }
    }
}

fn write_results(path: &str, results: &[(f32, usize)], k: usize) {
    let mut f = File::create(path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(f, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}
