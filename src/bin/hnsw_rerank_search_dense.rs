//! Two-stage dense search over an index written by `hnsw_rerank_build_dense` (or by
//! `DenseRerankIndex::save` in Rust).
//!
//! Index files carry no header, so `--encoder`, `--pq-subspaces`, `--distance`, `--graph-type`,
//! `--residual-rerank` and whether `--rerank-bits` is zero must match the build.
//! `--rabitq-query-bits` may differ from the build.
//!
//! Example:
//! ```text
//! hnsw_rerank_search_dense --index-file idx.bin --query-file queries.npy \
//!   --encoder rabitq-ext --rerank-bits 8 --distance dotproduct --graph-type permuted \
//!   -k 10 --k-candidates 50 --ef-search 50 --output-path results.tsv
//! ```

use std::fs::File;
use std::io::Write;
use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use kannolo::hnsw::EarlyTerminationStrategy;
use kannolo::rerank::{DenseEncoder, DenseRerankIndex, GraphLayout, RerankSearchParams};
use vectorium::dataset::ScoredVector;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::readers::read_npy_f32;
use vectorium::vector::DenseVectorView;
use vectorium::{Dataset, Index, PlainDenseDataset};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum EncoderArg {
    Pq,
    Rabitq,
    RabitqExt,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum GraphTypeArg {
    Standard,
    Permuted,
    Streamvbyte,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum EarlyTerminationArg {
    None,
    DistanceAdaptive,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The two-stage index written by `hnsw_rerank_build_dense`.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The `.npy` query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The output file to write the results (query_id, doc_id, rank, score).
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The encoder the first stage was built with.
    #[clap(long, value_enum)]
    encoder: EncoderArg,

    /// PQ subspace count, for `--encoder pq`. Must match the index.
    #[clap(long, value_parser, default_value_t = 32)]
    pq_subspaces: usize,

    /// Query code width, for `--encoder rabitq` (1..=8).
    #[clap(long, value_parser, default_value_t = 1)]
    rabitq_query_bits: u32,

    /// The rerank stage's encoding used at build: 0 for plain f16, or 2, 4 or 8 for RaBitQ-ext.
    #[clap(long, value_parser, default_value_t = 0)]
    rerank_bits: u32,

    /// The index was built with `--residual-rerank`.
    #[clap(long, action)]
    residual_rerank: bool,

    /// `standard` and `permuted` load as the same type.
    #[clap(long, value_enum, default_value_t = GraphTypeArg::Standard)]
    graph_type: GraphTypeArg,

    /// `dotproduct` or `euclidean`.
    #[clap(long, value_parser, default_value_t = String::from("dotproduct"))]
    distance: String,

    /// Final number of results per query.
    #[clap(short, long, value_parser, default_value_t = 10)]
    k: usize,

    /// First-stage candidates to rerank.
    #[clap(long, value_parser, default_value_t = 100)]
    k_candidates: usize,

    /// First-stage candidate list size.
    #[clap(long, value_parser, default_value_t = 100)]
    ef_search: usize,

    /// Candidate pruning on the first-stage score. Omit for no pruning.
    #[clap(long, value_parser)]
    alpha: Option<f32>,

    /// Early exit during reranking. Omit to disable it.
    #[clap(long, value_parser)]
    beta: Option<usize>,

    /// First-stage early termination.
    #[clap(long, value_enum, default_value_t = EarlyTerminationArg::None)]
    early_termination: EarlyTerminationArg,

    /// Lambda for `--early-termination distance-adaptive`.
    #[clap(long, value_parser)]
    lambda: Option<f32>,

    /// Timed passes over the query set.
    #[clap(long, value_parser, default_value_t = 1)]
    num_runs: usize,
}

fn exit_with(msg: impl std::fmt::Display) -> ! {
    eprintln!("Error: {msg}");
    process::exit(1);
}

fn main() {
    let args = Args::parse();

    let graph = match args.graph_type {
        GraphTypeArg::Standard => GraphLayout::Standard,
        GraphTypeArg::Permuted => GraphLayout::Permuted,
        GraphTypeArg::Streamvbyte => GraphLayout::Compressed,
    };
    let encoder = match args.encoder {
        EncoderArg::Pq => DenseEncoder::Pq {
            subspaces: args.pq_subspaces,
        },
        EncoderArg::Rabitq => DenseEncoder::Rabitq {
            query_bits: args.rabitq_query_bits,
        },
        // `total_bits` is stored with the index.
        EncoderArg::RabitqExt => DenseEncoder::RabitqExt { total_bits: 0 },
    };
    let early_termination = match args.early_termination {
        EarlyTerminationArg::None => EarlyTerminationStrategy::None,
        EarlyTerminationArg::DistanceAdaptive => EarlyTerminationStrategy::DistanceAdaptive {
            lambda: args.lambda.unwrap_or_else(|| {
                exit_with("--early-termination distance-adaptive requires --lambda.")
            }),
        },
    };
    let params = RerankSearchParams {
        k_candidates: args.k_candidates,
        ef_search: args.ef_search,
        alpha: args.alpha,
        beta: args.beta,
        early_termination,
    };

    macro_rules! load {
        ($dist:ty) => {
            DenseRerankIndex::<$dist>::load(
                &args.index_file,
                graph,
                encoder,
                args.rerank_bits,
                args.residual_rerank,
            )
            .unwrap_or_else(|e| exit_with(e))
        };
    }
    match args.distance.as_str() {
        "dotproduct" | "ip" => run(load!(DotProduct), &args, &params),
        "euclidean" | "l2" => run(load!(SquaredEuclideanDistance), &args, &params),
        d => exit_with(format!(
            "invalid --distance {d}. Choose 'euclidean' or 'dotproduct'."
        )),
    }
}

/// The timed loop, shared by both metrics.
fn run<D: Distance>(index: DenseRerankIndex<D>, args: &Args, params: &RerankSearchParams) {
    println!("[######] Space usage: {} bytes", index.space_usage_bytes());

    let queries: PlainDenseDataset<f32, DotProduct> = read_npy_f32(&args.query_file)
        .unwrap_or_else(|e| exit_with(format!("reading {}: {e:?}", args.query_file)));
    if queries.output_dim() != index.dim() {
        exit_with(format!(
            "query dimension {} does not match index dimension {}",
            queries.output_dim(),
            index.dim()
        ));
    }
    let num_queries = queries.len();
    let dim = index.dim();
    let values = queries.values();

    let mut results: Vec<Vec<ScoredVector<D>>> = Vec::with_capacity(num_queries);
    let mut total_time_search = 0u128;
    for _ in 0..args.num_runs {
        results.clear();
        for q in 0..num_queries {
            let query = DenseVectorView::new(&values[q * dim..(q + 1) * dim]);
            let start_time = Instant::now();
            let res = index.search(query, args.k, params);
            total_time_search += start_time.elapsed().as_micros();
            results.push(res);
        }
    }

    if num_queries == 0 || args.num_runs == 0 {
        exit_with("no queries to time: the query file is empty or --num-runs is 0");
    }
    let n = (num_queries * args.num_runs) as u128;
    println!(
        "[######] Average Query Time: {} \u{3bc}s",
        total_time_search / n
    );
    println!(
        "[######] QPS: {:.2}",
        1_000_000.0 * n as f64 / total_time_search as f64
    );

    if let Some(output_path) = &args.output_path {
        write_results(output_path, &results);
    }
}

fn write_results<D: Distance>(output_path: &str, results: &[Vec<ScoredVector<D>>]) {
    let mut file = File::create(output_path).unwrap_or_else(|e| exit_with(e));
    for (query_id, res) in results.iter().enumerate() {
        for (rank, r) in res.iter().enumerate() {
            writeln!(
                file,
                "{}\t{}\t{}\t{}",
                query_id,
                r.vector,
                rank + 1,
                r.distance.distance()
            )
            .unwrap_or_else(|e| exit_with(e));
        }
    }
}
