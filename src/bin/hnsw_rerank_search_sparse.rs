//! Two-stage sparse search over an index written by `hnsw_rerank_build_sparse` (or by
//! `SparseRerankHNSW.save` in Python).
//!
//! Index files carry no header, so `--first-stage-value-type`, `--graph-type` and
//! `--residual-rerank` must match the build.
//!
//! Example:
//! ```text
//! hnsw_rerank_search_sparse --index-file idx.bin --query-file queries.bin --residual-rerank \
//!   -k 10 --k-candidates 100 --ef-search 100 --output-path results.tsv
//! ```

use std::fs::File;
use std::io::Write;
use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use kannolo::hnsw::EarlyTerminationStrategy;
use kannolo::rerank::{GraphLayout, RerankSearchParams, SparseFirstStage, SparseRerankIndex};
use vectorium::dataset::ScoredVector;
use vectorium::distances::{Distance, DotProduct};
use vectorium::readers::read_seismic_format;
use vectorium::{Dataset, Index, PlainSparseDataset};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ValueTypeArg {
    F16,
    Fixedu8,
    #[value(name = "blocked-sparse")]
    BlockedSparse,
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
    /// The two-stage index written by `hnsw_rerank_build_sparse`.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The queries in Seismic binary format.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The output file to write the results (query_id, doc_id, rank, score).
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The index was built with `--residual-rerank`.
    #[clap(long, action)]
    residual_rerank: bool,

    #[clap(long, value_enum, default_value_t = ValueTypeArg::BlockedSparse)]
    first_stage_value_type: ValueTypeArg,

    /// `standard` and `permuted` load as the same type.
    #[clap(long, value_enum, default_value_t = GraphTypeArg::Standard)]
    graph_type: GraphTypeArg,

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

    let index = SparseRerankIndex::load(
        &args.index_file,
        match args.first_stage_value_type {
            ValueTypeArg::F16 => SparseFirstStage::F16,
            ValueTypeArg::Fixedu8 => SparseFirstStage::FixedU8,
            ValueTypeArg::BlockedSparse => SparseFirstStage::Blocked,
        },
        match args.graph_type {
            GraphTypeArg::Standard => GraphLayout::Standard,
            GraphTypeArg::Permuted => GraphLayout::Permuted,
            GraphTypeArg::Streamvbyte => GraphLayout::Compressed,
        },
        args.residual_rerank,
    )
    .unwrap_or_else(|e| exit_with(e));
    println!("[######] Space usage: {} bytes", index.space_usage_bytes());

    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| exit_with(format!("reading {}: {e:?}", args.query_file)));
    let num_queries = queries.len();

    let mut results: Vec<Vec<ScoredVector<DotProduct>>> = Vec::with_capacity(num_queries);
    let mut total_time_search = 0u128;
    for _ in 0..args.num_runs {
        results.clear();
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &params);
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

fn write_results(output_path: &str, results: &[Vec<ScoredVector<DotProduct>>]) {
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
