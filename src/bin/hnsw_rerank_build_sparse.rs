//! Builds a sparse two-stage index ([`kannolo::rerank::SparseRerankIndex`]) from a collection in
//! Seismic binary format and saves it to one file, for `hnsw_rerank_search_sparse`.
//!
//! - `--prune-percentage < 100` L1-prunes the first stage to the components carrying the top
//!   `prune_percentage`% of each vector's L1 mass.
//! - `--first-stage-value-type` stores the first stage as `blocked-sparse` (the default), `f16` or
//!   `fixedu8`. With `blocked-sparse` both stages are stored blocked and the graph is built over
//!   the `f16` values; otherwise the rerank stage is `f16`. The `streamvbyte` graph needs `f16` or
//!   `fixedu8`.
//! - `--residual-rerank` stores only the pruned-away components in the rerank stage and sums the
//!   two scores; with an `f16` first stage this is the exact score at `f16` precision. Without it the rerank stage
//!   holds the full vectors.
//!
//! Example (L1-pruned first stage, exact residual rescoring):
//! ```text
//! hnsw_rerank_build_sparse --data-file documents.bin --output-file idx.bin \
//!   --prune-percentage 60 --residual-rerank --m 32 --ef-construction 1000
//! ```

use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use kannolo::rerank::{GraphLayout, SparseFirstStage, SparseRerankConfig, SparseRerankIndex};
use vectorium::PlainSparseDataset;
use vectorium::distances::DotProduct;
use vectorium::readers::read_seismic_format;

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

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The collection in Seismic binary format.
    #[clap(short, long, value_parser)]
    data_file: String,

    /// Where to write the two-stage index.
    #[clap(short, long, value_parser)]
    output_file: String,

    /// In [0, 100]. 100 means no pruning.
    #[clap(long, value_parser)]
    prune_percentage: f32,

    /// Store only the pruned-away components in the rerank stage and sum the two scores. Needs
    /// `--prune-percentage` below 100.
    #[clap(long, action)]
    residual_rerank: bool,

    #[clap(long, value_enum, default_value_t = ValueTypeArg::BlockedSparse)]
    first_stage_value_type: ValueTypeArg,

    #[clap(long, value_enum, default_value_t = GraphTypeArg::Standard)]
    graph_type: GraphTypeArg,

    #[clap(long, value_parser, default_value_t = 32)]
    m: usize,

    #[clap(long, value_parser, default_value_t = 200)]
    ef_construction: usize,
}

fn main() {
    let args = Args::parse();

    let cfg = SparseRerankConfig {
        m: args.m,
        ef_construction: args.ef_construction,
        graph: match args.graph_type {
            GraphTypeArg::Standard => GraphLayout::Standard,
            GraphTypeArg::Permuted => GraphLayout::Permuted,
            GraphTypeArg::Streamvbyte => GraphLayout::Compressed,
        },
        first_stage: match args.first_stage_value_type {
            ValueTypeArg::F16 => SparseFirstStage::F16,
            ValueTypeArg::Fixedu8 => SparseFirstStage::FixedU8,
            ValueTypeArg::BlockedSparse => SparseFirstStage::Blocked,
        },
        prune_percentage: args.prune_percentage,
        residual_rerank: args.residual_rerank,
    };

    let collection: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.data_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading {}: {e:?}", args.data_file);
            process::exit(1);
        });

    let start_time = Instant::now();
    let index = SparseRerankIndex::build(collection, &cfg).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });
    println!(
        "Time to build: {} s (before serializing)",
        start_time.elapsed().as_secs()
    );
    println!("[######] Space usage: {} bytes", index.space_usage_bytes());

    index.save(&args.output_file).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });
}
