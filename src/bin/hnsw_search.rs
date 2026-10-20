use std::io::Write;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;
use std::fs::File;

use kannolo::graph::GraphFixedDegree;
use kannolo::graph::graph::{Graph, GraphTrait};
use kannolo::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::vector::SparseVectorView;
use vectorium::vector_encoder::VectorEncoder;
#[cfg(target_arch = "x86_64")]
use vectorium::{
    BlockedSparseDataset, BlockedSparseScalarU8Dataset, BlockedSparseScalarU16Dataset,
};
use vectorium::{
    BridgedDotPacking8U32ScalarU8Dataset, BridgedDotPacking8U32ScalarU16Dataset, Dataset,
    DenseDataset, DotPacking8U16F16Dataset, DotPacking8U16FixedU8Dataset,
    DotPacking8U16Scalar4BitDataset, DotPacking8U16ScalarU8Dataset, DotPacking8U16ScalarU16Dataset,
    FixedU8Q, FixedU16Q, PlainDenseDataset, PlainSparseDataset, RabitqExtQuantizer,
    RabitqQuantizer, RabitqQueryParams, RabitqSupportedDistance, ScalarSparseDataset,
    ScalarU8SparseDataset, ScalarU16SparseDataset, SpaceUsage,
};

#[derive(Debug, Clone, ValueEnum)]
enum DatasetType {
    Dense,
    Sparse,
}

/// Value type for stored values.
/// Dense plain: `f32`, `f16`. Sparse: every variant (see `docs/Binaries.md`).
/// Ignored for `pq`, `rabitq` and `rabitq-ext` (they fix their own value representation).
#[derive(Debug, Clone, ValueEnum, Default)]
enum ValueTypeArg {
    F16,
    #[default]
    F32,
    Fixedu8,
    Fixedu16,
    /// 8-bit codes with one trained quantization step per component.
    Scalaru8,
    /// 16-bit codes with one trained quantization step per component.
    Scalaru16,
    /// 64-byte blocks of 16 `u16` components with `f16` values (x86_64 only).
    #[cfg(target_arch = "x86_64")]
    #[value(name = "blocked-sparse")]
    BlockedSparse,
    #[cfg(target_arch = "x86_64")]
    #[value(name = "blocked-sparse-scalaru8")]
    BlockedSparseScalaru8,
    #[cfg(target_arch = "x86_64")]
    #[value(name = "blocked-sparse-scalaru16")]
    BlockedSparseScalaru16,
    /// DotPacking8 with fixed-point 8-bit values: the replacement of the removed `dotvbyte`.
    #[value(name = "dotpacking8-fixedu8")]
    Dotpacking8Fixedu8,
    /// DotPacking8 with per-component 8-bit codes (bridged encoding for `u32` components).
    #[value(name = "dotpacking8-scalaru8")]
    Dotpacking8Scalaru8,
    /// DotPacking8 with per-component 16-bit codes (bridged encoding for `u32` components).
    #[value(name = "dotpacking8-scalaru16")]
    Dotpacking8Scalaru16,
    #[value(name = "dotpacking8-f16")]
    Dotpacking8F16,
    #[value(name = "dotpacking8-scalar4bit")]
    Dotpacking8Scalar4bit,
}

impl ValueTypeArg {
    /// Whether dense plain datasets can store this value type.
    fn is_dense(&self) -> bool {
        matches!(self, ValueTypeArg::F32 | ValueTypeArg::F16)
    }

    /// Packed layouts (DotPacking8, blocked) score with dot product only.
    fn is_dot_product_only(&self) -> bool {
        self.is_packed()
    }

    /// Packed layouts with no `u32` encoding.
    fn is_u16_only(&self) -> bool {
        self.is_packed()
            && !matches!(
                self,
                ValueTypeArg::Dotpacking8Scalaru8 | ValueTypeArg::Dotpacking8Scalaru16
            )
    }

    fn is_packed(&self) -> bool {
        !matches!(
            self,
            ValueTypeArg::F16
                | ValueTypeArg::F32
                | ValueTypeArg::Fixedu8
                | ValueTypeArg::Fixedu16
                | ValueTypeArg::Scalaru8
                | ValueTypeArg::Scalaru16
        )
    }
}

impl std::fmt::Display for ValueTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self
            .to_possible_value()
            .expect("no value type is skipped by clap");
        write!(f, "{}", value.get_name())
    }
}

/// `dotvbyte` was removed in vectorium 0.11. Name its replacement instead of letting clap reject
/// the value with its generic "invalid value" message.
fn reject_removed_names() {
    if std::env::args()
        .skip(1)
        .any(|a| a == "dotvbyte" || a.ends_with("=dotvbyte"))
    {
        eprintln!(
            "Error: 'dotvbyte': DotVByte was removed in vectorium 0.11; use '--value-type dotpacking8-fixedu8'."
        );
        std::process::exit(2);
    }
}

/// Encoder type.
/// Dense: `plain`, `pq`, `rabitq`, `rabitq-ext`.
/// Sparse: `plain` (`--value-type` selects the representation).
#[derive(Debug, Clone, ValueEnum, Default)]
enum EncoderType {
    #[default]
    Plain,
    Pq,
    /// RaBitQ: 1-bit-per-component document codes, `--query-bits` per query component.
    Rabitq,
    /// Extended RaBitQ: multi-bit document codes. The code width is stored in the index.
    RabitqExt,
}

impl std::fmt::Display for EncoderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderType::Plain => write!(f, "plain"),
            EncoderType::Pq => write!(f, "pq"),
            EncoderType::Rabitq => write!(f, "rabitq"),
            EncoderType::RabitqExt => write!(f, "rabitq-ext"),
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum GraphType {
    /// Plain neighbor storage, original (insertion) node order.
    Standard,
    /// Fixed-degree (padded) neighbor storage, original node order.
    FixedDegree,
    /// Plain neighbor storage, nodes reordered by Enhanced Graph Bisection (EGB). Loads as the
    /// same Rust type as `standard`: the permutation lives in the serialized index data.
    Permuted,
    /// EGB-reordered nodes, neighbor lists delta-encoded and compressed with StreamVByte.
    Streamvbyte,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum ComponentTypeArg {
    #[default]
    U16,
    U32,
}

#[derive(Debug, Clone, ValueEnum)]
enum EarlyTerminationMethod {
    None,
    DistanceAdaptive,
}

#[derive(Clone, Copy, Debug)]
enum DistanceKind {
    Euclidean,
    DotProduct,
}

trait GraphBound: GraphTrait + for<'de> serde::Deserialize<'de> {}
impl<T> GraphBound for T where T: GraphTrait + for<'de> serde::Deserialize<'de> {}

fn parse_metric(metric: &str) -> DistanceKind {
    match metric {
        "euclidean" | "l2" => DistanceKind::Euclidean,
        "dotproduct" | "ip" => DistanceKind::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'euclidean' and 'dotproduct'.");
            std::process::exit(1);
        }
    }
}

fn read_npy_queries<D>(path: &str) -> PlainDenseDataset<f32, D>
where
    D: ScalarDenseSupportedDistance,
{
    read_npy_f32::<D>(path).unwrap_or_else(|e| {
        eprintln!("Error reading .npy file: {e:?}");
        std::process::exit(1);
    })
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the index.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The type of vectors (dense or sparse).
    #[clap(long, value_enum)]
    dataset_type: DatasetType,

    /// Value type for stored values. Dense plain: f32, f16. Sparse: f32, f16, fixedu8, fixedu16,
    /// scalaru8, scalaru16, blocked-sparse[-scalaru8|-scalaru16], dotpacking8-{fixedu8, scalaru8,
    /// scalaru16, f16, scalar4bit}. Ignored for pq and rabitq. Must match the build.
    #[clap(long = "value-type", value_enum)]
    #[arg(default_value_t = ValueTypeArg::F32)]
    value_type: ValueTypeArg,

    /// Component type for sparse datasets (`u16` or `u32`).
    /// Blocked and DotPacking8 value types support only `u16`, except dotpacking8-scalaru8 and
    /// dotpacking8-scalaru16.
    #[clap(long = "component-type", value_enum)]
    #[arg(default_value_t = ComponentTypeArg::U16)]
    component_type: ComponentTypeArg,

    /// Encoder type. Dense: plain, pq, rabitq, rabitq-ext. Sparse: plain.
    #[clap(long, value_enum)]
    #[arg(default_value_t = EncoderType::Plain)]
    encoder: EncoderType,

    /// The graph type: `standard`, `fixed-degree`, `permuted`, or `streamvbyte`.
    /// Must match the `--graph-type` used at build time.
    #[clap(long, value_enum)]
    #[arg(default_value_t = GraphType::Standard)]
    graph_type: GraphType,

    /// The distance metric ("euclidean" or "dotproduct").
    #[clap(long, value_parser)]
    distance: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    pq_subspaces: usize,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The ef_search parameter.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    ef_search: usize,

    /// Early termination strategy for search.
    #[clap(long, value_enum)]
    #[arg(default_value_t = EarlyTerminationMethod::None)]
    early_termination: EarlyTerminationMethod,

    /// Lambda parameter for DistanceAdaptive strategy.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1.0)]
    lambda: f32,

    /// Number of runs for timing.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    num_runs: usize,

    /// Bits per component used to scalar-quantize the *query* for `--encoder rabitq`
    /// (1..=8; 1 is a plain sign query). This is a query-side knob only: it is not baked
    /// into the index, so one index serves every setting. Ignored by other encoders,
    /// including rabitq-ext, whose query width follows the stored code width.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    query_bits: u32,
}

fn main() {
    reject_removed_names();
    let args: Args = Args::parse();
    let metric = parse_metric(&args.distance);

    // Cross-validation of encoder / dataset-type / value-type combinations
    match (&args.dataset_type, &args.encoder) {
        (DatasetType::Sparse, EncoderType::Pq) => {
            eprintln!("Error: PQ encoder is only available for dense vectors.");
            std::process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::Rabitq | EncoderType::RabitqExt) => {
            eprintln!("Error: RaBitQ encoders are only available for dense vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Plain) if !args.value_type.is_dense() => {
            eprintln!(
                "Error: value type {} is only available for sparse vectors.",
                args.value_type
            );
            std::process::exit(1);
        }
        (DatasetType::Dense, _) if !matches!(args.component_type, ComponentTypeArg::U16) => {
            eprintln!("Error: component-type is only applicable to sparse datasets.");
            std::process::exit(1);
        }
        (DatasetType::Sparse, _)
            if args.value_type.is_u16_only()
                && !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!(
                "Error: value type {} supports only component-type u16.",
                args.value_type
            );
            std::process::exit(1);
        }
        (DatasetType::Sparse, _)
            if args.value_type.is_dot_product_only()
                && !matches!(metric, DistanceKind::DotProduct) =>
        {
            eprintln!(
                "Error: value type {} supports only dotproduct distance.",
                args.value_type
            );
            std::process::exit(1);
        }
        _ => {}
    }

    match (
        &args.dataset_type,
        &args.encoder,
        &args.value_type,
        &args.graph_type,
    ) {
        // Dense plain f32
        (
            DatasetType::Dense,
            EncoderType::Plain,
            ValueTypeArg::F32,
            GraphType::Standard | GraphType::Permuted,
        ) => {
            search_dense_plain_f32::<Graph<PlainNeighbors>>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::FixedDegree) => {
            search_dense_plain_f32::<GraphFixedDegree>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Streamvbyte) => {
            search_dense_plain_f32::<Graph<StreamVByteNeighbors>>(&args, metric);
        }
        // Dense plain f16
        (
            DatasetType::Dense,
            EncoderType::Plain,
            ValueTypeArg::F16,
            GraphType::Standard | GraphType::Permuted,
        ) => {
            search_dense_plain_f16::<Graph<PlainNeighbors>>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::FixedDegree) => {
            search_dense_plain_f16::<GraphFixedDegree>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Streamvbyte) => {
            search_dense_plain_f16::<Graph<StreamVByteNeighbors>>(&args, metric);
        }
        // Dense PQ (value-type ignored)
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Standard | GraphType::Permuted) => {
            search_dense_pq::<Graph<PlainNeighbors>>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::FixedDegree) => {
            search_dense_pq::<GraphFixedDegree>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Streamvbyte) => {
            search_dense_pq::<Graph<StreamVByteNeighbors>>(&args, metric);
        }
        // Dense RaBitQ
        (DatasetType::Dense, EncoderType::Rabitq, _, GraphType::Standard | GraphType::Permuted) => {
            search_dense_rabitq::<Graph<PlainNeighbors>>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Rabitq, _, GraphType::FixedDegree) => {
            search_dense_rabitq::<GraphFixedDegree>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Rabitq, _, GraphType::Streamvbyte) => {
            search_dense_rabitq::<Graph<StreamVByteNeighbors>>(&args, metric);
        }
        // Dense RaBitQ-ext
        (
            DatasetType::Dense,
            EncoderType::RabitqExt,
            _,
            GraphType::Standard | GraphType::Permuted,
        ) => {
            search_dense_rabitq_ext::<Graph<PlainNeighbors>>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::RabitqExt, _, GraphType::FixedDegree) => {
            search_dense_rabitq_ext::<GraphFixedDegree>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::RabitqExt, _, GraphType::Streamvbyte) => {
            search_dense_rabitq_ext::<Graph<StreamVByteNeighbors>>(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Plain, _, _) => search_sparse(&args, metric),
        // Unreachable: caught by earlier validation
        (DatasetType::Sparse, EncoderType::Pq, _, _)
        | (DatasetType::Sparse, EncoderType::Rabitq | EncoderType::RabitqExt, _, _)
        | (DatasetType::Dense, EncoderType::Plain, _, _) => {
            unreachable!()
        }
    }
}

// ---------------------------------------------------------------------------
// RaBitQ
// ---------------------------------------------------------------------------

fn search_dense_rabitq<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_rabitq_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_rabitq_with_distance::<DotProduct, G>(args),
    }
}

/// The only search path that passes non-unit query parameters: `--query-bits` reaches
/// `RabitqQuantizer::query_evaluator` through `HNSWSearchConfiguration::query_params`.
fn search_dense_rabitq_with_distance<D, G>(args: &Args)
where
    D: RabitqSupportedDistance + ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    if !(1..=8).contains(&args.query_bits) {
        eprintln!(
            "Error: --query-bits must be in 1..=8, got {}.",
            args.query_bits
        );
        std::process::exit(1);
    }

    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();
    let index: HNSW<DenseDataset<RabitqQuantizer<D>>, G> =
        <HNSW<DenseDataset<RabitqQuantizer<D>>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();
    let config = create_search_config_with_params(args, RabitqQueryParams::new(args.query_bits));

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} \u{3bc}s");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_rabitq_ext<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_rabitq_ext_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_rabitq_ext_with_distance::<DotProduct, G>(args),
    }
}

/// RaBitQ-ext takes no query-side parameters: the query width follows the document code
/// width, which is baked into the stored encoder.
fn search_dense_rabitq_ext_with_distance<D, G>(args: &Args)
where
    D: RabitqSupportedDistance + ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();
    let index: HNSW<DenseDataset<RabitqExtQuantizer<D>>, G> =
        <HNSW<DenseDataset<RabitqExtQuantizer<D>>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} \u{3bc}s");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn create_search_config(args: &Args) -> HNSWSearchConfiguration {
    create_search_config_with_params(args, ())
}

/// The same graph-side configuration, carrying `query_params` for encoders that take
/// query-side configuration. The index's `SearchParams` pins `P` to its encoder's
/// `QueryParams`, so a mismatched setting is a compile error rather than a silent no-op.
fn create_search_config_with_params<P>(args: &Args, query_params: P) -> HNSWSearchConfiguration<P> {
    let early_termination = match args.early_termination {
        EarlyTerminationMethod::None => EarlyTerminationStrategy::None,
        EarlyTerminationMethod::DistanceAdaptive => EarlyTerminationStrategy::DistanceAdaptive {
            lambda: args.lambda,
        },
    };

    // Built by hand rather than from `default()`: `P` is not known to be `Default` here,
    // and `query_params` supplies it anyway.
    HNSWSearchConfiguration {
        ef_search: args.ef_search,
        early_termination,
        query_params,
    }
}

fn search_dense_plain_f32<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_plain_f32_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_plain_f32_with_distance::<DotProduct, G>(args),
    }
}

fn search_dense_plain_f32_with_distance<D, G>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();
    let index: HNSW<DenseDataset<PlainDenseQuantizer<f32, D>>, G> =
        <HNSW<DenseDataset<PlainDenseQuantizer<f32, D>>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_plain_f16<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_plain_f16_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_plain_f16_with_distance::<DotProduct, G>(args),
    }
}

fn search_dense_plain_f16_with_distance<D, G>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();
    let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, D>>, G> =
        <HNSW<DenseDataset<PlainDenseQuantizer<f16, D>>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_pq<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_pq_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_pq_with_distance::<DotProduct, G>(args),
    }
}

fn search_dense_pq_with_distance<D, G>(args: &Args)
where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();

    let config = create_search_config(args);

    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    match args.pq_subspaces {
        4 => {
            let index: HNSW<DenseDataset<ProductQuantizer<4, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<4, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        8 => {
            let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<8, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        16 => {
            let index: HNSW<DenseDataset<ProductQuantizer<16, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<16, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        24 => {
            let index: HNSW<DenseDataset<ProductQuantizer<24, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<24, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        32 => {
            let index: HNSW<DenseDataset<ProductQuantizer<32, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<32, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        48 => {
            let index: HNSW<DenseDataset<ProductQuantizer<48, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<48, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        64 => {
            let index: HNSW<DenseDataset<ProductQuantizer<64, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<64, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        96 => {
            let index: HNSW<DenseDataset<ProductQuantizer<96, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<96, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        128 => {
            let index: HNSW<DenseDataset<ProductQuantizer<128, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<128, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        192 => {
            let index: HNSW<DenseDataset<ProductQuantizer<192, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<192, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        256 => {
            let index: HNSW<DenseDataset<ProductQuantizer<256, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<256, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        384 => {
            let index: HNSW<DenseDataset<ProductQuantizer<384, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<384, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        _ => {
            eprintln!(
                "Error: Invalid pq-subspaces value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384."
            );
            std::process::exit(1);
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

/// Runs `$body` with `$c` and `$d` bound, as type aliases, to the `--component-type` and the
/// distance, so that one generic call covers the four (component type, distance) pairs.
macro_rules! with_component_and_distance {
    ($args:expr, $distance:expr, |$c:ident, $d:ident| $body:expr) => {
        match (&$args.component_type, $distance) {
            (ComponentTypeArg::U16, DistanceKind::DotProduct) => {
                type $c = u16;
                type $d = DotProduct;
                $body
            }
            (ComponentTypeArg::U16, DistanceKind::Euclidean) => {
                type $c = u16;
                type $d = SquaredEuclideanDistance;
                $body
            }
            (ComponentTypeArg::U32, DistanceKind::DotProduct) => {
                type $c = u32;
                type $d = DotProduct;
                $body
            }
            (ComponentTypeArg::U32, DistanceKind::Euclidean) => {
                type $c = u32;
                type $d = SquaredEuclideanDistance;
                $body
            }
        }
    };
}

/// Sparse datasets: picks the stored dataset type from `--value-type`, `--component-type` and the
/// distance. Value types that are dot-product-only or `u16`-only were validated in `main`.
fn search_sparse(args: &Args, metric: DistanceKind) {
    let u32_components = matches!(args.component_type, ComponentTypeArg::U32);
    match args.value_type {
        ValueTypeArg::F32 => with_component_and_distance!(args, metric, |C, D| {
            search_sparse_with::<C, D, PlainSparseDataset<C, f32, D>>(args)
        }),
        ValueTypeArg::F16 => with_component_and_distance!(args, metric, |C, D| {
            search_sparse_with::<C, D, PlainSparseDataset<C, f16, D>>(args)
        }),
        ValueTypeArg::Fixedu8 => with_component_and_distance!(args, metric, |C, D| {
            search_sparse_with::<C, D, ScalarSparseDataset<C, f32, FixedU8Q, D>>(args)
        }),
        ValueTypeArg::Fixedu16 => with_component_and_distance!(args, metric, |C, D| {
            search_sparse_with::<C, D, ScalarSparseDataset<C, f32, FixedU16Q, D>>(args)
        }),
        ValueTypeArg::Scalaru8 => with_component_and_distance!(args, metric, |C, D| {
            search_sparse_with::<C, D, ScalarU8SparseDataset<C, D>>(args)
        }),
        ValueTypeArg::Scalaru16 => with_component_and_distance!(args, metric, |C, D| {
            search_sparse_with::<C, D, ScalarU16SparseDataset<C, D>>(args)
        }),
        #[cfg(target_arch = "x86_64")]
        ValueTypeArg::BlockedSparse => {
            search_sparse_with::<u16, DotProduct, BlockedSparseDataset>(args)
        }
        #[cfg(target_arch = "x86_64")]
        ValueTypeArg::BlockedSparseScalaru8 => {
            search_sparse_with::<u16, DotProduct, BlockedSparseScalarU8Dataset>(args)
        }
        #[cfg(target_arch = "x86_64")]
        ValueTypeArg::BlockedSparseScalaru16 => {
            search_sparse_with::<u16, DotProduct, BlockedSparseScalarU16Dataset>(args)
        }
        ValueTypeArg::Dotpacking8Fixedu8 => {
            search_sparse_with::<u16, DotProduct, DotPacking8U16FixedU8Dataset>(args)
        }
        ValueTypeArg::Dotpacking8Scalaru8 if u32_components => {
            search_sparse_with::<u32, DotProduct, BridgedDotPacking8U32ScalarU8Dataset>(args)
        }
        ValueTypeArg::Dotpacking8Scalaru8 => {
            search_sparse_with::<u16, DotProduct, DotPacking8U16ScalarU8Dataset>(args)
        }
        ValueTypeArg::Dotpacking8Scalaru16 if u32_components => {
            search_sparse_with::<u32, DotProduct, BridgedDotPacking8U32ScalarU16Dataset>(args)
        }
        ValueTypeArg::Dotpacking8Scalaru16 => {
            search_sparse_with::<u16, DotProduct, DotPacking8U16ScalarU16Dataset>(args)
        }
        ValueTypeArg::Dotpacking8F16 => {
            search_sparse_with::<u16, DotProduct, DotPacking8U16F16Dataset>(args)
        }
        ValueTypeArg::Dotpacking8Scalar4bit => {
            search_sparse_with::<u16, DotProduct, DotPacking8U16Scalar4BitDataset>(args)
        }
    }
}

/// Picks the graph type. `permuted` loads as the same Rust type as `standard`.
fn search_sparse_with<C, D, Ds>(args: &Args)
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage,
    D: ScalarSparseSupportedDistance,
    Ds: Dataset + Sync + SpaceUsage + for<'de> serde::Deserialize<'de>,
    Ds::Encoder:
        for<'q> VectorEncoder<QueryVector<'q> = SparseVectorView<'q, C, f32>, QueryParams = ()>,
    <Ds::Encoder as VectorEncoder>::Distance: Distance,
{
    match args.graph_type {
        GraphType::Standard | GraphType::Permuted => {
            search_sparse_index::<C, D, Ds, Graph<PlainNeighbors>>(args)
        }
        GraphType::FixedDegree => search_sparse_index::<C, D, Ds, GraphFixedDegree>(args),
        GraphType::Streamvbyte => {
            search_sparse_index::<C, D, Ds, Graph<StreamVByteNeighbors>>(args)
        }
    }
}

fn search_sparse_index<C, D, Ds, G>(args: &Args)
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage,
    D: ScalarSparseSupportedDistance,
    Ds: Dataset + Sync + SpaceUsage + for<'de> serde::Deserialize<'de>,
    Ds::Encoder:
        for<'q> VectorEncoder<QueryVector<'q> = SparseVectorView<'q, C, f32>, QueryParams = ()>,
    <Ds::Encoder as VectorEncoder>::Distance: Distance,
    G: GraphBound,
{
    let config = create_search_config(args);
    let queries: PlainSparseDataset<C, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });
    let num_queries = queries.len();

    let index: HNSW<Ds, G> =
        <HNSW<Ds, G> as IndexSerializer>::load_index(&args.index_file).unwrap();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn write_results_to_file(output_path: &str, results: &[(f32, usize)], k: usize) {
    let mut file = File::create(output_path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}
