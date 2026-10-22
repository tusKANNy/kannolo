use std::{fmt::Debug, time::Instant};

use clap::{Parser, ValueEnum};
use half::f16;
use serde::{Deserialize, Serialize};
use std::process;

use kannolo::graph::GraphFixedDegree;
use kannolo::graph::graph::{Graph, GrowableGraph};
use kannolo::graph::neighbors::{
    MAX_NEIGHBORS_PER_NODE, NeighborData, Neighbors, PlainNeighbors, StreamVByteNeighbors,
};
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
use vectorium::IndexSerializer;
use vectorium::dataset::ConvertFrom;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use vectorium::{
    BlockedSparseDataset, BlockedSparseScalarU8Dataset, BlockedSparseScalarU16Dataset,
};
use vectorium::{
    BridgedDotPacking8U32ScalarU8Dataset, BridgedDotPacking8U32ScalarU16Dataset, Dataset,
    DenseDataset, DotPacking8U16F16Dataset, DotPacking8U16FixedU8Dataset,
    DotPacking8U16Scalar4BitDataset, DotPacking8U16ScalarU8Dataset, DotPacking8U16ScalarU16Dataset,
    FixedU8Q, FixedU16Q, Float, FromF32, PlainDenseDataset, PlainSparseDataset, RabitqConfig,
    RabitqExtConfig, RabitqExtQuantizer, RabitqQuantizer, RabitqSupportedDistance,
    ScalarSparseDataset, ScalarU8SparseDataset, ScalarU16SparseDataset, SpaceUsage, ValueType,
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
    /// 64-byte blocks of 16 `u16` components with `f16` values.
    #[value(name = "blocked-sparse")]
    BlockedSparse,
    #[value(name = "blocked-sparse-scalaru8")]
    BlockedSparseScalaru8,
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
        process::exit(2);
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
    /// RaBitQ: 1-bit-per-component document codes.
    Rabitq,
    /// Extended RaBitQ: `--rabitq-total-bits` per component (2, 4 or 8).
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
    /// Plain neighbor storage, nodes reordered by Enhanced Graph Bisection (EGB).
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

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the dataset file.
    #[clap(short, long, value_parser)]
    data_file: String,

    /// The output file where to save the index.
    #[clap(short, long, value_parser)]
    output_file: String,

    /// The type of vectors (dense or sparse).
    #[clap(long, value_enum)]
    dataset_type: DatasetType,

    /// Value type for stored values. Dense plain: f32, f16. Sparse: f32, f16, fixedu8, fixedu16,
    /// scalaru8, scalaru16, blocked-sparse[-scalaru8|-scalaru16], dotpacking8-{fixedu8, scalaru8,
    /// scalaru16, f16, scalar4bit}. Ignored for pq and rabitq (they determine their own values).
    /// Default: blocked-sparse for sparse data, f32 for dense data.
    #[clap(long = "value-type", value_enum)]
    value_type_arg: Option<ValueTypeArg>,

    /// `--value-type`, or its default; set in `main`.
    #[clap(skip)]
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

    /// The graph type: `standard`, `fixed-degree`, `permuted` (EGB-reordered, uncompressed),
    /// or `streamvbyte` (EGB-reordered, delta+StreamVByte compressed).
    #[clap(long, value_enum)]
    #[arg(default_value_t = GraphType::Standard)]
    graph_type: GraphType,

    /// The number of neighbors per node.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    /// The size of the candidate pool at construction time.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 150)]
    ef_construction: usize,

    /// The type of distance to use. Either 'euclidean' or 'dotproduct'.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("dotproduct"))]
    distance: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    pq_subspaces: usize,

    /// The number of bits per subspace for Product Quantization (ignored; vectorium PQ is fixed).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 8)]
    nbits: usize,

    /// The size of the sample used for training Product Quantization (ignored; vectorium PQ samples automatically).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 100_000)]
    sample_size: usize,

    /// Seed for RaBitQ's random orthogonal rotation (rabitq, rabitq-ext).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 42)]
    rabitq_seed: u64,

    /// Skip RaBitQ's random orthogonal rotation. Faster to encode and to query, but the codes
    /// lose the information-spreading guarantee, so recall on real data is expected to drop.
    #[clap(long, action)]
    rabitq_no_rotate: bool,

    /// Bits per component of the RaBitQ-ext document code: 2, 4 or 8 (rabitq-ext only).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 4)]
    rabitq_total_bits: u32,

    /// Quantize each RaBitQ-ext document with an exact per-vector rescale search instead of the
    /// constant factor estimated at training time. Slower to encode; buys least at large `d`.
    #[clap(long, action)]
    rabitq_exact_quant: bool,
}

/// RaBitQ packs document codes into 64-bit words, so the vector dimension must be a
/// multiple of 64. Checked here rather than inside the encoder, which would only fire
/// after the whole plain HNSW has been built.
fn check_rabitq_dim(dim: usize) {
    if !dim.is_multiple_of(64) {
        eprintln!("Error: RaBitQ requires a vector dimension that is a multiple of 64, got {dim}.");
        process::exit(1);
    }
}

fn rabitq_config(args: &Args) -> RabitqConfig {
    RabitqConfig {
        seed: args.rabitq_seed,
        rotate: !args.rabitq_no_rotate,
    }
}

fn rabitq_ext_config(args: &Args) -> RabitqExtConfig {
    if !matches!(args.rabitq_total_bits, 2 | 4 | 8) {
        eprintln!(
            "Error: --rabitq-total-bits must be 2, 4 or 8, got {}. Use --encoder rabitq for 1 bit.",
            args.rabitq_total_bits
        );
        process::exit(1);
    }
    RabitqExtConfig {
        total_bits: args.rabitq_total_bits,
        seed: args.rabitq_seed,
        rotate: !args.rabitq_no_rotate,
        faster_quant: !args.rabitq_exact_quant,
    }
}

#[derive(Clone, Copy)]
enum Distance {
    Euclidean,
    DotProduct,
}

fn parse_metric(metric: &str) -> Distance {
    match metric {
        "euclidean" | "l2" => Distance::Euclidean,
        "dotproduct" | "ip" => Distance::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'euclidean' and 'dotproduct'.");
            process::exit(1);
        }
    }
}

/// Order matters: `--pq-subspaces 0` picks the first entry that divides the dimension.
///
/// The two new widths sit at the end so that adding them cannot change what auto-selection
/// returns. Each is a multiple of an earlier entry (`8 | 24`, `128 | 256`), so any dimension one
/// of them divides is already matched further up the list and the search never reaches them.
/// They are only ever selected by being asked for explicitly.
const PQ_SUPPORTED_SUBSPACES: [usize; 11] = [128, 192, 96, 64, 48, 32, 16, 8, 4, 24, 256];

fn choose_pq_subspaces(dim: usize, requested: usize) -> usize {
    if requested == 0 {
        for &m in &PQ_SUPPORTED_SUBSPACES {
            if dim.is_multiple_of(m) {
                return m;
            }
        }
        eprintln!(
            "Error: Could not auto-select pq-subspaces for dimension {dim}. Supported: 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256."
        );
        process::exit(1);
    }

    if !PQ_SUPPORTED_SUBSPACES.contains(&requested) {
        eprintln!(
            "Error: Unsupported pq-subspaces value {requested}. Supported: 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256."
        );
        process::exit(1);
    }

    if !dim.is_multiple_of(requested) {
        eprintln!("Error: pq-subspaces ({requested}) must divide vector dimension ({dim}).");
        process::exit(1);
    }

    requested
}

fn main() {
    reject_removed_names();
    let mut args: Args = Args::parse();
    let distance = parse_metric(&args.distance);
    args.value_type = args
        .value_type_arg
        .clone()
        .unwrap_or(match args.dataset_type {
            DatasetType::Sparse => ValueTypeArg::BlockedSparse,
            DatasetType::Dense => ValueTypeArg::F32,
        });

    // Cross-validation of encoder / dataset-type / value-type combinations
    match (&args.dataset_type, &args.encoder) {
        (DatasetType::Sparse, EncoderType::Pq) => {
            eprintln!("Error: PQ encoder is only available for dense vectors.");
            process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::Rabitq | EncoderType::RabitqExt) => {
            eprintln!("Error: RaBitQ encoders are only available for dense vectors.");
            process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Plain) if !args.value_type.is_dense() => {
            eprintln!(
                "Error: value type {} is only available for sparse vectors.",
                args.value_type
            );
            process::exit(1);
        }
        (DatasetType::Dense, _) if !matches!(args.component_type, ComponentTypeArg::U16) => {
            eprintln!("Error: component-type is only applicable to sparse datasets.");
            process::exit(1);
        }
        (DatasetType::Sparse, _)
            if args.value_type.is_u16_only()
                && !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!(
                "Error: value type {} supports only component-type u16.",
                args.value_type
            );
            process::exit(1);
        }
        (DatasetType::Sparse, _)
            if args.value_type.is_dot_product_only()
                && !matches!(distance, Distance::DotProduct) =>
        {
            eprintln!(
                "Error: value type {} supports only dotproduct distance.",
                args.value_type
            );
            process::exit(1);
        }
        _ => {}
    }

    // Checked here rather than in `StreamVByteNeighbors::from`, which only runs once the whole
    // index (and the EGB permutation) has already been built.
    if matches!(args.graph_type, GraphType::Streamvbyte) && 2 * args.m > MAX_NEIGHBORS_PER_NODE {
        eprintln!(
            "Error: --graph-type streamvbyte caps the ground level at {MAX_NEIGHBORS_PER_NODE} neighbors per node, but --m {} gives 2 * m = {}.",
            args.m,
            2 * args.m
        );
        process::exit(1);
    }

    if matches!(args.encoder, EncoderType::Pq) {
        if args.nbits != 8 {
            eprintln!("Warning: vectorium PQ ignores --nbits (fixed codebook size).");
        }
        if args.sample_size != 100_000 {
            eprintln!("Warning: vectorium PQ ignores --sample-size and uses automatic sampling.");
        }
    }

    let config = HNSWBuildConfiguration::default()
        .with_num_neighbors(args.m)
        .with_ef_construction(args.ef_construction);

    println!(
        "Building Index with M: {}, ef_construction: {}",
        args.m, args.ef_construction
    );

    match (
        &args.dataset_type,
        &args.encoder,
        &args.value_type,
        &args.graph_type,
    ) {
        // Dense plain f32
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Standard) => {
            build_dense_plain::<f32, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::FixedDegree) => {
            build_dense_plain::<f32, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Permuted) => {
            build_dense_plain_permuted::<f32, PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Streamvbyte) => {
            build_dense_plain_permuted::<f32, StreamVByteNeighbors>(&args, distance, &config);
        }
        // Dense plain f16
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Standard) => {
            build_dense_plain::<f16, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::FixedDegree) => {
            build_dense_plain::<f16, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Permuted) => {
            build_dense_plain_permuted::<f16, PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Streamvbyte) => {
            build_dense_plain_permuted::<f16, StreamVByteNeighbors>(&args, distance, &config);
        }
        // Dense PQ (value-type ignored)
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Standard) => {
            build_dense_pq::<Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::FixedDegree) => {
            build_dense_pq::<GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Permuted) => {
            build_dense_pq_permuted::<PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Streamvbyte) => {
            build_dense_pq_permuted::<StreamVByteNeighbors>(&args, distance, &config);
        }
        // Dense RaBitQ (value-type ignored)
        (DatasetType::Dense, EncoderType::Rabitq, _, GraphType::Standard) => {
            build_dense_rabitq::<Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Rabitq, _, GraphType::FixedDegree) => {
            build_dense_rabitq::<GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Rabitq, _, GraphType::Permuted) => {
            build_dense_rabitq_permuted::<PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Rabitq, _, GraphType::Streamvbyte) => {
            build_dense_rabitq_permuted::<StreamVByteNeighbors>(&args, distance, &config);
        }
        // Dense RaBitQ-ext (value-type ignored)
        (DatasetType::Dense, EncoderType::RabitqExt, _, GraphType::Standard) => {
            build_dense_rabitq_ext::<Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::RabitqExt, _, GraphType::FixedDegree) => {
            build_dense_rabitq_ext::<GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::RabitqExt, _, GraphType::Permuted) => {
            build_dense_rabitq_ext_permuted::<PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::RabitqExt, _, GraphType::Streamvbyte) => {
            build_dense_rabitq_ext_permuted::<StreamVByteNeighbors>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, _, _) => build_sparse(&args, distance, &config),
        // Unreachable: caught by earlier validation
        (DatasetType::Sparse, EncoderType::Pq, _, _)
        | (DatasetType::Sparse, EncoderType::Rabitq | EncoderType::RabitqExt, _, _)
        | (DatasetType::Dense, EncoderType::Plain, _, _) => {
            unreachable!()
        }
    }
}

fn build_dense_plain<V, G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => {
            build_dense_plain_with_distance::<V, SquaredEuclideanDistance, G>(args, config)
        }
        Distance::DotProduct => build_dense_plain_with_distance::<V, DotProduct, G>(args, config),
    }
}

fn build_dense_plain_with_distance<V, D, G>(args: &Args, config: &HNSWBuildConfiguration)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarDenseSupportedDistance,
    G: GraphBound,
{
    let dataset = read_dense_v_dataset::<V, D>(args);
    build_and_save::<_, _, G>(dataset, config, &args.output_file, |i| i);
}

fn build_dense_plain_permuted<V, Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize + Default,
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => {
            build_dense_plain_permuted_with_distance::<V, SquaredEuclideanDistance, Ndst>(
                args, config,
            )
        }
        Distance::DotProduct => {
            build_dense_plain_permuted_with_distance::<V, DotProduct, Ndst>(args, config)
        }
    }
}

fn build_dense_plain_permuted_with_distance<V, D, Ndst>(
    args: &Args,
    config: &HNSWBuildConfiguration,
) where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize + Default,
    D: ScalarDenseSupportedDistance,
    Ndst: NeighborsBound,
{
    let dataset = read_dense_v_dataset::<V, D>(args);
    build_permuted_and_save::<_, _, Ndst>(dataset, config, &args.output_file, |i| i);
}

fn build_dense_pq<G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => build_dense_pq_l2::<G>(args, config),
        Distance::DotProduct => build_dense_pq_ip::<G>(args, config),
    }
}

fn build_dense_pq_permuted<Ndst>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => build_dense_pq_permuted_l2::<Ndst>(args, config),
        Distance::DotProduct => build_dense_pq_permuted_ip::<Ndst>(args, config),
    }
}

fn build_dense_pq_l2<G>(args: &Args, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
        read_dense_plain_dataset::<SquaredEuclideanDistance>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_l2::<4, G>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_l2::<8, G>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_l2::<16, G>(dataset, config, &args.output_file),
        24 => build_dense_pq_with_m_l2::<24, G>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_l2::<32, G>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_l2::<48, G>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_l2::<64, G>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_l2::<96, G>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_l2::<128, G>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_l2::<192, G>(dataset, config, &args.output_file),
        256 => build_dense_pq_with_m_l2::<256, G>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn build_dense_pq_ip<G>(args: &Args, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, DotProduct> = read_dense_plain_dataset::<DotProduct>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_ip::<4, G>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_ip::<8, G>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_ip::<16, G>(dataset, config, &args.output_file),
        24 => build_dense_pq_with_m_ip::<24, G>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_ip::<32, G>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_ip::<48, G>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_ip::<64, G>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_ip::<96, G>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_ip::<128, G>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_ip::<192, G>(dataset, config, &args.output_file),
        256 => build_dense_pq_with_m_ip::<256, G>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn build_dense_pq_permuted_l2<Ndst>(args: &Args, config: &HNSWBuildConfiguration)
where
    Ndst: NeighborsBound,
{
    let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
        read_dense_plain_dataset::<SquaredEuclideanDistance>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_l2_permuted::<4, Ndst>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_l2_permuted::<8, Ndst>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_l2_permuted::<16, Ndst>(dataset, config, &args.output_file),
        24 => build_dense_pq_with_m_l2_permuted::<24, Ndst>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_l2_permuted::<32, Ndst>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_l2_permuted::<48, Ndst>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_l2_permuted::<64, Ndst>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_l2_permuted::<96, Ndst>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_l2_permuted::<128, Ndst>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_l2_permuted::<192, Ndst>(dataset, config, &args.output_file),
        256 => build_dense_pq_with_m_l2_permuted::<256, Ndst>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn build_dense_pq_permuted_ip<Ndst>(args: &Args, config: &HNSWBuildConfiguration)
where
    Ndst: NeighborsBound,
{
    let dataset: PlainDenseDataset<f32, DotProduct> = read_dense_plain_dataset::<DotProduct>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_ip_permuted::<4, Ndst>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_ip_permuted::<8, Ndst>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_ip_permuted::<16, Ndst>(dataset, config, &args.output_file),
        24 => build_dense_pq_with_m_ip_permuted::<24, Ndst>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_ip_permuted::<32, Ndst>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_ip_permuted::<48, Ndst>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_ip_permuted::<64, Ndst>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_ip_permuted::<96, Ndst>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_ip_permuted::<128, Ndst>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_ip_permuted::<192, Ndst>(dataset, config, &args.output_file),
        256 => build_dense_pq_with_m_ip_permuted::<256, Ndst>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn read_dense_plain_dataset<D>(args: &Args) -> PlainDenseDataset<f32, D>
where
    D: ScalarDenseSupportedDistance,
{
    read_npy_f32::<D>(&args.data_file).unwrap_or_else(|e| {
        eprintln!("Error reading .npy file: {e:?}");
        process::exit(1);
    })
}

/// Reads the `.npy` dataset and re-quantizes it into a `DenseDataset` with value type `V`.
fn read_dense_v_dataset<V, D>(args: &Args) -> DenseDataset<PlainDenseQuantizer<V, D>>
where
    V: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    let dataset_f32 = read_npy_f32::<D>(&args.data_file).unwrap_or_else(|e| {
        eprintln!("Error reading .npy file: {e:?}");
        process::exit(1);
    });
    let d = dataset_f32.input_dim();
    let n_vecs = dataset_f32.len();
    let data: Vec<V> = dataset_f32
        .values()
        .iter()
        .map(|v| V::from_f32_saturating(*v))
        .collect();

    let encoder = PlainDenseQuantizer::<V, D>::new(d);
    DenseDataset::from_raw(data.into_boxed_slice(), n_vecs, encoder)
}

/// Runs `$body` with `$c` and `$d` bound, as type aliases, to the `--component-type` and the
/// distance, so that one generic call covers the four (component type, distance) pairs.
macro_rules! with_component_and_distance {
    ($args:expr, $distance:expr, |$c:ident, $d:ident| $body:expr) => {
        match (&$args.component_type, $distance) {
            (ComponentTypeArg::U16, Distance::DotProduct) => {
                type $c = u16;
                type $d = DotProduct;
                $body
            }
            (ComponentTypeArg::U16, Distance::Euclidean) => {
                type $c = u16;
                type $d = SquaredEuclideanDistance;
                $body
            }
            (ComponentTypeArg::U32, Distance::DotProduct) => {
                type $c = u32;
                type $d = DotProduct;
                $body
            }
            (ComponentTypeArg::U32, Distance::Euclidean) => {
                type $c = u32;
                type $d = SquaredEuclideanDistance;
                $body
            }
        }
    };
}

/// Sparse datasets. `f32` and `f16` build the graph on the stored values; every other value type
/// builds it on `f32` and then re-encodes the dataset. Value types that are dot-product-only or
/// `u16`-only were validated in `main`.
fn build_sparse(args: &Args, distance: Distance, config: &HNSWBuildConfiguration) {
    let u32_components = matches!(args.component_type, ComponentTypeArg::U32);
    match args.value_type {
        ValueTypeArg::F32 => with_component_and_distance!(args, distance, |C, D| {
            build_sparse_plain::<C, f32, D>(args, config)
        }),
        ValueTypeArg::F16 => with_component_and_distance!(args, distance, |C, D| {
            build_sparse_plain::<C, f16, D>(args, config)
        }),
        ValueTypeArg::Fixedu8 => with_component_and_distance!(args, distance, |C, D| {
            build_sparse_converted::<C, D, ScalarSparseDataset<C, f32, FixedU8Q, D>>(args, config)
        }),
        ValueTypeArg::Fixedu16 => with_component_and_distance!(args, distance, |C, D| {
            build_sparse_converted::<C, D, ScalarSparseDataset<C, f32, FixedU16Q, D>>(args, config)
        }),
        ValueTypeArg::Scalaru8 => with_component_and_distance!(args, distance, |C, D| {
            build_sparse_converted::<C, D, ScalarU8SparseDataset<C, D>>(args, config)
        }),
        ValueTypeArg::Scalaru16 => with_component_and_distance!(args, distance, |C, D| {
            build_sparse_converted::<C, D, ScalarU16SparseDataset<C, D>>(args, config)
        }),
        ValueTypeArg::BlockedSparse => {
            build_sparse_converted_from_f16::<BlockedSparseDataset>(args, config)
        }
        ValueTypeArg::BlockedSparseScalaru8 => {
            build_sparse_converted::<u16, DotProduct, BlockedSparseScalarU8Dataset>(args, config)
        }
        ValueTypeArg::BlockedSparseScalaru16 => {
            build_sparse_converted::<u16, DotProduct, BlockedSparseScalarU16Dataset>(args, config)
        }
        ValueTypeArg::Dotpacking8Fixedu8 => {
            build_sparse_converted::<u16, DotProduct, DotPacking8U16FixedU8Dataset>(args, config)
        }
        ValueTypeArg::Dotpacking8Scalaru8 if u32_components => {
            build_sparse_converted::<u32, DotProduct, BridgedDotPacking8U32ScalarU8Dataset>(
                args, config,
            )
        }
        ValueTypeArg::Dotpacking8Scalaru8 => {
            build_sparse_converted::<u16, DotProduct, DotPacking8U16ScalarU8Dataset>(args, config)
        }
        ValueTypeArg::Dotpacking8Scalaru16 if u32_components => {
            build_sparse_converted::<u32, DotProduct, BridgedDotPacking8U32ScalarU16Dataset>(
                args, config,
            )
        }
        ValueTypeArg::Dotpacking8Scalaru16 => {
            build_sparse_converted::<u16, DotProduct, DotPacking8U16ScalarU16Dataset>(args, config)
        }
        ValueTypeArg::Dotpacking8F16 => {
            build_sparse_converted::<u16, DotProduct, DotPacking8U16F16Dataset>(args, config)
        }
        ValueTypeArg::Dotpacking8Scalar4bit => {
            build_sparse_converted::<u16, DotProduct, DotPacking8U16Scalar4BitDataset>(args, config)
        }
    }
}

fn read_sparse_dataset<C, V, D>(args: &Args) -> PlainSparseDataset<C, V, D>
where
    C: vectorium::ComponentType + num_traits::FromPrimitive,
    V: ValueType + Float + FromF32 + SpaceUsage,
    D: ScalarSparseSupportedDistance,
{
    read_seismic_format::<C, V, D>(&args.data_file).unwrap_or_else(|e| {
        eprintln!("Error reading dataset: {e:?}");
        process::exit(1);
    })
}

/// Builds the graph directly on the `V` values read from the dataset file.
fn build_sparse_plain<C, V, D>(args: &Args, config: &HNSWBuildConfiguration)
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarSparseSupportedDistance,
{
    let dataset: PlainSparseDataset<C, V, D> = read_sparse_dataset(args);
    let output_file = &args.output_file;
    match args.graph_type {
        GraphType::Standard => {
            build_and_save::<_, _, Graph<PlainNeighbors>>(dataset, config, output_file, |i| i)
        }
        GraphType::FixedDegree => {
            build_and_save::<_, _, GraphFixedDegree>(dataset, config, output_file, |i| i)
        }
        GraphType::Permuted => {
            build_permuted_and_save::<_, _, PlainNeighbors>(dataset, config, output_file, |i| i)
        }
        GraphType::Streamvbyte => build_permuted_and_save::<_, _, StreamVByteNeighbors>(
            dataset,
            config,
            output_file,
            |i| i,
        ),
    }
}

/// Builds the graph on the `f16` dataset, then re-encodes the dataset into `Dfinal`.
///
/// For encoders that store `f16` values (blocked sparse): the graph is the one a plain `f16`
/// build produces, and every score during construction and search is an `f16` product.
fn build_sparse_converted_from_f16<Dfinal>(args: &Args, config: &HNSWBuildConfiguration)
where
    Dfinal: ConvertFrom<PlainSparseDataset<u16, f16, DotProduct>, Config = ()>
        + Dataset<Owned = Dfinal>
        + Sync
        + SpaceUsage
        + Serialize,
{
    let dataset: PlainSparseDataset<u16, f16, DotProduct> = read_sparse_dataset(args);
    let output_file = &args.output_file;
    match args.graph_type {
        GraphType::Standard => {
            build_and_save::<_, Dfinal, Graph<PlainNeighbors>>(dataset, config, output_file, |i| {
                i.convert_dataset_into(())
            })
        }
        GraphType::FixedDegree => {
            build_and_save::<_, Dfinal, GraphFixedDegree>(dataset, config, output_file, |i| {
                i.convert_dataset_into(())
            })
        }
        GraphType::Permuted => build_permuted_and_save::<_, Dfinal, PlainNeighbors>(
            dataset,
            config,
            output_file,
            |i| i.convert_dataset_into(()),
        ),
        GraphType::Streamvbyte => build_permuted_and_save::<_, Dfinal, StreamVByteNeighbors>(
            dataset,
            config,
            output_file,
            |i| i.convert_dataset_into(()),
        ),
    }
}

/// Builds the graph on the `f32` dataset, then re-encodes the dataset into `Dfinal`.
fn build_sparse_converted<C, D, Dfinal>(args: &Args, config: &HNSWBuildConfiguration)
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    D: ScalarSparseSupportedDistance,
    Dfinal: ConvertFrom<PlainSparseDataset<C, f32, D>, Config = ()>
        + Dataset<Owned = Dfinal>
        + Sync
        + SpaceUsage
        + Serialize,
{
    let dataset: PlainSparseDataset<C, f32, D> = read_sparse_dataset(args);
    let output_file = &args.output_file;
    match args.graph_type {
        GraphType::Standard => {
            build_and_save::<_, Dfinal, Graph<PlainNeighbors>>(dataset, config, output_file, |i| {
                i.convert_dataset_into(())
            })
        }
        GraphType::FixedDegree => {
            build_and_save::<_, Dfinal, GraphFixedDegree>(dataset, config, output_file, |i| {
                i.convert_dataset_into(())
            })
        }
        GraphType::Permuted => build_permuted_and_save::<_, Dfinal, PlainNeighbors>(
            dataset,
            config,
            output_file,
            |i| i.convert_dataset_into(()),
        ),
        GraphType::Streamvbyte => build_permuted_and_save::<_, Dfinal, StreamVByteNeighbors>(
            dataset,
            config,
            output_file,
            |i| i.convert_dataset_into(()),
        ),
    }
}

fn build_dense_pq_with_m_l2<const M: usize, G>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    G: GraphBound,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Encoder = ProductQuantizer<M, SquaredEuclideanDistance>>,
    for<'a> DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<&'a PlainDenseDataset<f32, SquaredEuclideanDistance>, Config = ()>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    build_and_save::<_, DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, G>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into_ref(()),
    );
}

fn build_dense_pq_with_m_ip<const M: usize, G>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    G: GraphBound,
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    build_and_save::<_, DenseDataset<ProductQuantizer<M, DotProduct>>, G>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into_ref(()),
    );
}

fn build_dense_pq_with_m_l2_permuted<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    Ndst: NeighborsBound,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>: Dataset<
            Encoder = ProductQuantizer<M, SquaredEuclideanDistance>,
            Owned = DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>,
        >,
    for<'a> DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<&'a PlainDenseDataset<f32, SquaredEuclideanDistance>, Config = ()>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    build_permuted_and_save::<_, DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, Ndst>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into_ref(()),
    );
}

fn build_dense_pq_with_m_ip_permuted<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    Ndst: NeighborsBound,
    DenseDataset<ProductQuantizer<M, DotProduct>>: Dataset<
            Encoder = ProductQuantizer<M, DotProduct>,
            Owned = DenseDataset<ProductQuantizer<M, DotProduct>>,
        >,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    build_permuted_and_save::<_, DenseDataset<ProductQuantizer<M, DotProduct>>, Ndst>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into_ref(()),
    );
}

// ---------------------------------------------------------------------------
// RaBitQ
// ---------------------------------------------------------------------------
//
// Same shape as the PQ path: build the graph over the *plain* f32 vectors, so the
// neighbor lists come from exact distances, then re-encode the dataset in place and
// keep the graph. The encoder trains on the source, hence `convert_dataset_into_ref`.
//
// Unlike PQ there is no const-generic fan-out — the code width is a runtime field of
// `RabitqExtConfig`, not a type parameter — so one function per (metric, graph kind).

fn build_dense_rabitq<G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => {
            build_dense_rabitq_with_distance::<SquaredEuclideanDistance, G>(args, config)
        }
        Distance::DotProduct => build_dense_rabitq_with_distance::<DotProduct, G>(args, config),
    }
}

fn build_dense_rabitq_with_distance<D, G>(args: &Args, config: &HNSWBuildConfiguration)
where
    D: RabitqSupportedDistance + ScalarDenseSupportedDistance,
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, D> = read_dense_plain_dataset::<D>(args);
    check_rabitq_dim(dataset.input_dim());
    let rabitq = rabitq_config(args);
    build_and_save::<_, DenseDataset<RabitqQuantizer<D>>, G>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into_ref(rabitq),
    );
}

fn build_dense_rabitq_permuted<Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => build_dense_rabitq_permuted_with_distance::<
            SquaredEuclideanDistance,
            Ndst,
        >(args, config),
        Distance::DotProduct => {
            build_dense_rabitq_permuted_with_distance::<DotProduct, Ndst>(args, config)
        }
    }
}

fn build_dense_rabitq_permuted_with_distance<D, Ndst>(args: &Args, config: &HNSWBuildConfiguration)
where
    D: RabitqSupportedDistance + ScalarDenseSupportedDistance,
    Ndst: NeighborsBound,
{
    let dataset: PlainDenseDataset<f32, D> = read_dense_plain_dataset::<D>(args);
    check_rabitq_dim(dataset.input_dim());
    let rabitq = rabitq_config(args);
    // Quantize before permuting, never after — see `build_permuted_and_save`.
    build_permuted_and_save::<_, DenseDataset<RabitqQuantizer<D>>, Ndst>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into_ref(rabitq),
    );
}

fn build_dense_rabitq_ext<G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => {
            build_dense_rabitq_ext_with_distance::<SquaredEuclideanDistance, G>(args, config)
        }
        Distance::DotProduct => build_dense_rabitq_ext_with_distance::<DotProduct, G>(args, config),
    }
}

fn build_dense_rabitq_ext_with_distance<D, G>(args: &Args, config: &HNSWBuildConfiguration)
where
    D: RabitqSupportedDistance + ScalarDenseSupportedDistance,
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, D> = read_dense_plain_dataset::<D>(args);
    check_rabitq_dim(dataset.input_dim());
    let rabitq = rabitq_ext_config(args);
    build_and_save::<_, DenseDataset<RabitqExtQuantizer<D>>, G>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into_ref(rabitq),
    );
}

fn build_dense_rabitq_ext_permuted<Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => build_dense_rabitq_ext_permuted_with_distance::<
            SquaredEuclideanDistance,
            Ndst,
        >(args, config),
        Distance::DotProduct => {
            build_dense_rabitq_ext_permuted_with_distance::<DotProduct, Ndst>(args, config)
        }
    }
}

fn build_dense_rabitq_ext_permuted_with_distance<D, Ndst>(
    args: &Args,
    config: &HNSWBuildConfiguration,
) where
    D: RabitqSupportedDistance + ScalarDenseSupportedDistance,
    Ndst: NeighborsBound,
{
    let dataset: PlainDenseDataset<f32, D> = read_dense_plain_dataset::<D>(args);
    check_rabitq_dim(dataset.input_dim());
    let rabitq = rabitq_ext_config(args);
    // Quantize before permuting, never after — see `build_permuted_and_save`.
    build_permuted_and_save::<_, DenseDataset<RabitqExtQuantizer<D>>, Ndst>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into_ref(rabitq),
    );
}

trait GraphBound:
    kannolo::graph::GraphTrait + Serialize + for<'de> serde::Deserialize<'de> + From<GrowableGraph>
{
}
impl<T> GraphBound for T where
    T: kannolo::graph::GraphTrait
        + Serialize
        + for<'de> serde::Deserialize<'de>
        + From<GrowableGraph>
{
}

/// Bound satisfied by [`PlainNeighbors`] and [`StreamVByteNeighbors`], the two destination
/// backends of the `permuted`/`streamvbyte` graph types.
trait NeighborsBound: Neighbors + From<NeighborData> + Serialize + for<'de> Deserialize<'de> {}
impl<T> NeighborsBound for T where
    T: Neighbors + From<NeighborData> + Serialize + for<'de> Deserialize<'de>
{
}

/// Builds a `standard`/`fixed-degree` index, applies `convert`, then reports timing and space
/// usage and serializes it to `output_file`.
///
/// `convert` is the only per-encoder variation: identity (`|i| i`) when the built index is
/// already final, or `|i| i.convert_dataset_into(())` to re-encode the dataset (PQ, scalar,
/// DotPacking8, blocked) while keeping the graph unchanged.
fn build_and_save<Dsrc, Dfinal, G>(
    dataset: Dsrc,
    config: &HNSWBuildConfiguration,
    output_file: &str,
    convert: impl FnOnce(HNSW<Dsrc, G>) -> HNSW<Dfinal, G>,
) where
    Dsrc: Dataset + Sync + SpaceUsage,
    <Dsrc::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    Dfinal: Dataset + Sync + SpaceUsage + Serialize,
    G: GraphBound,
{
    let start_time = Instant::now();
    let plain_index: HNSW<Dsrc, G> = HNSW::build_index(dataset, config);
    let index = convert(plain_index);
    println!(
        "Time to build: {} s (before serializing)",
        start_time.elapsed().as_secs()
    );

    index.print_space_usage_bytes();
    index.save_index(output_file).unwrap_or_else(|e| {
        eprintln!("Error saving the index to {output_file}: {e:?}");
        process::exit(1);
    });
}

/// Like [`build_and_save`], but for the `permuted`/`streamvbyte` graph types: `convert` runs
/// first, then the index is rebuilt under an EGB node reordering into a fresh `Ndst` backend.
///
/// **`convert` runs before `permute_and_encode`.** Encoders fitted to the data (the DotPacking8
/// component order, the per-component quantization steps) are then trained exactly as on the
/// `standard` path, and `Dataset::permute` copies the already-encoded rows verbatim, so the
/// permuted dataset is byte-identical to the unpermuted one up to row order.
fn build_permuted_and_save<Dsrc, Dfinal, Ndst>(
    dataset: Dsrc,
    config: &HNSWBuildConfiguration,
    output_file: &str,
    convert: impl FnOnce(HNSW<Dsrc, Graph<PlainNeighbors>>) -> HNSW<Dfinal, Graph<PlainNeighbors>>,
) where
    Dsrc: Dataset + Sync + SpaceUsage,
    <Dsrc::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    // `permute_and_encode` hands the permuted dataset back as `Dfinal::Owned`. Every concrete dataset
    // sets `Owned = Self`, but the compiler cannot normalize that through the callers' generic
    // parameters, so those restate it in their own `where` clauses.
    Dfinal: Dataset<Owned = Dfinal> + Sync + SpaceUsage + Serialize,
    Ndst: NeighborsBound,
{
    let start_time = Instant::now();
    let plain_index: HNSW<Dsrc, Graph<PlainNeighbors>> = HNSW::build_index(dataset, config);
    let converted_index = convert(plain_index);
    let index = converted_index.permute_and_encode::<Ndst>();
    println!(
        "Time to build (incl. EGB permutation + compression): {} s (before serializing)",
        start_time.elapsed().as_secs()
    );

    index.print_space_usage_bytes();
    index.save_index(output_file).unwrap_or_else(|e| {
        eprintln!("Error saving the index to {output_file}: {e:?}");
        process::exit(1);
    });
}
