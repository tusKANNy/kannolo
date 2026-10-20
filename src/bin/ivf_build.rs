/// Build an IVF index over dense (`.npy`) or sparse (Seismic binary format) vectors.
use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;
use serde::Serialize;

use kannolo::graph::Graph;
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
use kannolo::ivf::{
    BuildArtifacts, IVF, KMEANS_HNSW_THRESHOLD, ReportedMetric, encode_clusters, prepare_ivf,
};
use vectorium::core::dataset::ConvertFrom;
use vectorium::core::index::Index;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::ScalarDenseSupportedDistance;
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::vector_encoder::VectorEncoder;
use vectorium::{
    Dataset, DenseDataset, FlatIndex, KMeansBuilder, MeanDataset, PlainDenseDataset,
    PlainDenseDatasetGrowable, PlainSparseDatasetGrowable, PlainSparseQuantizer, RabitqConfig,
    RabitqExtConfig, RabitqExtQuantizer, RabitqQuantizer, RabitqQueryParams, ScalarDenseDataset,
    ScalarDenseQuantizer, ScalarSparseDataset, SpaceUsage, WithDistance,
};

/// Dense input, read as f32. k-means on dense data always clusters under squared Euclidean.
type DenseRaw = PlainDenseDataset<f32, SquaredEuclideanDistance>;
/// Dense centroids are stored in f16, under the reported metric `M`.
type DenseCentroids<M> = ScalarDenseDataset<f32, f16, M>;
/// Sparse input and centroids: f32, u16 components, dot product.
type Sparse = vectorium::PlainSparseDataset<u16, f32, DotProduct>;
/// Clusters are encoded from the reordered vectors (or residuals) as f32 dot products.
type DenseStorage = PlainDenseDataset<f32, DotProduct>;

const PQ_SUPPORTED_SUBSPACES: [usize; 9] = [4, 8, 16, 32, 48, 64, 96, 128, 192];

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
#[derive(Debug, Clone, ValueEnum, Default)]
enum EncoderType {
    #[default]
    Plain,
    Pq,
    /// RaBitQ: 1-bit-per-component codes.
    Rabitq,
    /// Extended RaBitQ: `--rabitq-total-bits` per component (2, 4 or 8).
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
    data_file: String,
    #[clap(short, long)]
    output_file: String,
    /// Dense vectors (`.npy`) or sparse vectors (Seismic binary format).
    #[clap(long, value_enum, default_value_t = DatasetType::Dense)]
    dataset_type: DatasetType,
    /// Reported metric. Default: euclidean for dense, dotproduct for sparse (the only one
    /// sparse IVF supports).
    #[clap(long, value_enum)]
    distance: Option<MetricArg>,
    #[clap(long, value_enum, default_value_t = ValueTypeArg::F32)]
    value_type: ValueTypeArg,
    #[clap(long, default_value_t = 1024)]
    n_clusters: usize,
    #[clap(long, default_value_t = 25)]
    kmeans_n_iter: usize,
    #[clap(long, default_value_t = 1)]
    kmeans_n_redo: usize,
    #[clap(long)]
    kmeans_sample_size: Option<usize>,
    /// Dense only.
    #[clap(long, default_value_t = false)]
    residuals: bool,
    #[clap(long, default_value_t = false)]
    hnsw: bool,
    #[clap(long, alias = "m", default_value_t = 32)]
    m_hnsw: usize,
    #[clap(long, default_value_t = 200)]
    ef_construction: usize,
    /// Cluster storage encoder (dense only for anything but plain). `--m-pq` alone implies pq.
    #[clap(long, value_enum, default_value_t = EncoderType::Plain)]
    encoder: EncoderType,
    /// PQ subspaces (dense only).
    #[clap(long)]
    m_pq: Option<usize>,
    /// Seed for RaBitQ's random orthogonal rotation (rabitq, rabitq-ext).
    #[clap(long, default_value_t = 42)]
    rabitq_seed: u64,
    /// Skip RaBitQ's random orthogonal rotation.
    #[clap(long, default_value_t = false)]
    rabitq_no_rotate: bool,
    /// Bits per component of the RaBitQ-ext code: 2, 4 or 8 (rabitq-ext only).
    #[clap(long, default_value_t = 4)]
    rabitq_total_bits: u32,
    /// Bits per component of the quantized query used to compute the L2 correction (rabitq
    /// only; 1..=8, 1 only for plain dot-product indexes). Search with the same value.
    #[clap(long, default_value_t = 4)]
    rabitq_query_bits: u32,
    /// Quantize each RaBitQ-ext vector with an exact per-vector rescale search.
    #[clap(long, default_value_t = false)]
    rabitq_exact_quant: bool,
    /// Use HNSW for centroid assignment during k-means (default: auto when n_clusters >= 32768).
    #[clap(long, default_value_t = false)]
    kmeans_hnsw: bool,
    /// Spherical k-means: L2-normalize centroids after each update. Dense only: on sparse data
    /// the centroids' magnitudes are what dot-product clustering relies on.
    #[clap(long, default_value_t = false)]
    kmeans_spherical: bool,
    /// Sparse only: metric k-means clusters under (default dotproduct). The index always
    /// reports dot product; dense k-means always clusters under squared Euclidean.
    #[clap(long, value_enum)]
    kmeans_distance: Option<MetricArg>,
    /// Sparse only: prune each centroid, after every k-means update, to this many times its
    /// cluster's average nnz (largest components).
    #[clap(long)]
    centroid_nnz_factor: Option<f32>,
}

fn fail(msg: &str) -> ! {
    eprintln!("Error: {msg}");
    process::exit(1);
}

fn main() {
    let mut args = Args::parse();
    if args.m_pq.is_some() && matches!(args.encoder, EncoderType::Plain) {
        args.encoder = EncoderType::Pq;
    }
    match (&args.encoder, args.m_pq) {
        (EncoderType::Pq, None) => fail("--encoder pq needs --m-pq."),
        (EncoderType::Rabitq | EncoderType::RabitqExt, Some(_)) => {
            fail("--m-pq is for --encoder pq only.")
        }
        _ => {}
    }
    if matches!(args.encoder, EncoderType::Rabitq) && !(1..=8).contains(&args.rabitq_query_bits) {
        fail("--rabitq-query-bits must be in 1..=8.");
    }
    // A 1-bit RaBitQ query carries a per-query scale, which breaks adding exact terms (the
    // L2 correction, a residual's centroid term) to its estimate.
    let reports_l2 = matches!(args.dataset_type, DatasetType::Dense)
        && !matches!(args.distance, Some(MetricArg::Dotproduct));
    if matches!(args.encoder, EncoderType::Rabitq)
        && args.rabitq_query_bits == 1
        && (reports_l2 || args.residuals)
    {
        fail("--rabitq-query-bits 1 is only valid for plain dot-product RaBitQ indexes.");
    }
    if matches!(args.encoder, EncoderType::RabitqExt)
        && !matches!(args.rabitq_total_bits, 2 | 4 | 8)
    {
        fail("--rabitq-total-bits must be 2, 4 or 8.");
    }
    match args.dataset_type {
        DatasetType::Dense => {
            if args.kmeans_distance.is_some() {
                fail(
                    "--kmeans-distance is sparse only: dense k-means clusters under squared Euclidean.",
                );
            }
            if args.centroid_nnz_factor.is_some() {
                fail("--centroid-nnz-factor is sparse only.");
            }
            if matches!(args.value_type, ValueTypeArg::F16)
                && !matches!(args.encoder, EncoderType::Plain)
            {
                eprintln!("Note: --value-type f16 only applies to --encoder plain.");
            }
            if args.residuals && matches!(args.encoder, EncoderType::Plain) {
                eprintln!("Warning: --residuals gives no benefit with --encoder plain.");
            }
            if let Some(m) = args.m_pq
                && !PQ_SUPPORTED_SUBSPACES.contains(&m)
            {
                fail(&format!(
                    "unsupported --m-pq {m}. Supported: {PQ_SUPPORTED_SUBSPACES:?}"
                ));
            }
        }
        DatasetType::Sparse => {
            if !matches!(args.encoder, EncoderType::Plain) {
                fail("sparse IVF supports --encoder plain only.");
            }
            if args.residuals {
                fail(
                    "sparse IVF does not support residuals: a sparse residual v − c is effectively dense.",
                );
            }
            if args.kmeans_spherical {
                fail("--kmeans-spherical is dense only.");
            }
            if matches!(args.distance, Some(MetricArg::Euclidean)) {
                fail("sparse IVF reports dot product only.");
            }
        }
    }

    let kmeans_hnsw = args.kmeans_hnsw || args.n_clusters >= KMEANS_HNSW_THRESHOLD;
    println!(
        "Running k-means: n_clusters={} n_iter={} n_redo={} hnsw_assignment={}",
        args.n_clusters, args.kmeans_n_iter, args.kmeans_n_redo, kmeans_hnsw
    );
    let kmeans = KMeansBuilder::new()
        .n_iter(args.kmeans_n_iter)
        .n_redo(args.kmeans_n_redo)
        .verbose(true)
        .spherical(args.kmeans_spherical);

    match args.dataset_type {
        DatasetType::Dense => {
            let raw: DenseRaw = read_npy_f32(&args.data_file)
                .unwrap_or_else(|e| fail(&format!("reading {}: {e:?}", args.data_file)));
            // Checked here rather than in the encoder, which would only fire after k-means.
            if matches!(args.encoder, EncoderType::Rabitq | EncoderType::RabitqExt)
                && !raw.input_dim().is_multiple_of(64)
            {
                fail(&format!(
                    "RaBitQ needs a dimension that is a multiple of 64, got {}.",
                    raw.input_dim()
                ));
            }
            match args.distance.as_ref().unwrap_or(&MetricArg::Euclidean) {
                MetricArg::Euclidean => build_dense(raw, kmeans, kmeans_hnsw, &args),
                MetricArg::Dotproduct => build_dense(
                    PlainDenseDataset::<f32, DotProduct>::from(raw),
                    kmeans,
                    kmeans_hnsw,
                    &args,
                ),
            }
        }
        DatasetType::Sparse => {
            let data: Sparse = read_seismic_format(&args.data_file)
                .unwrap_or_else(|e| fail(&format!("reading {}: {e:?}", args.data_file)));
            println!(
                "Read {} vectors, {} nnz, dim {}",
                data.len(),
                data.nnz(),
                data.input_dim()
            );
            match args
                .kmeans_distance
                .as_ref()
                .unwrap_or(&MetricArg::Dotproduct)
            {
                MetricArg::Dotproduct => {
                    build_sparse::<DotProduct>(data, kmeans, kmeans_hnsw, &args)
                }
                MetricArg::Euclidean => {
                    build_sparse::<SquaredEuclideanDistance>(data, kmeans, kmeans_hnsw, &args)
                }
            }
        }
    }
}

/// Dense: cluster under squared Euclidean, route with f16 centroids under the reported `M`.
fn build_dense<M>(
    data: PlainDenseDataset<f32, M>,
    kmeans: KMeansBuilder,
    kmeans_hnsw: bool,
    args: &Args,
) where
    M: ScalarDenseSupportedDistance + ReportedMetric,
    PlainDenseDataset<f32, M>: Into<DenseStorage>,
    DenseCentroids<SquaredEuclideanDistance>: Into<DenseCentroids<M>>,
    DenseCentroids<M>: Serialize + for<'de> serde::Deserialize<'de>,
{
    let build_start = Instant::now();
    let a = prepare_ivf::<_, PlainDenseDatasetGrowable<f32, SquaredEuclideanDistance>>(
        &data,
        args.n_clusters,
        kmeans,
        ScalarDenseQuantizer::new(data.input_dim()),
        args.kmeans_sample_size,
        kmeans_hnsw,
        |_, _| {},
    );
    println!(
        "K-means + assignment: {:.1}s",
        build_start.elapsed().as_secs_f32()
    );
    // f16 centroids, relabelled with the reported metric.
    let stored: DenseStorage = if args.residuals {
        a.residuals::<_, PlainDenseDatasetGrowable<f32, DotProduct>>(
            &data,
            ScalarDenseQuantizer::new(data.input_dim()),
        )
    } else {
        a.reorder(&data).into()
    };
    let to_f16 = |c: DenseRaw| -> DenseCentroids<M> {
        DenseCentroids::<SquaredEuclideanDistance>::convert_from(&c, ()).into()
    };

    macro_rules! pq_arms {
        ($($m:literal),*) => {
            match (args.m_pq, &args.value_type) {
                (None, ValueTypeArg::F32) => {
                    finish::<_, _, _, DenseStorage, _>(stored, a, to_f16, (), &(), args, build_start)
                }
                (None, ValueTypeArg::F16) => finish::<_, _, _, ScalarDenseDataset<f32, f16, DotProduct>, _>(
                    stored, a, to_f16, (), &(), args, build_start,
                ),
                $((Some($m), _) => finish::<_, _, _, DenseDataset<ProductQuantizer<$m, DotProduct>>, _>(
                    stored, a, to_f16, (), &(), args, build_start,
                ),)*
                (Some(m), _) => unreachable!("m_pq={m}"),
            }
        };
    }
    match args.encoder {
        EncoderType::Plain | EncoderType::Pq => pq_arms!(4, 8, 16, 32, 48, 64, 96, 128, 192),
        EncoderType::Rabitq => finish::<_, _, _, DenseDataset<RabitqQuantizer<DotProduct>>, _>(
            stored,
            a,
            to_f16,
            RabitqConfig {
                seed: args.rabitq_seed,
                rotate: !args.rabitq_no_rotate,
            },
            &RabitqQueryParams {
                query_bits: args.rabitq_query_bits,
            },
            args,
            build_start,
        ),
        EncoderType::RabitqExt => {
            finish::<_, _, _, DenseDataset<RabitqExtQuantizer<DotProduct>>, _>(
                stored,
                a,
                to_f16,
                RabitqExtConfig {
                    total_bits: args.rabitq_total_bits,
                    seed: args.rabitq_seed,
                    rotate: !args.rabitq_no_rotate,
                    faster_quant: !args.rabitq_exact_quant,
                },
                &(),
                args,
                build_start,
            )
        }
    }
}

/// Sparse: cluster under `M`, route with f32 centroids under dot product.
fn build_sparse<M>(data: Sparse, kmeans: KMeansBuilder, kmeans_hnsw: bool, args: &Args)
where
    M: ScalarSparseSupportedDistance + From<f32>,
    vectorium::PlainSparseDataset<u16, f32, M>: Into<Sparse>,
{
    let build_start = Instant::now();
    let prune = args
        .centroid_nnz_factor
        .map(|f| vectorium::sparse_centroid_pruning(&data, f));
    let a = prepare_ivf::<_, PlainSparseDatasetGrowable<u16, f32, M>>(
        &data,
        args.n_clusters,
        kmeans,
        PlainSparseQuantizer::new(data.input_dim(), data.input_dim()),
        args.kmeans_sample_size,
        kmeans_hnsw,
        |members, centroid| {
            if let Some(prune) = &prune {
                prune(members, centroid)
            }
        },
    );
    println!(
        "K-means + assignment: {:.1}s, centroid nnz: {}",
        build_start.elapsed().as_secs_f32(),
        a.centroids.nnz()
    );
    // Same centroids, relabelled for dot-product routing.
    let stored = a.reorder(&data);
    let to_dp = |c: vectorium::PlainSparseDataset<u16, f32, M>| -> Sparse { c.into() };
    match args.value_type {
        ValueTypeArg::F32 => finish::<_, _, _, ScalarSparseDataset<u16, f32, f32, DotProduct>, _>(
            stored,
            a,
            to_dp,
            (),
            &(),
            args,
            build_start,
        ),
        ValueTypeArg::F16 => finish::<_, _, _, ScalarSparseDataset<u16, f32, f16, DotProduct>, _>(
            stored,
            a,
            to_dp,
            (),
            &(),
            args,
            build_start,
        ),
    }
}

/// Encode the cluster storage `D`, build the centroid index over `to_centroids(k-means
/// centroids)`, and save.
fn finish<S, C, DQ, D, Cfg>(
    stored: S,
    a: BuildArtifacts<C>,
    to_centroids: impl FnOnce(C) -> DQ,
    cluster_config: Cfg,
    cluster_query_params: &<D::Encoder as VectorEncoder>::QueryParams,
    args: &Args,
    build_start: Instant,
) where
    S: Dataset,
    C: MeanDataset,
    DQ: MeanDataset + Sync + SpaceUsage + Serialize,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    D: Dataset + for<'a> ConvertFrom<&'a S, Config = Cfg> + Serialize,
    D::Encoder: WithDistance<SquaredEuclideanDistance>,
    for<'a> <D::Encoder as VectorEncoder>::QueryVector<'a>: From<&'a DQ::Vector>,
    FlatIndex<DQ>: Index + Serialize,
    HNSW<DQ, Graph>: Index + Serialize,
{
    let t = Instant::now();
    let centroids = to_centroids(a.centroids);
    // Corrections read the stored centroids, so the residual term matches what the search reads.
    let (cluster, correction) =
        encode_clusters::<<DQ::Encoder as VectorEncoder>::Distance, _, D, _, _>(
            &stored,
            cluster_config,
            cluster_query_params,
            &centroids,
            &a.cluster_offsets,
            args.residuals,
        );
    println!("Clusters encoded in {:.1}s", t.elapsed().as_secs_f32());

    let t = Instant::now();
    if args.hnsw {
        let cfg = HNSWBuildConfiguration::default()
            .with_num_neighbors(args.m_hnsw)
            .with_ef_construction(args.ef_construction);
        let ci: HNSW<DQ, Graph> = HNSW::build_index(centroids, &cfg);
        println!("Centroid HNSW built in {:.1}s", t.elapsed().as_secs_f32());
        save(
            IVF::<DQ, D, _>::from_parts(
                ci,
                cluster,
                a.cluster_offsets,
                a.external_ids,
                correction,
                args.residuals,
            ),
            args,
            build_start,
        );
    } else {
        let ci = FlatIndex::from(centroids);
        println!(
            "Centroid FlatIndex built in {:.1}s",
            t.elapsed().as_secs_f32()
        );
        save(
            IVF::<DQ, D, _>::from_parts(
                ci,
                cluster,
                a.cluster_offsets,
                a.external_ids,
                correction,
                args.residuals,
            ),
            args,
            build_start,
        );
    }
}

fn save<DQ, D, CI>(index: IVF<DQ, D, CI>, args: &Args, build_start: Instant)
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
    IVF<DQ, D, CI>: vectorium::IndexSerializer + Serialize,
{
    println!(
        "Time to build: {} s (before serializing)",
        build_start.elapsed().as_secs()
    );
    vectorium::IndexSerializer::save_index(&index, &args.output_file)
        .unwrap_or_else(|e| fail(&format!("{e:?}")));
    println!("Saved → {}", args.output_file);
}
