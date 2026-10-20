use std::io::Write;
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use half::f16;
use std::fs::File;

use kannolo::graph::Graph;
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use kannolo::rerank_index::RerankIndex;
use vectorium::IndexSerializer;
use vectorium::core::vector::DenseVectorView;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::ScalarDenseSupportedDistance;
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::read_seismic_format;
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, DenseMultiVectorView, MultiVecTwoLevelProductQuantizer,
    MultiVectorDataset, PlainDenseDatasetGrowable, PlainDenseQuantizer, PlainMultiVecQuantizer,
    PlainSparseDataset,
};

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::ReadNpyExt;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the sparse HNSW index.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The sparse query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The folder path containing multivector data.
    /// For "plain": expects documents.npy, doclens.npy, queries.npy
    /// For "two-levels": additionally expects centroids.npy, pq_centroids.npy, residuals.npy, index_assignment.npy
    #[clap(long, value_parser)]
    multivec_data_folder: String,

    /// The multivector quantizer type: "plain" or "two-levels".
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("plain"))]
    multivector_quantizer: String,

    /// Number of PQ subspaces (M) - required only for "two-levels" quantizer.
    #[clap(long, value_parser)]
    pq_subspaces: Option<usize>,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The ef_search parameter for first-stage index.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    ef_search: usize,

    /// The distance metric ("euclidean" or "dotproduct").
    #[clap(long, value_parser)]
    distance: String,

    /// Number of candidates to retrieve from first stage.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 100)]
    k_candidates: usize,

    /// Alpha parameter for candidate pruning (optional, range 0-1).
    #[clap(long, value_parser)]
    alpha: Option<f32>,

    /// Beta parameter for early exit during reranking (optional).
    #[clap(long, value_parser)]
    beta: Option<usize>,

    /// Score candidates by the sum of first-stage and rerank scores (residual reranking).
    /// Only meaningful when the two scores are summable, e.g. the rerank dataset holds the
    /// residual part of a decomposed representation.
    #[clap(long, action)]
    residuals: bool,

    /// Early termination strategy for first-stage search ("none", "distance-adaptive", etc).
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("none"))]
    early_termination: String,

    /// Lambda parameter for early termination (optional, used with distance-adaptive).
    #[clap(long, value_parser)]
    lambda: Option<f32>,

    /// Number of runs for timing.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    num_runs: usize,
}

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

#[derive(Clone, Copy, Debug)]
enum DistanceKind {
    Euclidean,
    DotProduct,
}

/// Load multivector dataset for plain quantizer
fn load_multivec_dataset_plain(
    data_folder: &str,
) -> MultiVectorDataset<PlainMultiVecQuantizer<f32>> {
    let documents_path = Path::new(data_folder).join("documents.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");

    // Load documents as u16 array (shape: n_tokens x token_dim) and reinterpret as f16
    let documents_file = std::fs::File::open(&documents_path).unwrap_or_else(|e| {
        eprintln!(
            "Error opening documents file at {:?}: {}",
            documents_path, e
        );
        std::process::exit(1);
    });
    let documents_reader = std::io::BufReader::new(documents_file);
    let documents_u16: Array2<u16> = Array2::read_npy(documents_reader).unwrap_or_else(|e| {
        eprintln!("Error reading documents array: {}", e);
        std::process::exit(1);
    });

    let (_n_tokens, token_dim) = documents_u16.dim();

    // Reinterpret u16 as f16 and convert to f32
    let documents_raw = documents_u16.into_raw_vec_and_offset().0;
    let mut documents_flat: Vec<f32> = Vec::with_capacity(documents_raw.len());
    for u16_val in documents_raw {
        let f16_val = f16::from_bits(u16_val);
        documents_flat.push(f32::from(f16_val));
    }

    // Load doclens as int32 array (shape: n_docs)
    let doclens_file = std::fs::File::open(&doclens_path).unwrap_or_else(|e| {
        eprintln!("Error opening doclens file at {:?}: {}", doclens_path, e);
        std::process::exit(1);
    });
    let doclens_reader = std::io::BufReader::new(doclens_file);
    let doclens_array: Array1<i32> = Array1::read_npy(doclens_reader).unwrap_or_else(|e| {
        eprintln!("Error reading doclens array: {}", e);
        std::process::exit(1);
    });

    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Build offsets array from doclens
    let mut offsets = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * token_dim);
    }

    // Create encoder
    let encoder = PlainMultiVecQuantizer::new(token_dim);

    MultiVectorDataset::from_raw(documents_flat.into(), offsets.into(), encoder)
}

/// Load multivector dataset for two-level PQ quantizer
fn load_multivec_dataset_pq<const M: usize>(
    data_folder: &str,
) -> MultiVectorDataset<MultiVecTwoLevelProductQuantizer<M, f16>> {
    use std::path::Path;

    let coarse_path = Path::new(data_folder).join("centroids.npy");
    let pq_centroids_path = Path::new(data_folder).join("pq_centroids.npy");
    let residuals_path = Path::new(data_folder).join("residuals.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");
    let assignment_path = Path::new(data_folder).join("index_assignment.npy");
    let norms_path = Path::new(data_folder).join("residual_norms.npy");

    // Load coarse centroids (n_centroids, dim)
    let coarse_file = File::open(&coarse_path).unwrap_or_else(|e| {
        eprintln!("Error opening centroids.npy at {:?}: {}", coarse_path, e);
        std::process::exit(1);
    });
    let coarse_reader = std::io::BufReader::new(coarse_file);
    let coarse_array: Array2<f32> =
        Array2::read_npy(coarse_reader).expect("Cannot read centroids.npy");
    let (n_coarse, token_dim) = coarse_array.dim();

    if token_dim % M != 0 {
        eprintln!(
            "Error: token_dim {} is not divisible by M={} for two-level PQ",
            token_dim, M
        );
        std::process::exit(1);
    }
    let dsub = token_dim / M;

    let mut coarse_ds = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
        PlainDenseQuantizer::new(token_dim),
        n_coarse,
    );
    for row in coarse_array.rows() {
        coarse_ds.push(DenseVectorView::new(row.as_slice().unwrap()));
    }
    let coarse_centroids: DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>> =
        coarse_ds.into();

    // Load PQ centroids (new flat Array1 or legacy Array3)
    let pq_file = File::open(&pq_centroids_path).unwrap_or_else(|e| {
        eprintln!(
            "Error opening pq_centroids.npy at {:?}: {}",
            pq_centroids_path, e
        );
        std::process::exit(1);
    });
    let pq_reader = std::io::BufReader::new(pq_file);
    let expected_pq_len = M * 256 * dsub;
    let pq_flat: Array1<f32> = match Array1::read_npy(pq_reader) {
        Ok(arr1) => {
            if arr1.len() != expected_pq_len {
                eprintln!(
                    "Error: pq_centroids.npy size mismatch: got {}, expected {}",
                    arr1.len(),
                    expected_pq_len
                );
                std::process::exit(1);
            }
            arr1
        }
        Err(err_arr1) => {
            let arr3: Array3<f32> = Array3::read_npy(std::io::BufReader::new(
                File::open(&pq_centroids_path).unwrap_or_else(|e| {
                    eprintln!(
                        "Error opening pq_centroids.npy at {:?}: {}",
                        pq_centroids_path, e
                    );
                    std::process::exit(1);
                }),
            ))
            .unwrap_or_else(|err_arr3| {
                eprintln!(
                    "Error: failed to read pq_centroids.npy as Array1 ({}) or Array3 ({})",
                    err_arr1, err_arr3
                );
                std::process::exit(1);
            });
            let (m_old, ksub_old, dsub_old) = arr3.dim();
            if !(m_old == M && ksub_old == 256 && dsub_old == dsub) {
                eprintln!(
                    "Error: legacy pq_centroids.npy shape mismatch: got [{}, {}, {}], expected [{}, 256, {}]",
                    m_old, ksub_old, dsub_old, M, dsub
                );
                std::process::exit(1);
            }
            Array1::from(arr3.into_iter().collect::<Vec<f32>>())
        }
    };

    let mut pq_centroids: Vec<DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>> =
        Vec::with_capacity(M);
    for m_idx in 0..M {
        let mut ds = PlainDenseDatasetGrowable::with_capacity(PlainDenseQuantizer::new(dsub), 256);
        for k in 0..256 {
            let offset = m_idx * 256 * dsub + k * dsub;
            ds.push(DenseVectorView::new(
                pq_flat.as_slice().unwrap()[offset..offset + dsub].as_ref(),
            ));
        }
        pq_centroids.push(ds.into());
    }

    // Load doclens with dtype fallbacks
    let doclens_file = File::open(&doclens_path).unwrap_or_else(|e| {
        eprintln!("Error opening doclens.npy at {:?}: {}", doclens_path, e);
        std::process::exit(1);
    });
    let doclens_reader = std::io::BufReader::new(doclens_file);
    let doclens_array: Array1<i32> = match Array1::<i32>::read_npy(doclens_reader) {
        Ok(arr) => arr,
        Err(err_i32) => match Array1::<i64>::read_npy(std::io::BufReader::new(
            File::open(&doclens_path).unwrap_or_else(|e| {
                eprintln!("Error opening doclens.npy at {:?}: {}", doclens_path, e);
                std::process::exit(1);
            }),
        )) {
            Ok(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for &v in arr.iter() {
                    if v < 0 || v > i32::MAX as i64 {
                        eprintln!("Error: invalid doc length value {} in doclens.npy", v);
                        std::process::exit(1);
                    }
                    out.push(v as i32);
                }
                Array1::from(out)
            }
            Err(err_i64) => match Array1::<u32>::read_npy(std::io::BufReader::new(
                File::open(&doclens_path).unwrap_or_else(|e| {
                    eprintln!("Error opening doclens.npy at {:?}: {}", doclens_path, e);
                    std::process::exit(1);
                }),
            )) {
                Ok(arr) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for &v in arr.iter() {
                        if v > i32::MAX as u32 {
                            eprintln!("Error: invalid doc length value {} in doclens.npy", v);
                            std::process::exit(1);
                        }
                        out.push(v as i32);
                    }
                    Array1::from(out)
                }
                Err(err_u32) => match Array1::<u64>::read_npy(std::io::BufReader::new(
                    File::open(&doclens_path).unwrap_or_else(|e| {
                        eprintln!("Error opening doclens.npy at {:?}: {}", doclens_path, e);
                        std::process::exit(1);
                    }),
                )) {
                    Ok(arr) => {
                        let mut out = Vec::with_capacity(arr.len());
                        for &v in arr.iter() {
                            if v > i32::MAX as u64 {
                                eprintln!("Error: invalid doc length value {} in doclens.npy", v);
                                std::process::exit(1);
                            }
                            out.push(v as i32);
                        }
                        Array1::from(out)
                    }
                    Err(err_u64) => {
                        eprintln!(
                            "Error: cannot read doclens.npy as i32 ({}) / i64 ({}) / u32 ({}) / u64 ({})",
                            err_i32, err_i64, err_u32, err_u64
                        );
                        std::process::exit(1);
                    }
                },
            },
        },
    };
    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Load residuals (PQ codes)
    let residuals_file = File::open(&residuals_path).unwrap_or_else(|e| {
        eprintln!("Error opening residuals.npy at {:?}: {}", residuals_path, e);
        std::process::exit(1);
    });
    let residuals_reader = std::io::BufReader::new(residuals_file);
    let residuals_array: Array2<u8> =
        Array2::read_npy(residuals_reader).expect("Cannot read residuals.npy");
    let (n_tokens, m_check) = residuals_array.dim();
    assert_eq!(
        m_check, M,
        "residuals.npy has {} subspaces, expected {}",
        m_check, M
    );

    // Load index assignments with dtype fallbacks
    let assignment_file = File::open(&assignment_path).unwrap_or_else(|e| {
        eprintln!(
            "Error opening index_assignment.npy at {:?}: {}",
            assignment_path, e
        );
        std::process::exit(1);
    });
    let assignment_reader = std::io::BufReader::new(assignment_file);
    let assignment_array: Array1<u64> = match Array1::<u64>::read_npy(assignment_reader) {
        Ok(arr) => arr,
        Err(err_u64) => match Array1::<u32>::read_npy(std::io::BufReader::new(
            File::open(&assignment_path).unwrap_or_else(|e| {
                eprintln!(
                    "Error opening index_assignment.npy at {:?}: {}",
                    assignment_path, e
                );
                std::process::exit(1);
            }),
        )) {
            Ok(arr) => arr.mapv(|x| x as u64),
            Err(err_u32) => match Array1::<i64>::read_npy(std::io::BufReader::new(
                File::open(&assignment_path).unwrap_or_else(|e| {
                    eprintln!(
                        "Error opening index_assignment.npy at {:?}: {}",
                        assignment_path, e
                    );
                    std::process::exit(1);
                }),
            )) {
                Ok(arr) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for &v in arr.iter() {
                        if v < 0 {
                            eprintln!("Error: negative centroid id {} in index_assignment.npy", v);
                            std::process::exit(1);
                        }
                        out.push(v as u64);
                    }
                    Array1::from(out)
                }
                Err(err_i64) => match Array1::<i32>::read_npy(std::io::BufReader::new(
                    File::open(&assignment_path).unwrap_or_else(|e| {
                        eprintln!(
                            "Error opening index_assignment.npy at {:?}: {}",
                            assignment_path, e
                        );
                        std::process::exit(1);
                    }),
                )) {
                    Ok(arr) => {
                        let mut out = Vec::with_capacity(arr.len());
                        for &v in arr.iter() {
                            if v < 0 {
                                eprintln!(
                                    "Error: negative centroid id {} in index_assignment.npy",
                                    v
                                );
                                std::process::exit(1);
                            }
                            out.push(v as u64);
                        }
                        Array1::from(out)
                    }
                    Err(err_i32) => {
                        eprintln!(
                            "Error: cannot read index_assignment.npy as u64 ({}) / u32 ({}) / i64 ({}) / i32 ({})",
                            err_u64, err_u32, err_i64, err_i32
                        );
                        std::process::exit(1);
                    }
                },
            },
        },
    };
    assert_eq!(assignment_array.len(), n_tokens);

    // Optional residual norms
    let norms_array: Option<Array1<f32>> = if norms_path.exists() {
        let norms_file = File::open(&norms_path).unwrap_or_else(|e| {
            eprintln!(
                "Error opening residual_norms.npy at {:?}: {}",
                norms_path, e
            );
            std::process::exit(1);
        });
        let norms_reader = std::io::BufReader::new(norms_file);
        let norms_arr: Array1<f32> =
            Array1::read_npy(norms_reader).expect("Cannot read residual_norms.npy");
        if norms_arr.len() != n_tokens {
            eprintln!(
                "Error: residual_norms.npy length mismatch: got {}, expected {}",
                norms_arr.len(),
                n_tokens
            );
            std::process::exit(1);
        }
        Some(norms_arr)
    } else {
        None
    };

    let with_norms = norms_array.is_some();

    // Build the two-level PQ encoder from pretrained centroids
    let quantizer = MultiVecTwoLevelProductQuantizer::<M, f16>::from_pretrained(
        token_dim,
        coarse_centroids,
        pq_centroids,
        with_norms,
    );

    // Build encoded blocked payload per document:
    // [coarse_ids: 4*n][pq_codes: M*n][norms: 2*n (f16) if enabled]
    let bytes_per_token = 4 + M + if with_norms { 2 } else { 0 };
    let mut encoded_data: Vec<u8> = Vec::with_capacity(n_tokens * bytes_per_token);

    let mut token_offset = 0usize;
    for &doclen in doclens_array.iter() {
        let dlen = doclen as usize;

        for i in 0..dlen {
            let coarse_id = assignment_array[token_offset + i] as u32;
            encoded_data.extend(coarse_id.to_le_bytes());
        }
        for i in 0..dlen {
            for m_idx in 0..M {
                encoded_data.push(residuals_array[[token_offset + i, m_idx]]);
            }
        }
        if let Some(ref norms) = norms_array {
            for i in 0..dlen {
                encoded_data
                    .extend_from_slice(&f16::from_f32(norms[token_offset + i]).to_le_bytes());
            }
        }

        token_offset += dlen;
    }

    if token_offset != n_tokens {
        eprintln!(
            "Error: processed token count mismatch: got {}, expected {}",
            token_offset, n_tokens
        );
        std::process::exit(1);
    }

    let mut offsets: Vec<usize> = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * bytes_per_token);
    }

    MultiVectorDataset::from_raw(encoded_data.into_boxed_slice(), offsets.into(), quantizer)
}

fn load_multivec_queries(data_folder: &str) -> Array3<f32> {
    let queries_path = Path::new(data_folder).join("queries.npy");
    let queries_file = std::fs::File::open(&queries_path).unwrap_or_else(|e| {
        eprintln!("Error opening queries file at {:?}: {}", queries_path, e);
        std::process::exit(1);
    });
    let queries_reader = std::io::BufReader::new(queries_file);

    Array3::read_npy(queries_reader).unwrap_or_else(|e| {
        eprintln!("Error reading queries array: {}", e);
        std::process::exit(1);
    })
}

fn read_seismic_queries<D>(path: &str) -> PlainSparseDataset<u16, f32, D>
where
    D: ScalarSparseSupportedDistance,
{
    read_seismic_format::<u16, f32, D>(path).unwrap_or_else(|e| {
        eprintln!("Error reading sparse queries: {:?}", e);
        std::process::exit(1);
    })
}

fn create_search_config(args: &Args) -> HNSWSearchConfiguration {
    let strategy = match args.early_termination.as_str() {
        "distance-adaptive" => {
            let lambda = args.lambda.unwrap_or(0.0);
            EarlyTerminationStrategy::DistanceAdaptive { lambda }
        }
        _ => EarlyTerminationStrategy::None,
    };

    HNSWSearchConfiguration::default()
        .with_ef_search(args.ef_search)
        .with_early_termination(strategy)
}

fn write_results_to_file(output_path: &str, results: &[(f32, usize)], k: usize) {
    let mut file = File::create(output_path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}

fn main() {
    let args: Args = Args::parse();

    // Validate quantizer type
    match args.multivector_quantizer.as_str() {
        "plain" => {
            let metric = parse_metric(&args.distance);
            match metric {
                DistanceKind::Euclidean => {
                    if try_search_plain_f16::<SquaredEuclideanDistance>(&args).is_ok() {
                        return;
                    }
                    search_plain_f32::<SquaredEuclideanDistance>(&args);
                }
                DistanceKind::DotProduct => {
                    if try_search_plain_f16::<DotProduct>(&args).is_ok() {
                        return;
                    }
                    search_plain_f32::<DotProduct>(&args);
                }
            }
        }
        "two-levels" => {
            if args.pq_subspaces.is_none() {
                eprintln!("Error: --pq-subspaces is required for 'two-levels' quantizer");
                std::process::exit(1);
            }
            let metric = parse_metric(&args.distance);
            match metric {
                DistanceKind::Euclidean => match args.pq_subspaces.unwrap() {
                    8 => {
                        if try_search_pq_f16::<8, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<8, SquaredEuclideanDistance>(&args)
                    }
                    16 => {
                        if try_search_pq_f16::<16, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<16, SquaredEuclideanDistance>(&args)
                    }
                    32 => {
                        if try_search_pq_f16::<32, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<32, SquaredEuclideanDistance>(&args)
                    }
                    64 => {
                        if try_search_pq_f16::<64, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<64, SquaredEuclideanDistance>(&args)
                    }
                    m => {
                        eprintln!(
                            "Error: Unsupported number of PQ subspaces: {}. Supported: 8, 16, 32, 64",
                            m
                        );
                        std::process::exit(1);
                    }
                },
                DistanceKind::DotProduct => match args.pq_subspaces.unwrap() {
                    8 => {
                        if try_search_pq_f16::<8, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<8, DotProduct>(&args)
                    }
                    16 => {
                        if try_search_pq_f16::<16, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<16, DotProduct>(&args)
                    }
                    32 => {
                        if try_search_pq_f16::<32, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<32, DotProduct>(&args)
                    }
                    64 => {
                        if try_search_pq_f16::<64, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<64, DotProduct>(&args)
                    }
                    m => {
                        eprintln!(
                            "Error: Unsupported number of PQ subspaces: {}. Supported: 8, 16, 32, 64",
                            m
                        );
                        std::process::exit(1);
                    }
                },
            }
        }
        q => {
            eprintln!(
                "Error: Invalid quantizer type '{}'. Choose 'plain' or 'two-levels'.",
                q
            );
            std::process::exit(1);
        }
    }
}

fn try_search_plain_f16<D>(args: &Args) -> Result<(), String>
where
    D: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f16, D>, Graph> =
        <HNSW<PlainSparseDataset<u16, f16, D>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .map_err(|e| format!("Failed to load f16 index: {:?}", e))?;

    let sparse_queries = read_seismic_queries::<D>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_plain(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        return Err(format!(
            "Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        ));
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                &(),
                args.alpha,
                args.beta,
                args.residuals,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }

    Ok(())
}

fn search_plain_f32<D>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f32, D>, Graph> =
        <HNSW<PlainSparseDataset<u16, f32, D>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error loading HNSW index: {:?}", e);
            std::process::exit(1);
        });

    let sparse_queries = read_seismic_queries::<D>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_plain(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        eprintln!(
            "Error: Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        );
        std::process::exit(1);
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                &(),
                args.alpha,
                args.beta,
                args.residuals,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn try_search_pq_f16<const M: usize, D1>(args: &Args) -> Result<(), String>
where
    D1: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f16, D1>, Graph> =
        <HNSW<PlainSparseDataset<u16, f16, D1>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .map_err(|e| format!("Failed to load f16 index for PQ: {:?}", e))?;

    let sparse_queries = read_seismic_queries::<D1>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_pq::<M>(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        return Err(format!(
            "Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        ));
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                &(),
                args.alpha,
                args.beta,
                args.residuals,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }

    Ok(())
}

fn search_pq_f32<const M: usize, D1>(args: &Args)
where
    D1: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f32, D1>, Graph> =
        <HNSW<PlainSparseDataset<u16, f32, D1>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error loading HNSW index: {:?}", e);
            std::process::exit(1);
        });

    let sparse_queries = read_seismic_queries::<D1>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_pq::<M>(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        eprintln!(
            "Error: Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        );
        std::process::exit(1);
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                &(),
                args.alpha,
                args.beta,
                args.residuals,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}
