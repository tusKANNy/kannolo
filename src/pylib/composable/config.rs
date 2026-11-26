//! The index configuration: one value per choice of the design space.
//!
//! An [`IndexConfig`] is a first stage (representation, metric, encoder, index structure, and an
//! optional transform of its input) followed by zero or more rescoring stages, each with its
//! own representation, metric and encoder. Zero rescoring stages is a single-stage index; one
//! is a two-stage index; more is a cascade (Table 7 of the kANNolo paper).
//!
//! Python builds the configuration from small composable objects (`kannolo.IndexSpec`) and
//! passes it to Rust as JSON. The same JSON is the header of every saved index, so each file
//! describes its own contents. [`IndexConfig::validate`] is the only code that checks which
//! combinations exist, and [`super::build_load`] relies on it.

use serde::{Deserialize, Serialize};

use crate::graph::neighbors::MAX_NEIGHBORS_PER_NODE;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Representation {
    Dense,
    Sparse,
    Multivector,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Metric {
    L2,
    Ip,
    /// Late interaction: the sum over query tokens of the best inner product with any
    /// document token. Multi-vector only.
    Maxsim,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ComponentType {
    U16,
    U32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ValueDtype {
    F32,
    F16,
    Bf16,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BlockedValues {
    F16,
    ScalarU8,
    ScalarU16,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Dp8Values {
    FixedU8,
    ScalarU8,
    ScalarU16,
    F16,
    #[serde(rename = "scalar_4bit")]
    Scalar4Bit,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum GraphKind {
    Standard,
    Permuted,
    Compressed,
}

/// How vectors are stored (and therefore scored).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum EncoderConfig {
    /// Uncompressed values (a cast for `f16`/`bf16`). Any representation.
    Plain { dtype: ValueDtype },
    /// Product Quantization: dense, or per token for multi-vector.
    Pq { subspaces: usize },
    /// RaBitQ, dense only: `bits == 1` is the binary code, `2 | 4 | 8` is RaBitQ-ext.
    Rabitq { bits: u32, seed: u64, rotate: bool },
    /// Values as fixed-point integers of `bits` bits. Dense or sparse.
    FixedPoint { bits: u32 },
    /// Sparse values as `bits`-bit codes with one trained step per component.
    Scalar { bits: u32 },
    /// Sparse blocked layout (64-byte blocks of 16 `u16` components).
    Blocked { values: BlockedValues },
    /// Sparse DotPacking8 layout (bit-packed component gaps).
    #[serde(rename = "dotpacking8")]
    DotPacking8 { values: Dp8Values },
    /// Multi-vector two-level PQ: a coarse centroid per token plus PQ codes of its residual.
    TwoLevelPq { subspaces: usize, n_coarse: usize },
}

/// The centroid index of an IVF.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum CentroidIndexConfig {
    Flat,
    Hnsw { m: usize, ef_construction: usize },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct KMeansConfig {
    pub epochs: Option<f64>,
    pub sample_size: Option<usize>,
    pub seed: Option<u64>,
    pub spherical: bool,
    pub n_redo: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum IndexStructureConfig {
    /// Exhaustive scan.
    Flat,
    Hnsw {
        m: usize,
        ef_construction: usize,
        graph: GraphKind,
    },
    Ivf {
        n_lists: usize,
        /// Store centroid-relative residuals instead of the vectors (dense only).
        residual: bool,
        centroids: CentroidIndexConfig,
        kmeans: KMeansConfig,
        /// Sparse only: prune each centroid to this many times its cluster's mean nonzeros.
        centroid_nnz_factor: Option<f32>,
    },
}

/// A transform of the first stage's input; the rescoring stages see the original data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum TransformConfig {
    /// Sparse only: keep, per vector, the largest components holding `percent`% of its L1 mass.
    L1Prune { percent: f32 },
}

/// The first stage: an index over (possibly transformed) data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StageConfig {
    pub representation: Representation,
    pub metric: Metric,
    /// Sparse only: the integer type of the component ids.
    #[serde(default)]
    pub component_type: Option<ComponentType>,
    pub encoder: EncoderConfig,
    pub index: IndexStructureConfig,
    #[serde(default)]
    pub transform: Option<TransformConfig>,
}

/// A rescoring stage: a dataset that rescores the candidates of the stage before it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RerankConfig {
    pub representation: Representation,
    pub metric: Metric,
    #[serde(default)]
    pub component_type: Option<ComponentType>,
    pub encoder: EncoderConfig,
    /// Store what the first stage leaves out: the pruned-away components (sparse), or `x - x̂`
    /// for the first stage's RaBitQ reconstruction `x̂` (dense). A candidate's score is then the
    /// sum of the scores of the two stages. Inner product only.
    #[serde(default)]
    pub residual: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct IndexConfig {
    pub first: StageConfig,
    #[serde(default)]
    pub rerank: Vec<RerankConfig>,
}

/// Subspace counts compiled into the bindings for dense [`EncoderConfig::Pq`]. Each count is
/// compiled separately in every role (HNSW, Flat, rescoring, IVF clusters), so the set is
/// kept small. With ten counts, PQ took about a quarter of the build time and a third of the
/// library size.
pub(crate) const PQ_SUBSPACES: &[usize] = &[8, 16, 32, 64, 128];
/// Subspace counts compiled in for multi-vector PQ and two-level PQ (per token; must be a
/// multiple of 4 and divide the token dimension).
pub(crate) const MV_PQ_SUBSPACES: &[usize] = &[8, 16, 32, 64];

/// The fields of a stage that determine its stored encoding. First and rescoring stages share
/// them.
pub(crate) struct EncodingConfig<'a> {
    pub representation: Representation,
    pub metric: Metric,
    pub component_type: Option<ComponentType>,
    pub encoder: &'a EncoderConfig,
}

impl StageConfig {
    pub(crate) fn encoding(&self) -> EncodingConfig<'_> {
        EncodingConfig {
            representation: self.representation,
            metric: self.metric,
            component_type: self.component_type,
            encoder: &self.encoder,
        }
    }
}

impl RerankConfig {
    pub(crate) fn encoding(&self) -> EncodingConfig<'_> {
        EncodingConfig {
            representation: self.representation,
            metric: self.metric,
            component_type: self.component_type,
            encoder: &self.encoder,
        }
    }
}

fn encoder_name(e: &EncoderConfig) -> &'static str {
    match e {
        EncoderConfig::Plain { .. } => "Plain",
        EncoderConfig::Pq { .. } => "PQ",
        EncoderConfig::Rabitq { .. } => "RaBitQ",
        EncoderConfig::FixedPoint { .. } => "FixedPoint",
        EncoderConfig::Scalar { .. } => "Scalar",
        EncoderConfig::Blocked { .. } => "Blocked",
        EncoderConfig::DotPacking8 { .. } => "DotPacking8",
        EncoderConfig::TwoLevelPq { .. } => "TwoLevelPQ",
    }
}

/// Returns an error unless this build compiled `feature`. The error says how to get a build that
/// has it.
fn require_feature(enabled: bool, feature: &str, what: &str) -> Result<(), String> {
    if enabled {
        return Ok(());
    }
    Err(format!(
        "this build of kannolo was compiled without {what}: rebuild it with the `{feature}` \
         Cargo feature (for example `maturin build --release --features python,{feature}`). \
         kannolo.compiled_features() lists what this build has"
    ))
}

impl EncodingConfig<'_> {
    /// Checks one (representation, metric, component type, encoder) choice.
    pub(crate) fn validate(&self) -> Result<(), String> {
        use EncoderConfig as E;
        use Representation as R;

        match self.representation {
            R::Dense => require_feature(cfg!(feature = "dense"), "dense", "dense data")?,
            R::Sparse => require_feature(cfg!(feature = "sparse"), "sparse", "sparse data")?,
            R::Multivector => {
                require_feature(cfg!(feature = "multivec"), "multivec", "multi-vector data")?
            }
        }

        match (self.representation, self.component_type) {
            (R::Sparse, None) => {
                return Err("sparse data needs a component_type (\"u16\" or \"u32\")".into());
            }
            (R::Dense | R::Multivector, Some(_)) => {
                return Err("component_type applies to sparse data only".into());
            }
            _ => {}
        }
        match (self.representation, self.metric) {
            (R::Multivector, Metric::Maxsim) | (R::Dense | R::Sparse, Metric::L2 | Metric::Ip) => {}
            (R::Multivector, _) => {
                return Err("multi-vector data is scored with metric=\"maxsim\"".into());
            }
            (_, Metric::Maxsim) => {
                return Err("metric=\"maxsim\" applies to multi-vector data only".into());
            }
        }

        let name = encoder_name(self.encoder);
        let choices = match self.representation {
            R::Dense => "Plain, FixedPoint, PQ, RaBitQ",
            R::Sparse => "Plain, FixedPoint, Scalar, Blocked, DotPacking8",
            R::Multivector => "Plain, PQ, TwoLevelPQ",
        };
        let repr = match self.representation {
            R::Dense => "dense",
            R::Sparse => "sparse",
            R::Multivector => "multi-vector",
        };
        let wrong = || {
            Err(format!(
                "{name} is not available for {repr} data; choose one of: {choices}"
            ))
        };
        match (self.encoder, self.representation) {
            (E::Plain { dtype }, R::Sparse) if *dtype == ValueDtype::Bf16 => {
                return Err("Plain(\"bf16\") is available for dense data only".into());
            }
            (E::Plain { dtype }, R::Multivector) if *dtype == ValueDtype::Bf16 => {
                return Err("Plain(\"bf16\") is available for dense data only".into());
            }
            (E::Plain { .. }, _) => {}
            (E::Pq { subspaces }, R::Dense) => {
                if !PQ_SUBSPACES.contains(subspaces) {
                    return Err(format!(
                        "PQ(subspaces={subspaces}) is not compiled in. Choose one of \
                         {PQ_SUBSPACES:?}, or use RaBitQ(bits=1, 2, 4 or 8) for other code sizes: \
                         1, 2, 4 or 8 bits per component, no training, and a dimension that is \
                         a multiple of 64"
                    ));
                }
            }
            (E::Pq { subspaces }, R::Multivector)
            | (E::TwoLevelPq { subspaces, .. }, R::Multivector) => {
                if !MV_PQ_SUBSPACES.contains(subspaces) {
                    return Err(format!(
                        "{name}(subspaces={subspaces}) for multi-vector data is not compiled in; \
                         choose one of {MV_PQ_SUBSPACES:?}"
                    ));
                }
                if let E::TwoLevelPq { n_coarse, .. } = self.encoder
                    && *n_coarse == 0
                {
                    return Err("TwoLevelPQ: n_coarse must be at least 1".into());
                }
            }
            (E::Rabitq { bits, .. }, R::Dense) => {
                if !matches!(bits, 1 | 2 | 4 | 8) {
                    return Err(format!("RaBitQ(bits={bits}): bits must be 1, 2, 4 or 8"));
                }
            }
            (E::FixedPoint { bits }, R::Dense | R::Sparse) | (E::Scalar { bits }, R::Sparse) => {
                if !matches!(bits, 8 | 16) {
                    return Err(format!("{name}: bits must be 8 or 16"));
                }
            }
            (E::Blocked { .. } | E::DotPacking8 { .. }, R::Sparse) => {
                if self.metric != Metric::Ip {
                    return Err(format!(
                        "{name} scores with inner product only; use metric=\"ip\""
                    ));
                }
                let u32_ok = matches!(
                    self.encoder,
                    E::DotPacking8 {
                        values: Dp8Values::ScalarU8 | Dp8Values::ScalarU16
                    }
                );
                if self.component_type == Some(ComponentType::U32) && !u32_ok {
                    return Err(format!(
                        "{name} stores u16 components only (DotPacking8 with scalar_u8/scalar_u16 \
                         values also supports u32)"
                    ));
                }
            }
            _ => return wrong(),
        }
        Ok(())
    }

    /// Checks that need the data dimensionality, run once it is known at build time.
    pub(crate) fn validate_dim(&self, dim: usize) -> Result<(), String> {
        match *self.encoder {
            EncoderConfig::Pq { subspaces } | EncoderConfig::TwoLevelPq { subspaces, .. }
                if !dim.is_multiple_of(subspaces) =>
            {
                Err(format!(
                    "{}(subspaces={subspaces}) must divide the vector dimension, got dim={dim}",
                    encoder_name(self.encoder)
                ))
            }
            EncoderConfig::Rabitq { .. } if !dim.is_multiple_of(64) => Err(format!(
                "RaBitQ needs a vector dimension that is a multiple of 64, got dim={dim}"
            )),
            _ => Ok(()),
        }
    }
}

impl IndexConfig {
    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        let spec: IndexConfig =
            serde_json::from_str(json).map_err(|e| format!("invalid spec: {e}"))?;
        spec.validate()?;
        Ok(spec)
    }

    pub(crate) fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a IndexConfig always serializes")
    }

    /// Rejects every combination the bindings do not compile. The dispatch in `build_load` relies
    /// on this having run: it never sees a combination rejected here.
    pub(crate) fn validate(&self) -> Result<(), String> {
        let first = &self.first;
        first.encoding().validate()?;
        self.validate_index()?;

        if let Some(TransformConfig::L1Prune { percent }) = &first.transform {
            if first.representation != Representation::Sparse {
                return Err("L1Prune applies to sparse data only".into());
            }
            if !(0.0 < *percent && *percent <= 100.0) {
                return Err(format!(
                    "L1Prune(percent={percent}): percent must be in (0, 100]"
                ));
            }
        }

        for (i, r) in self.rerank.iter().enumerate() {
            r.encoding()
                .validate()
                .map_err(|e| format!("rescoring stage {}: {e}", i + 1))?;
            if r.residual {
                self.validate_residual(i, r)
                    .map_err(|e| format!("rescoring stage {} (residual): {e}", i + 1))?;
            }
        }
        Ok(())
    }

    fn validate_index(&self) -> Result<(), String> {
        let first = &self.first;
        match &first.index {
            IndexStructureConfig::Flat => Ok(()),
            IndexStructureConfig::Hnsw {
                m,
                ef_construction,
                graph,
            } => {
                check_hnsw(*m, *ef_construction)?;
                if *graph == GraphKind::Compressed && 2 * m > MAX_NEIGHBORS_PER_NODE {
                    return Err(format!(
                        "HNSW(graph=\"compressed\") stores at most {MAX_NEIGHBORS_PER_NODE} \
                         neighbors per node, but m={m} gives 2*m={} on the ground level; use \
                         m <= {} or another graph",
                        2 * m,
                        MAX_NEIGHBORS_PER_NODE / 2
                    ));
                }
                Ok(())
            }
            IndexStructureConfig::Ivf {
                n_lists,
                residual,
                centroids,
                kmeans,
                centroid_nnz_factor,
            } => {
                require_feature(cfg!(feature = "ivf"), "ivf", "IVF indexes")?;
                if *n_lists == 0 {
                    return Err("IVF: n_lists must be at least 1".into());
                }
                if kmeans.n_redo == 0 {
                    return Err("IVF: kmeans n_redo must be at least 1".into());
                }
                if let CentroidIndexConfig::Hnsw { m, ef_construction } = centroids {
                    check_hnsw(*m, *ef_construction)?;
                }
                use EncoderConfig as E;
                match first.representation {
                    Representation::Dense => {
                        if centroid_nnz_factor.is_some() {
                            return Err(
                                "IVF: centroid_nnz_factor applies to sparse data only".into()
                            );
                        }
                        match first.encoder {
                            E::Plain {
                                dtype: ValueDtype::F32 | ValueDtype::F16,
                            }
                            | E::Pq { .. }
                            | E::Rabitq { .. } => Ok(()),
                            _ => Err("IVF over dense data stores clusters as Plain(\"f32\"), \
                                      Plain(\"f16\"), PQ or RaBitQ"
                                .into()),
                        }
                    }
                    Representation::Sparse => {
                        if *residual {
                            return Err("IVF: residual clusters apply to dense data only (a \
                                        sparse residual has the nonzeros of both the vector and \
                                        its centroid)"
                                .into());
                        }
                        if kmeans.spherical {
                            return Err("IVF: spherical k-means applies to dense data only".into());
                        }
                        if first.metric != Metric::Ip {
                            return Err("IVF over sparse data reports inner product; use \
                                        metric=\"ip\""
                                .into());
                        }
                        if first.component_type != Some(ComponentType::U16) {
                            return Err("IVF over sparse data stores u16 components".into());
                        }
                        match first.encoder {
                            E::Plain {
                                dtype: ValueDtype::F32 | ValueDtype::F16,
                            } => Ok(()),
                            _ => Err("IVF over sparse data stores clusters as Plain(\"f32\") or \
                                      Plain(\"f16\")"
                                .into()),
                        }
                    }
                    Representation::Multivector => {
                        Err("IVF is not available for multi-vector data; use Flat or HNSW".into())
                    }
                }
            }
        }
    }

    fn validate_residual(&self, i: usize, r: &RerankConfig) -> Result<(), String> {
        let first = &self.first;
        if i != 0 {
            return Err("only the first rescoring stage can store residuals".into());
        }
        if r.representation != first.representation {
            return Err("a residual has the representation of the first stage".into());
        }
        if first.metric != Metric::Ip || r.metric != Metric::Ip {
            return Err(
                "residual scores add up only under inner product; use metric=\"ip\" \
                        in both stages"
                    .into(),
            );
        }
        match first.representation {
            Representation::Sparse => {
                if first.transform.is_none() {
                    return Err(
                        "a sparse residual holds the components that L1 pruning removes: build \
                                the first stage with transform=L1Prune(...)"
                            .into(),
                    );
                }
                if r.component_type != first.component_type {
                    return Err("the residual uses the first stage's component type".into());
                }
                Ok(())
            }
            Representation::Dense => {
                if !matches!(first.encoder, EncoderConfig::Rabitq { .. }) {
                    return Err("a dense residual is x - x̂ for the first stage's RaBitQ \
                                reconstruction x̂: use a RaBitQ first-stage encoder"
                        .into());
                }
                if matches!(first.index, IndexStructureConfig::Ivf { .. }) {
                    return Err(
                        "an IVF first stage encodes vectors against their centroids; use \
                                a Flat or HNSW first stage"
                            .into(),
                    );
                }
                if let (
                    EncoderConfig::Rabitq { seed: s1, .. },
                    EncoderConfig::Rabitq { seed: s2, .. },
                ) = (&first.encoder, &r.encoder)
                    && s1 == s2
                {
                    return Err(
                        "a RaBitQ residual stage needs a different seed from the first stage. \
                                The residual is aligned with the first stage's quantization grid, \
                                and the same rotation breaks RaBitQ's assumption that rotated \
                                components are concentrated"
                            .into(),
                    );
                }
                Ok(())
            }
            Representation::Multivector => {
                Err("residual rescoring is available for dense and sparse data".into())
            }
        }
    }
}

fn check_hnsw(m: usize, ef_construction: usize) -> Result<(), String> {
    if m < 2 {
        return Err(format!("HNSW(m={m}): m must be at least 2"));
    }
    if ef_construction == 0 {
        return Err("HNSW: ef_construction must be at least 1".into());
    }
    Ok(())
}

/// The parts of the composable API this build compiled.
pub(crate) fn compiled_features() -> Vec<&'static str> {
    [
        ("dense", cfg!(feature = "dense")),
        ("sparse", cfg!(feature = "sparse")),
        ("multivec", cfg!(feature = "multivec")),
        ("ivf", cfg!(feature = "ivf")),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect()
}
