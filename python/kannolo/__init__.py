"""kANNolo: approximate nearest neighbor search over dense, sparse and multi-vector data.

``kannolo.build`` builds an index from independent choices (representation, metric, encoder,
index structure, rescoring stages); ``kannolo.load`` reads one back. See
``kannolo.composable`` and docs/PythonUsage.md.
"""

import warnings as _warnings

from ._kannolo import _build_profile
from .composable import (
    HNSW,
    IVF,
    PQ,
    Blocked,
    Dense,
    DotPacking8,
    FixedPoint,
    Flat,
    Index,
    IndexSpec,
    L1Prune,
    MultiVector,
    Plain,
    RaBitQ,
    Rerank,
    Scalar,
    Sparse,
    TwoLevelPQ,
    build,
    check,
    compiled_features,
    compose,
    load,
)

__all__ = [
    "Dense",
    "Sparse",
    "MultiVector",
    "Plain",
    "PQ",
    "RaBitQ",
    "FixedPoint",
    "Scalar",
    "Blocked",
    "DotPacking8",
    "TwoLevelPQ",
    "Flat",
    "HNSW",
    "IVF",
    "L1Prune",
    "Rerank",
    "IndexSpec",
    "Index",
    "build",
    "load",
    "check",
    "compiled_features",
    "compose",
    "build_profile",
]

#: The Cargo profile the native library was built with: "release" (the published wheels) or
#: "dev-python" (a development build: faster to compile, about 2% slower search).
build_profile = _build_profile()
if build_profile != "release":
    _warnings.warn(
        f"this kannolo was built with the {build_profile!r} Cargo profile, whose search is about 2% "
        "slower than a release build: do not use it for benchmarks. Rebuild with "
        "`maturin develop --release` (or install the published wheel).",
        stacklevel=2,
    )
