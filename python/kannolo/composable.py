"""Composable index construction.

An index is described by independent choices, one per axis of the design space::

    import kannolo

    index = kannolo.build(
        data,                                       # dense, sparse or multi-vector data
        metric="ip",                                # "ip", "l2", or "maxsim" (multi-vector)
        encoder=kannolo.PQ(subspaces=32),                # how vectors are stored
        index=kannolo.HNSW(m=32, graph="compressed"),    # Flat, HNSW or IVF
        rerank=kannolo.Rerank(kannolo.Plain("f16")),          # optional rescoring stage(s)
    )
    distances, ids = index.search(queries, k=10, ef_search=100)
    index.save("my.index")
    index = kannolo.load("my.index")                     # the file records how it was built

The representation (dense, sparse or multi-vector) is inferred from the data; pass
``representation=`` to override it, for example ``kannolo.Sparse(components="u32")``.

A rescoring stage may use a different representation from the first stage, with its own data:
``rerank=kannolo.Rerank(kannolo.TwoLevelPQ(16), data=token_vectors)`` rescores the candidates of a sparse
first stage with multi-vector MaxSim. A list of ``Rerank`` stages is a cascade.

Every combination is compiled ahead of time and specialized, so search runs the same code a
Rust program would. ``kannolo.check(...)`` reports whether a combination exists without building
anything.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, fields, replace
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._kannolo import _ComposedIndex, _compiled_features, _validate_spec

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
]

_METRICS = {"ip": "ip", "dotproduct": "ip", "l2": "l2", "euclidean": "l2", "maxsim": "maxsim"}


# --------------------------------------------------------------------------- representations


@dataclass(frozen=True)
class Dense:
    """Dense vectors: one float per dimension."""

    _kind = "dense"

    def _fields(self) -> dict:
        return {"representation": "dense", "component_type": None}


@dataclass(frozen=True)
class Sparse:
    """Sparse vectors: (component, value) pairs.

    ``components`` is the integer type of the component ids: ``"u16"`` (ids below 65536),
    ``"u32"``, or ``"auto"`` to pick ``"u16"`` whenever the data allows it.
    """

    components: str = "auto"
    _kind = "sparse"

    def __post_init__(self):
        if self.components not in ("auto", "u16", "u32"):
            raise ValueError(f'Sparse(components={self.components!r}): use "u16", "u32" or "auto"')

    def _fields(self) -> dict:
        return {"representation": "sparse", "component_type": self.components}


@dataclass(frozen=True)
class MultiVector:
    """Multi-vector (late-interaction) data: one dense vector per token, scored with MaxSim."""

    _kind = "multivec"

    def _fields(self) -> dict:
        return {"representation": "multivector", "component_type": None}


_Representation = Union[Dense, Sparse, MultiVector]


def _representation_from(name: str, component_type: Optional[str]) -> _Representation:
    if name == "dense":
        return Dense()
    if name == "sparse":
        return Sparse(components=component_type)
    return MultiVector()


# --------------------------------------------------------------------------- encoders


class _Encoder:
    _kind: str

    def _fields(self) -> dict:
        return {"kind": self._kind, **{f.name: getattr(self, f.name) for f in fields(self)}}


@dataclass(frozen=True)
class Plain(_Encoder):
    """Uncompressed values, ``dtype`` "f32", "f16" or "bf16" (dense only). Any representation."""

    dtype: str = "f32"
    _kind = "plain"


@dataclass(frozen=True)
class PQ(_Encoder):
    """Product Quantization with ``subspaces`` one-byte codes per vector (per token for
    multi-vector data). Dense or multi-vector.

    ``subspaces`` must divide the dimension. Dense: 8, 16, 32, 64 or 128; multi-vector: 8, 16,
    32 or 64. For other dense code sizes use ``RaBitQ``, with 1, 2, 4 or 8 bits per component.
    """

    subspaces: int = 32
    _kind = "pq"


@dataclass(frozen=True)
class RaBitQ(_Encoder):
    """RaBitQ with ``bits`` per component: 1 (binary codes) or 2, 4, 8 (RaBitQ-ext). Dense only.

    The dimension must be a multiple of 64. With ``bits=1``, ``search(..., query_bits=b)``
    quantizes the query to ``b`` bits (1..8).
    """

    bits: int = 1
    seed: int = 42
    rotate: bool = True
    _kind = "rabitq"


@dataclass(frozen=True)
class FixedPoint(_Encoder):
    """Values as ``bits``-bit (8 or 16) unsigned fixed-point numbers. Dense or sparse.

    8 bits represent [0, 3.98] in steps of 1/64, 16 bits [0, 8) in steps of 1/8192; values
    outside the range saturate (``build`` warns). Suited to non-negative weights such as
    learned sparse ones. Not suited to signed dense embeddings.
    """

    bits: int = 8
    _kind = "fixed_point"


@dataclass(frozen=True)
class Scalar(_Encoder):
    """Sparse values as ``bits``-bit (8 or 16) codes, one trained step per component."""

    bits: int = 8
    _kind = "scalar"


@dataclass(frozen=True)
class Blocked(_Encoder):
    """Blocked sparse layout (16 ``u16`` components per 64-byte block). Inner product only.

    ``values``: "f16", "scalar_u8" or "scalar_u16".
    """

    values: str = "f16"
    _kind = "blocked"


@dataclass(frozen=True)
class DotPacking8(_Encoder):
    """DotPacking8 sparse layout (bit-packed component gaps). Inner product only.

    ``values``: "fixed_u8", "scalar_u8", "scalar_u16", "f16" or "scalar_4bit". Only
    "scalar_u8" and "scalar_u16" support ``u32`` components.
    """

    values: str = "fixed_u8"
    _kind = "dotpacking8"


@dataclass(frozen=True)
class TwoLevelPQ(_Encoder):
    """Multi-vector two-level PQ: each token is a coarse centroid (one of ``n_coarse``) plus
    PQ codes of its residual, ``subspaces`` bytes. ``subspaces``: 8, 16, 32 or 64."""

    subspaces: int = 16
    n_coarse: int = 1024
    _kind = "two_level_pq"


_ENCODERS = {
    cls._kind: cls
    for cls in (Plain, PQ, RaBitQ, FixedPoint, Scalar, Blocked, DotPacking8, TwoLevelPQ)
}


# --------------------------------------------------------------------------- index structures


@dataclass(frozen=True)
class Flat:
    """Exhaustive scan: exact under the stored encoding."""

    def _fields(self) -> dict:
        return {"kind": "flat"}


@dataclass(frozen=True)
class HNSW:
    """Hierarchical Navigable Small World graph.

    ``graph`` sets how the graph is stored; all three give identical results:
      * "standard": original node order, uncompressed.
      * "permuted": nodes reordered for locality, faster queries.
      * "compressed": reordered and compressed, about half the graph size (needs m <= 128).
    """

    m: int = 32
    ef_construction: int = 200
    graph: str = "standard"

    def _fields(self) -> dict:
        return {"kind": "hnsw", "m": self.m, "ef_construction": self.ef_construction, "graph": self.graph}


@dataclass(frozen=True)
class IVF:
    """Inverted file: k-means into ``n_lists`` clusters; a query scans the ``nprobe`` nearest ones.

    ``centroids`` routes queries to clusters: ``kannolo.Flat()`` (exhaustive) or ``kannolo.HNSW(m, ...)``.
    ``residual`` stores each vector relative to its centroid (dense only). Dense clusters can be
    stored as Plain("f32"/"f16"), PQ or RaBitQ; sparse clusters as Plain("f32"/"f16").
    """

    n_lists: int = 1024
    residual: bool = False
    centroids: Union[Flat, HNSW] = Flat()
    kmeans_epochs: Optional[float] = None
    kmeans_sample_size: Optional[int] = None
    seed: Optional[int] = None
    spherical: bool = False
    n_redo: int = 1
    centroid_nnz_factor: Optional[float] = None

    def _fields(self) -> dict:
        if isinstance(self.centroids, HNSW):
            if self.centroids.graph != "standard":
                raise ValueError("IVF centroids use a standard HNSW graph")
            centroids = {"kind": "hnsw", "m": self.centroids.m, "ef_construction": self.centroids.ef_construction}
        elif isinstance(self.centroids, Flat):
            centroids = {"kind": "flat"}
        else:
            raise TypeError(f"IVF centroids must be kannolo.Flat() or kannolo.HNSW(...), got {self.centroids!r}")
        return {
            "kind": "ivf",
            "n_lists": self.n_lists,
            "residual": self.residual,
            "centroids": centroids,
            "kmeans": {
                "epochs": self.kmeans_epochs,
                "sample_size": self.kmeans_sample_size,
                "seed": self.seed,
                "spherical": self.spherical,
                "n_redo": self.n_redo,
            },
            "centroid_nnz_factor": self.centroid_nnz_factor,
        }


def _index_from(d: dict):
    d = dict(d)
    kind = d.pop("kind")
    if kind == "flat":
        return Flat()
    if kind == "hnsw":
        return HNSW(**d)
    c = d.pop("centroids")
    centroids = Flat() if c["kind"] == "flat" else HNSW(m=c["m"], ef_construction=c["ef_construction"])
    k = d.pop("kmeans")
    return IVF(
        centroids=centroids,
        kmeans_epochs=k["epochs"],
        kmeans_sample_size=k["sample_size"],
        seed=k["seed"],
        spherical=k["spherical"],
        n_redo=k["n_redo"],
        **d,
    )


# --------------------------------------------------------------------------- transforms, stages


@dataclass(frozen=True)
class L1Prune:
    """Sparse first-stage transform: keep, per vector, the largest components that hold
    ``percent``% of its L1 mass. Rescoring stages see the unpruned data; with
    ``Rerank(..., residual=True)`` they store the pruned-away part."""

    percent: float = 50.0

    def _fields(self) -> dict:
        return {"kind": "l1_prune", "percent": float(self.percent)}


@dataclass(frozen=True)
class Rerank:
    """A rescoring stage: rescore the candidates of the previous stage with ``encoder``.

    ``data`` is the collection in this stage's representation (default: the first stage's data,
    untransformed); it must hold the same items, in the same order, as the first stage.
    ``metric`` defaults to the first stage's (``"maxsim"`` for multi-vector data).

    ``residual=True`` stores what the first stage leaves out and scores a candidate as the sum
    of both stages. For sparse data, the stage stores the components ``L1Prune`` dropped. For
    dense data, it stores ``x - x̂``, where ``x̂`` is the first stage's RaBitQ reconstruction.
    Inner product only. A residual RaBitQ stage with the first stage's seed gets a derived
    seed (a different rotation), as in the Rust ``DenseRerankIndex``.
    """

    encoder: _Encoder = Plain("f16")
    data: Any = field(default=None, repr=False, compare=False)
    metric: Optional[str] = None
    representation: Optional[_Representation] = None
    residual: bool = False


# --------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class IndexSpec:
    """A complete description of an index: a first stage, an optional transform of its input,
    and zero or more rescoring stages (``Rerank`` objects without data)."""

    representation: _Representation
    metric: str
    encoder: _Encoder
    index: Union[Flat, HNSW, IVF]
    transform: Optional[L1Prune] = None
    rerank: Tuple[Rerank, ...] = ()

    def to_json(self) -> str:
        rep = self.representation._fields()
        first = {
            "representation": rep["representation"],
            "metric": self.metric,
            "component_type": rep["component_type"],
            "encoder": self.encoder._fields(),
            "index": self.index._fields(),
            "transform": self.transform._fields() if self.transform else None,
        }
        rerank = []
        for r in self.rerank:
            rr = r.representation._fields()
            rerank.append(
                {
                    "representation": rr["representation"],
                    "metric": r.metric,
                    "component_type": rr["component_type"],
                    "encoder": r.encoder._fields(),
                    "residual": r.residual,
                }
            )
        return json.dumps({"first": first, "rerank": rerank})

    @classmethod
    def from_json(cls, text: str) -> "IndexSpec":
        d = json.loads(text)
        f = d["first"]

        def encoder(e):
            e = dict(e)
            return _ENCODERS[e.pop("kind")](**e)

        t = f.get("transform")
        rerank = tuple(
            Rerank(
                encoder=encoder(r["encoder"]),
                metric=r["metric"],
                representation=_representation_from(r["representation"], r["component_type"]),
                residual=r["residual"],
            )
            for r in d.get("rerank", [])
        )
        return cls(
            representation=_representation_from(f["representation"], f["component_type"]),
            metric=f["metric"],
            encoder=encoder(f["encoder"]),
            index=_index_from(f["index"]),
            transform=L1Prune(t["percent"]) if t else None,
            rerank=rerank,
        )

    def __str__(self) -> str:
        parts = [f"{self.representation}", f"metric={self.metric!r}", f"encoder={self.encoder}", f"index={self.index}"]
        if self.transform:
            parts.append(f"transform={self.transform}")
        for r in self.rerank:
            parts.append(f"rerank={r}")
        return ", ".join(parts)


def _metric(metric, representation) -> str:
    if metric is None:
        return "maxsim" if isinstance(representation, MultiVector) else "ip"
    m = _METRICS.get(str(metric).lower())
    if m is None:
        raise ValueError(f"metric={metric!r}: use 'ip' (inner product), 'l2' (squared Euclidean) or 'maxsim'")
    return m


def _resolve_components(rep: _Representation, max_component: Optional[int]) -> _Representation:
    """Replaces Sparse(components="auto") by a concrete type."""
    if not isinstance(rep, Sparse) or rep.components != "auto":
        return rep
    fits_u16 = max_component is None or max_component < 2**16
    return Sparse(components="u16" if fits_u16 else "u32")


def _check_types(encoder, index, transform, rerank):
    if not isinstance(encoder, _Encoder):
        names = ", ".join(c.__name__ for c in _ENCODERS.values())
        raise TypeError(f"encoder must be one of {names}, got {encoder!r}")
    if not isinstance(index, (Flat, HNSW, IVF)):
        raise TypeError(f"index must be kannolo.Flat(), kannolo.HNSW(...) or kannolo.IVF(...), got {index!r}")
    if transform is not None and not isinstance(transform, L1Prune):
        raise TypeError(f"transform must be kannolo.L1Prune(...), got {transform!r}")
    for r in rerank:
        if not isinstance(r, Rerank):
            raise TypeError(f"rerank stages must be kannolo.Rerank(...), got {r!r}")


def _as_stages(rerank) -> Tuple[Rerank, ...]:
    if rerank is None:
        return ()
    if isinstance(rerank, Rerank):
        return (rerank,)
    return tuple(rerank)


def compiled_features() -> frozenset:
    """The parts of the API this build of kannolo contains: a subset of ``"dense"``,
    ``"sparse"``, ``"multivec"`` and ``"ivf"``.

    The published wheels contain all of them. A build from source can leave some out (Cargo
    features of the same names) to be smaller and faster to compile; using a missing part raises
    a ``ValueError`` naming the feature to enable.
    """
    return frozenset(_compiled_features())


def check(
    *,
    representation=None,
    metric=None,
    encoder=None,
    index=None,
    transform=None,
    rerank=None,
) -> IndexSpec:
    """Validates a combination without building anything.

    Returns the resolved spec, or raises ``ValueError`` explaining why the combination does not
    exist. ``Sparse(components="auto")`` is checked as ``u16``.
    """
    rep = _resolve_components(representation if representation is not None else Dense(), None)
    stages = _as_stages(rerank)
    encoder = encoder if encoder is not None else Plain()
    index = index if index is not None else HNSW()
    _check_types(encoder, index, transform, stages)
    m = _metric(metric, rep)
    stages = tuple(_residual_rotation(_resolve_stage(r, rep, m, None), encoder) for r in stages)
    spec = IndexSpec(rep, m, encoder, index, transform, stages)
    _validate_spec(spec.to_json())
    return spec


# --------------------------------------------------------------------------- data normalization


def _as_f32(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).ravel())


def _is_multivec(data) -> bool:
    """(tokens 2-D, lengths 1-D) or a sequence of 2-D token arrays."""
    if isinstance(data, tuple) and len(data) == 2 and np.ndim(data[0]) == 2 and np.ndim(data[1]) == 1:
        return True
    return isinstance(data, (list, tuple)) and len(data) > 0 and all(
        isinstance(d, np.ndarray) and d.ndim == 2 for d in data
    )


def _is_sparse(data) -> bool:
    if hasattr(data, "indptr") and hasattr(data, "indices"):
        return True
    return isinstance(data, (tuple, list)) and len(data) == 3 and np.ndim(data[0]) == 1


def _infer_representation(data) -> _Representation:
    if isinstance(data, (str, os.PathLike)):
        ext = os.path.splitext(os.fspath(data))[1].lower()
        if ext == ".npy":
            return Dense()
        if ext == ".bin":
            return Sparse()
        raise ValueError(
            f"cannot infer the representation of {data!r}: pass representation=kannolo.Dense() or kannolo.Sparse()"
        )
    if _is_multivec(data):
        return MultiVector()
    if _is_sparse(data):
        return Sparse()
    return Dense()


def _sparse_parts(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(components, values, offsets) from a CSR matrix or a 3-tuple."""
    if hasattr(data, "indptr") and hasattr(data, "indices"):  # scipy.sparse, without importing it
        csr = data.tocsr() if getattr(data, "format", "csr") != "csr" else data
        comps, vals, offs = csr.indices, csr.data, csr.indptr
    elif isinstance(data, (tuple, list)) and len(data) == 3:
        comps, vals, offs = data
    else:
        raise TypeError("sparse data must be a scipy.sparse matrix or a (components, values, offsets) tuple")
    comps = np.asarray(comps)
    if comps.size and comps.min() < 0:
        raise ValueError("sparse components must be non-negative")
    return (
        np.ascontiguousarray(comps.ravel(), dtype=np.uint32),
        _as_f32(vals),
        np.ascontiguousarray(np.asarray(offs).ravel(), dtype=np.int64),
    )


def _multivec_parts(data) -> Tuple[np.ndarray, np.ndarray, int]:
    """(values, token_offsets, dim) from (tokens, lengths) or a sequence of (n_tokens, dim)."""
    if isinstance(data, tuple) and len(data) == 2 and np.ndim(data[0]) == 2 and np.ndim(data[1]) == 1:
        tokens = np.asarray(data[0], dtype=np.float32)
        lengths = np.asarray(data[1], dtype=np.int64)
        if lengths.sum() != tokens.shape[0]:
            raise ValueError(f"lengths sum to {lengths.sum()} but there are {tokens.shape[0]} tokens")
    else:
        docs = [np.asarray(d, dtype=np.float32) for d in data]
        dims = {d.shape[1] for d in docs}
        if len(dims) != 1:
            raise ValueError(f"every token must have the same dimension, got {sorted(dims)}")
        tokens = np.concatenate(docs, axis=0)
        lengths = np.array([d.shape[0] for d in docs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return _as_f32(tokens), offsets, int(tokens.shape[1])


def _descriptor(data, rep: _Representation):
    """The data descriptor the native module takes, and the largest sparse component (or None)."""
    if isinstance(data, (str, os.PathLike)):
        path = os.fspath(data)
        if isinstance(rep, Dense):
            return ("npy", path), None
        if isinstance(rep, Sparse):
            return ("bin", path), None
        raise ValueError("multi-vector data cannot be read from a file; pass arrays")
    if isinstance(rep, Dense):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"dense data must be a 2-D array (n, dim), got shape {arr.shape}")
        return ("dense", _as_f32(arr), arr.shape[1]), None
    if isinstance(rep, Sparse):
        c, v, o = _sparse_parts(data)
        return ("sparse", c, v, o), (int(c.max()) if c.size else None)
    vals, offs, dim = _multivec_parts(data)
    return ("multivec", vals, offs, dim), None


_FIXED_POINT_RANGE = {8: 255 / 64, 16: 65535 / 8192}


def _warn_fixed_point_range(encoder, descriptor, stage: str) -> None:
    """Fixed-point values saturate outside their range. Warn when the data exceeds it, so
    accuracy is not lost silently."""
    if not isinstance(encoder, FixedPoint) or descriptor is None or descriptor[0] not in ("dense", "sparse"):
        return
    values = descriptor[1] if descriptor[0] == "dense" else descriptor[2]
    if values.size == 0:
        return
    hi = _FIXED_POINT_RANGE.get(encoder.bits)
    lo_v, hi_v = float(values.min()), float(values.max())
    if hi is not None and (lo_v < 0 or hi_v > hi):
        clipped = float(np.mean((values < 0) | (values > hi)))
        warnings.warn(
            f"{stage}: FixedPoint(bits={encoder.bits}) represents values in [0, {hi:.3g}], but the "
            f"data spans [{lo_v:.3g}, {hi_v:.3g}]; {clipped:.1%} of the values will saturate. "
            "Rescale the data, or use Plain or Scalar.",
            stacklevel=3,
        )


def _resolve_stage(r: Rerank, first_rep, first_metric, max_component) -> Rerank:
    """Fills in a stage's representation, metric and component type."""
    rep = r.representation
    if rep is None:
        rep = _infer_representation(r.data) if r.data is not None else first_rep
    if isinstance(rep, Sparse) and rep.components == "auto":
        if r.data is None and isinstance(first_rep, Sparse):
            rep = first_rep  # rescores the first stage's data: same components
        else:
            rep = _resolve_components(rep, max_component)
    if r.metric is not None:
        metric = _metric(r.metric, rep)
    elif type(rep) is type(first_rep):
        metric = first_metric
    else:
        metric = _metric(None, rep)
    return Rerank(encoder=r.encoder, metric=metric, representation=rep, residual=r.residual)


# The seed transformation `DenseRerankIndex` applies to its residual RaBitQ stage.
_RESIDUAL_SEED_XOR = 0x5265_7369_6475_616C


def _residual_rotation(stage: Rerank, first_encoder) -> Rerank:
    """A residual RaBitQ stage needs a different rotation from the first stage. The residual is
    aligned with the first stage's quantization grid. With the same rotation, RaBitQ's
    assumption that rotated components are concentrated would not hold. When the two seeds
    would coincide, derive the stage's seed as `DenseRerankIndex` does."""
    if (
        stage.residual
        and isinstance(stage.encoder, RaBitQ)
        and isinstance(first_encoder, RaBitQ)
        and stage.encoder.seed == first_encoder.seed
    ):
        seed = (first_encoder.seed ^ _RESIDUAL_SEED_XOR) & (2**64 - 1)
        return replace(stage, encoder=replace(stage.encoder, seed=seed))
    return stage


# --------------------------------------------------------------------------- queries


def _query_batch(queries, rep: _Representation):
    """(batch for the native module, whether it was a single query)."""
    if isinstance(rep, Dense):
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim not in (1, 2):
            raise ValueError(f"dense queries must be 1-D or 2-D, got shape {q.shape}")
        return _as_f32(q), q.ndim == 1
    if isinstance(rep, Sparse):
        if isinstance(queries, np.ndarray):
            raise TypeError(
                "this stage stores sparse vectors; pass (components, values) for one query, "
                "or a scipy.sparse matrix / (components, values, offsets) for a batch"
            )
        single = isinstance(queries, (tuple, list)) and len(queries) == 2
        if single:
            comps, vals = queries
            queries = (comps, vals, np.array([0, np.asarray(comps).size]))
        return _sparse_parts(queries), single
    if isinstance(queries, np.ndarray) and queries.ndim == 2:
        return _multivec_parts([queries]), True
    if not _is_multivec(queries):
        raise TypeError(
            "this stage stores multi-vector data; pass a 2-D (tokens, dim) array for one query, "
            "or a list of them / (tokens, lengths) for a batch"
        )
    return _multivec_parts(queries), False


def _per_stage(value, n: int, name: str) -> list:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} has {len(value)} entries for {n} rescoring stages")
        return list(value)
    return [value] * n


# --------------------------------------------------------------------------- index


def _batch_len(batch, rep, dim: int) -> int:
    if isinstance(rep, Dense):
        return batch.size // dim
    return len(batch[1] if isinstance(rep, MultiVector) else batch[2]) - 1


def _ids_from(f, n: int, what: str) -> np.ndarray:
    """Allowed ids from a mask, an id array or a callable."""
    if callable(f):
        f = np.fromiter((bool(f(i)) for i in range(n)), dtype=bool, count=n)
    a = np.asarray(f)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError(f"{what}: a mask needs one entry per vector ({n}), got shape {a.shape}")
        return np.flatnonzero(a).astype(np.uint64)
    if a.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if not np.issubdtype(a.dtype, np.integer) or a.ndim != 1:
        raise TypeError(f"{what}: pass a boolean mask, an array of ids or a callable, got {a.dtype} {a.shape}")
    if a.min() < 0 or a.max() >= n:
        raise ValueError(f"{what}: ids must be in [0, {n}), got [{a.min()}, {a.max()}]")
    return a.astype(np.uint64)


def _is_per_query(f) -> bool:
    return isinstance(f, (list, tuple)) and len(f) > 0 and all(
        callable(x) or isinstance(x, (np.ndarray, list, tuple)) for x in f
    )


def _filter_arg(f, n: int, n_queries: int):
    """The filter descriptor the native module takes, or None.

    Masks go to the native module as they are (no conversion to ids): a shared mask as one
    array, per-query masks as one 2-D array.
    """
    if f is None:
        return None
    if isinstance(f, np.ndarray) and f.ndim == 2:
        if f.dtype != bool or f.shape != (n_queries, n):
            raise ValueError(f"per-query masks must be a boolean array of shape ({n_queries}, {n}), got {f.dtype} {f.shape}")
        return ("masks", np.ascontiguousarray(f).ravel(), n)
    if _is_per_query(f):
        if len(f) != n_queries:
            raise ValueError(f"{len(f)} per-query filters for {n_queries} queries")
        if all(isinstance(x, np.ndarray) and x.dtype == bool and x.shape == (n,) for x in f):
            return ("masks", np.stack(f).ravel(), n)
        ids = [_ids_from(x, n, f"filter of query {i}") for i, x in enumerate(f)]
        offsets = np.concatenate([[0], np.cumsum([len(x) for x in ids])]).astype(np.int64)
        return ("per_query", np.concatenate(ids) if ids else np.zeros(0, np.uint64), offsets)
    if isinstance(f, np.ndarray) and f.dtype == bool:
        if f.shape != (n,):
            raise ValueError(f"filter: a mask needs one entry per vector ({n}), got shape {f.shape}")
        return ("mask", np.ascontiguousarray(f))
    mask = np.zeros(n, dtype=bool)
    mask[_ids_from(f, n, "filter")] = True
    return ("mask", mask)


class Index:
    """An index built by :func:`build` or read by :func:`load`."""

    def __init__(self, inner: _ComposedIndex):
        self._inner = inner
        self.spec = IndexSpec.from_json(inner.spec_json)

    @property
    def dim(self) -> int:
        """The first stage's dimension (the token dimension for multi-vector data)."""
        return self._inner.dim

    @property
    def nbytes(self) -> int:
        """Bytes held by every stage (datasets, graphs, centroids)."""
        return self._inner.space_usage_bytes

    def __len__(self) -> int:
        return self._inner.n_elements

    def __repr__(self) -> str:
        return f"kannolo.Index(n={len(self)}, dim={self.dim}, {self.spec})"

    def search(
        self,
        queries,
        k: int = 10,
        *,
        rerank_queries=None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        early_exit_threshold: Optional[float] = None,
        query_bits: Optional[int] = None,
        k_candidates=None,
        alpha=None,
        beta=None,
        rerank_query_bits=None,
        filter=None,
        acorn_gamma: Optional[bool] = None,
        num_threads: int = 0,
    ):
        """Returns ``(distances, ids)``.

        ``queries`` are in the first stage's representation. Dense: a 1-D array (one query,
        results of shape ``(k,)``) or a 2-D array (results ``(n_queries, k)``). Sparse:
        ``(components, values)`` for one query, a scipy.sparse matrix or
        ``(components, values, offsets)`` for a batch. Multi-vector: a 2-D ``(tokens, dim)``
        array for one query, a list of them or ``(tokens, lengths)`` for a batch.

        ``rerank_queries`` are the queries of rescoring stages in another representation (a
        dict ``{"dense"|"sparse"|"multivec": queries}`` if there are several).

        First stage: ``ef_search`` and ``early_exit_threshold`` (HNSW, and IVF with HNSW
        centroids), ``nprobe`` (IVF), ``query_bits`` (1-bit RaBitQ). Rescoring stages:
        ``k_candidates`` (default ``max(10k, 100)``), ``alpha`` (candidate pruning), ``beta``
        (early exit) and ``rerank_query_bits``, each a value for every stage or a list with one
        per stage. Unset parameters take their defaults; parameters that do not apply raise an
        error.

        ``filter`` restricts the results to some ids of the collection. Filtered search uses
        ACORN and is available for single-stage HNSW indexes (for now). A filter shared by all
        queries is a boolean mask over the collection, an array of allowed ids, or a callable
        ``id -> bool`` (evaluated once per id before the search). Per-query filters are a list
        of these, or a 2-D boolean array of shape ``(n_queries, len(index))``, the fastest form
        for many queries. ``acorn_gamma`` selects the ACORN-γ neighbor lists (see
        :meth:`build_acorn_gamma`) instead of ACORN-1; by default they are used once built.

        Missing results are padded with distance ``inf`` and id ``-1``. ``num_threads``: 0 uses
        every core, 1 runs serially, n uses n threads.
        """
        spec = self.spec
        batches = {}
        batch, single = _query_batch(queries, spec.representation)
        batches[spec.representation._kind] = batch
        other = {
            r.representation._kind: r.representation
            for r in spec.rerank
            if r.representation._kind not in batches
        }
        if other:
            if rerank_queries is None:
                raise ValueError(f"this index rescores {', '.join(sorted(other))} data: pass rerank_queries=")
            if not isinstance(rerank_queries, dict):
                if len(other) > 1:
                    raise ValueError(f"pass rerank_queries as a dict with keys {sorted(other)}")
                rerank_queries = {next(iter(other)): rerank_queries}
            for kind, rep in other.items():
                if kind not in rerank_queries:
                    raise ValueError(f"rerank_queries has no {kind!r} queries")
                batches[kind], _ = _query_batch(rerank_queries[kind], rep)
        elif rerank_queries is not None:
            raise ValueError("rerank_queries applies to rescoring stages in another representation")

        n_stages = len(spec.rerank)
        given = any(v is not None for v in (k_candidates, alpha, beta, rerank_query_bits))
        if n_stages == 0 and given:
            raise ValueError("k_candidates, alpha, beta and rerank_query_bits apply to indexes with rescoring stages")
        rerank = list(
            zip(
                [int(v) if v is not None else 0 for v in _per_stage(k_candidates, n_stages, "k_candidates")],
                _per_stage(alpha, n_stages, "alpha"),
                _per_stage(beta, n_stages, "beta"),
                _per_stage(rerank_query_bits, n_stages, "rerank_query_bits"),
            )
        )
        n_queries = _batch_len(batch, spec.representation, self.dim)
        filter_arg = _filter_arg(filter, len(self), n_queries)
        n, dist, ids = self._inner.search(
            batches, int(k), ef_search, early_exit_threshold, query_bits, nprobe, rerank, int(num_threads),
            filter_arg, acorn_gamma,
        )
        dist, ids = dist.reshape(n, k), ids.reshape(n, k)
        return (dist[0], ids[0]) if single else (dist, ids)

    def build_acorn_gamma(self, gamma: int = 3) -> None:
        """Builds the ACORN-γ neighbor lists of a single-stage HNSW index. Each node keeps up to
        ``max(gamma * m, 2m)`` two-hop neighbors. Filtered search then skips filtered-out nodes
        without expanding them, which keeps recall up under very selective filters. A larger
        ``gamma`` uses more memory and build time; 2-4 is a good starting range. The lists stay
        in memory: ``save`` does not write them."""
        self._inner.build_acorn_gamma(int(gamma))

    @property
    def has_acorn_gamma(self) -> bool:
        """Whether :meth:`build_acorn_gamma` has run (filtered search then uses ACORN-γ)."""
        return self._inner.has_acorn_gamma

    def save(self, path) -> None:
        """Writes the index; :func:`load` reads it back with no other argument."""
        self._inner.save(os.fspath(path))


def build(
    data,
    *,
    metric: Optional[str] = None,
    encoder: Optional[_Encoder] = None,
    index=None,
    representation: Optional[_Representation] = None,
    transform: Optional[L1Prune] = None,
    rerank: Union[Rerank, Sequence[Rerank], None] = None,
) -> Index:
    """Builds an index over ``data``.

    ``data`` is one of:
      * a 2-D float array ``(n, dim)`` (dense);
      * a scipy.sparse matrix, or a ``(components, values, offsets)`` CSR-style tuple (sparse);
      * a list of 2-D ``(n_tokens, dim)`` arrays, or ``(tokens, lengths)`` (multi-vector);
      * a path: ``.npy`` (dense float32) or ``.bin`` (sparse, seismic format).

    ``metric`` defaults to ``"ip"`` (``"maxsim"`` for multi-vector data), ``encoder`` to
    ``Plain()``, ``index`` to ``HNSW()``. ``transform`` (``L1Prune``) applies to the first
    stage's input only. ``rerank`` adds rescoring stages (a list is a cascade).
    """
    rep = representation if representation is not None else _infer_representation(data)
    encoder = encoder if encoder is not None else Plain()
    index = index if index is not None else HNSW()
    stages = _as_stages(rerank)
    _check_types(encoder, index, transform, stages)
    m = _metric(metric, rep)

    first_desc, max_c = _descriptor(data, rep)
    if isinstance(rep, Sparse) and rep.components == "auto":
        # A file's components are only known once read: "auto" means u16 for files.
        rep = _resolve_components(rep, max_c)

    if transform is None:
        _warn_fixed_point_range(encoder, first_desc, "first stage")
    descriptors = [first_desc]
    resolved = []
    for r in stages:
        if r.data is None:
            descriptors.append(None)
            if not r.residual:
                _warn_fixed_point_range(r.encoder, first_desc, "rescoring stage")
            resolved.append(_resolve_stage(r, rep, m, None))
        else:
            r_rep = r.representation or _infer_representation(r.data)
            desc, r_max = _descriptor(r.data, r_rep)
            _warn_fixed_point_range(r.encoder, desc, "rescoring stage")
            descriptors.append(desc)
            resolved.append(_resolve_stage(replace(r, representation=r_rep), rep, m, r_max))

    resolved = [_residual_rotation(r, encoder) for r in resolved]
    spec = IndexSpec(rep, m, encoder, index, transform, tuple(resolved))
    return Index(_ComposedIndex.build(spec.to_json(), descriptors))


def load(path) -> Index:
    """Reads an index written by :meth:`Index.save`."""
    return Index(_ComposedIndex.load(os.fspath(path)))


def compose(index: Index, rerank: Union[Rerank, Sequence[Rerank]]) -> Index:
    """A new index: ``index`` followed by rescoring stage(s) built from in-memory data.

    ``index`` is shared without copying and is left unchanged, so one built or loaded first
    stage can be composed with several rescoring datasets. Each ``Rerank`` needs its ``data``, the collection
    in that stage's representation (the input ``build`` takes), holding the same items in the
    same order as ``index``. For ``residual=True`` it is the first stage's original collection,
    from which the residual is computed as ``build`` would.

    For example, with first stages and datasets saved separately::

        first = kannolo.load("sparse_hnsw.index")
        index = kannolo.compose(first, kannolo.Rerank(kannolo.Plain("f16"), data=(tokens, lengths)))
    """
    stages = _as_stages(rerank)
    if not stages:
        raise ValueError("compose needs at least one kannolo.Rerank(...) stage")
    for r in stages:
        if not isinstance(r, Rerank):
            raise TypeError(f"rerank stages must be kannolo.Rerank(...), got {r!r}")
        if r.data is None:
            raise ValueError("kannolo.compose builds each rescoring stage from its data: pass Rerank(..., data=...)")
    spec = index.spec
    descriptors, resolved = [], []
    for r in stages:
        rep = spec.representation if r.residual else (r.representation or _infer_representation(r.data))
        desc, max_c = _descriptor(r.data, rep)
        if not r.residual:
            _warn_fixed_point_range(r.encoder, desc, "rescoring stage")
        stage = _resolve_stage(replace(r, representation=rep), spec.representation, spec.metric, max_c)
        resolved.append(_residual_rotation(stage, spec.encoder))
        descriptors.append(desc)
    new_spec = replace(spec, rerank=spec.rerank + tuple(resolved))
    return Index(_ComposedIndex.compose(index._inner, new_spec.to_json(), descriptors))
