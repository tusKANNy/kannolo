"""Every composition of Table 7 of the kANNolo paper (https://arxiv.org/abs/2611.03927, 
\"Index compositions expressible with Vectorium and kANNolo\"), built through kannolo.build.

Each row is built on small synthetic data, searched, checked against an exact ground truth,
then saved, loaded and searched again (results must be identical).

Rescoring stages are checked with an exact property that does not depend on the quality of the
first stage. With a Flat first stage and k_candidates = n, every item is a candidate. The results
must then be the exact top-k under the rescoring stage's own scoring.

Run with `pytest python/tests` or `python python/tests/test_table7.py`.
"""

import os
import tempfile

import numpy as np

import kannolo

K = 10


# ----------------------------------------------------------------------------- data


def clustered(n, dim, seed, n_clusters=40):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    return (centers[rng.integers(n_clusters, size=n)] + 0.4 * rng.normal(size=(n, dim))).astype(np.float32)


def sparse_from_dense(x, keep):
    """CSR arrays of the `keep` largest entries per row, plus the dense matrix they represent.
    Values are shifted and scaled into (0, 3], like learned sparse weights and within the range
    of FixedPoint(8)."""
    x = x - x.min() + 0.01
    x = (3.0 * x / x.max()).astype(np.float32)
    idx = np.argsort(-x, axis=1)[:, :keep]
    dense = np.zeros_like(x)
    rows = np.arange(len(x))[:, None]
    dense[rows, idx] = x[rows, idx]
    comps = np.sort(idx, axis=1)
    vals = dense[rows, comps]
    offs = np.arange(0, len(x) * keep + 1, keep)
    return (comps.ravel().astype(np.int64), vals.ravel(), offs), dense


def multivec(n, dim, seed, tokens=(4, 12)):
    """Documents of a few tokens each, drawn around per-document topics."""
    rng = np.random.default_rng(seed)
    base = clustered(n, dim, seed)
    docs = []
    for i in range(n):
        t = rng.integers(*tokens)
        docs.append((base[i] + 0.5 * rng.normal(size=(t, dim))).astype(np.float32))
    return docs


N, NQ, DIM = 3000, 60, 128
DENSE = clustered(N, DIM, 0)
DENSE_Q = clustered(NQ, DIM, 1)
SPARSE, SPARSE_DENSE = sparse_from_dense(clustered(N, 1000, 2), 40)
SPARSE_Q, SPARSE_Q_DENSE = sparse_from_dense(clustered(NQ, 1000, 3), 40)
MV_DIM = 64
MV = multivec(N, MV_DIM, 4)
MV_Q = multivec(NQ, MV_DIM, 5, tokens=(3, 8))


# ----------------------------------------------------------------------------- ground truth


def topk(scores, k=K):
    """Ids of the k smallest scores per row."""
    return np.argsort(scores, axis=1, kind="stable")[:, :k]


def dense_scores(data, queries, metric):
    if metric == "ip":
        return -(queries @ data.T)
    return (queries**2).sum(1)[:, None] - 2 * queries @ data.T + (data**2).sum(1)[None, :]


def maxsim_scores(docs, queries):
    tokens = np.concatenate(docs)
    starts = np.concatenate([[0], np.cumsum([len(d) for d in docs])[:-1]])
    out = np.empty((len(queries), len(docs)), dtype=np.float32)
    for i, q in enumerate(queries):
        sims = q @ tokens.T  # (q_tokens, all doc tokens)
        out[i] = np.maximum.reduceat(sims, starts, axis=1).sum(0)
    return -out


TRUTH = {
    ("dense", "ip"): topk(dense_scores(DENSE, DENSE_Q, "ip")),
    ("dense", "l2"): topk(dense_scores(DENSE, DENSE_Q, "l2")),
    ("sparse", "ip"): topk(dense_scores(SPARSE_DENSE, SPARSE_Q_DENSE, "ip")),
    ("sparse", "l2"): topk(dense_scores(SPARSE_DENSE, SPARSE_Q_DENSE, "l2")),
    ("multivec", "maxsim"): topk(maxsim_scores(MV, MV_Q)),
}


def recall(ids, truth):
    return float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(ids, truth)]))


# ----------------------------------------------------------------------------- harness


def roundtrip(index, search):
    """Saves and loads `index`; the loaded index must give identical results."""
    d0, i0 = search(index)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "index.kannolo")
        index.save(path)
        loaded = kannolo.load(path)
    assert loaded.spec == index.spec, (loaded.spec, index.spec)
    assert loaded.nbytes == index.nbytes and len(loaded) == len(index)
    d1, i1 = search(loaded)
    assert (i0 == i1).all() and np.array_equal(d0, d1), "results changed after save/load"
    return i0


def run(name, data, queries, truth_key, min_recall, search_kwargs=None, rerank_queries=None, **build_kwargs):
    index = kannolo.build(data, **build_kwargs)
    kw = dict(search_kwargs or {})
    if rerank_queries is not None:
        kw["rerank_queries"] = rerank_queries
    ids = roundtrip(index, lambda ix: ix.search(queries, K, **kw))
    r = recall(ids, TRUTH[truth_key])
    print(f"{name:<70} recall@{K} {r:.3f}  {index.nbytes / 2**10:8.0f} KiB")
    assert r >= min_recall, (name, r)
    return index


TRAINED = (kannolo.PQ, kannolo.TwoLevelPQ)  # k-means is unseeded: every build trains another codebook


def exact_rescoring(name, data, queries, truth_key, first_encoder, rerank, rerank_queries=None, metric=None):
    """Builds a Flat first stage and makes every item a candidate. The results are then the top-k
    under the last stage's own scoring, that is, the results of a Flat index with that stage's
    encoder over that stage's data."""
    index = kannolo.build(data, metric=metric, encoder=first_encoder, index=kannolo.Flat(), rerank=rerank)
    stages = len(index.spec.rerank)
    kw = {"k_candidates": [N] * stages}
    if rerank_queries is not None:
        kw["rerank_queries"] = rerank_queries
    dists = {}

    def search(ix):
        d, i = ix.search(queries, K, **kw)
        dists["d"] = d
        return d, i

    ids = roundtrip(index, search)
    r = recall(ids, TRUTH[truth_key])

    last = index.spec.rerank[-1]
    last_data = (rerank[-1] if isinstance(rerank, list) else rerank).data
    ref_index = kannolo.build(data if last_data is None else last_data, metric=last.metric,
                         representation=last.representation, encoder=last.encoder, index=kannolo.Flat())
    ref_d, ref_ids = ref_index.search(queries if rerank_queries is None else rerank_queries, K)
    r_ref = recall(ref_ids, TRUTH[truth_key])
    print(f"{name:<70} recall@{K} {r:.3f} (Flat {last.encoder}: {r_ref:.3f})")
    if isinstance(last.encoder, TRAINED):
        assert abs(r - r_ref) <= 0.08, (name, r, r_ref)
    else:
        # Same scores as the reference, up to rounding (and the order of ties).
        assert np.allclose(np.sort(dists["d"], axis=1), np.sort(ref_d, axis=1), rtol=1e-4, atol=1e-4), name


def residual_rescoring(name, first_encoder, rerank, min_recall, data=DENSE, queries=DENSE_Q,
                       truth_key=("dense", "ip"), search_kwargs=None, **build):
    """Builds a Flat first stage with residual rescoring and makes every item a candidate. The
    first-stage score plus the residual score approximates the full inner product."""
    index = kannolo.build(data, encoder=first_encoder, index=kannolo.Flat(), rerank=rerank, **build)
    ids = roundtrip(index, lambda ix: ix.search(queries, K, k_candidates=N, **(search_kwargs or {})))
    r = recall(ids, TRUTH[truth_key])
    print(f"{name:<70} recall@{K} {r:.3f}")
    assert r >= min_recall, (name, r)


# ----------------------------------------------------------------------------- dense rows


def test_dense_flat():
    for metric in ("ip", "l2"):
        t = ("dense", metric)
        run(f"Flat Plain(f32) {metric}", DENSE, DENSE_Q, t, 1.0, metric=metric, encoder=kannolo.Plain("f32"), index=kannolo.Flat())
        for enc, floor in [(kannolo.Plain("f16"), 0.99), (kannolo.Plain("bf16"), 0.9), (kannolo.PQ(32), 0.3),
                           (kannolo.RaBitQ(1), 0.05), (kannolo.RaBitQ(4), 0.6)]:
            run(f"Flat {enc} {metric}", DENSE, DENSE_Q, t, floor, metric=metric, encoder=enc, index=kannolo.Flat())


def test_dense_fixed_point():
    """Unsigned fixed point represents [0, 3.98] with 8 bits and [0, 8) with 16 bits. It is
    accurate on data in range, and build warns when data is out of range."""
    import warnings

    lo, hi = DENSE.min(), DENSE.max()
    scaled = ((DENSE - lo) / (hi - lo) * 3.9).astype(np.float32)
    scaled_q = ((DENSE_Q - lo) / (hi - lo) * 3.9).astype(np.float32)
    truth = topk(dense_scores(scaled, scaled_q, "l2"))
    for bits, floor in ((8, 0.8), (16, 0.99)):
        index = kannolo.build(scaled, metric="l2", encoder=kannolo.FixedPoint(bits), index=kannolo.Flat())
        ids = roundtrip(index, lambda ix: ix.search(scaled_q, K))
        r = recall(ids, truth)
        print(f"{'Flat FixedPoint(' + str(bits) + ') l2, data in [0, 3.9]':<70} recall@{K} {r:.3f}")
        assert r >= floor, r
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kannolo.build(DENSE[:300], encoder=kannolo.FixedPoint(8), index=kannolo.Flat())
    assert any("saturate" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_dense_flat_rescored():
    for first in (kannolo.PQ(32), kannolo.RaBitQ(1)):
        for rescore in (kannolo.Plain("f32"), kannolo.PQ(64), kannolo.RaBitQ(8)):
            exact_rescoring(f"Flat {first} -> {rescore}", DENSE, DENSE_Q, ("dense", "ip"), first, kannolo.Rerank(rescore))


def test_dense_hnsw_rescored():
    # HNSWPQ + refine, HNSW-RaBitQ: rescore the first stage's candidates.
    for first in (kannolo.PQ(32), kannolo.RaBitQ(1)):
        for rescore in (kannolo.Plain("f16"), kannolo.PQ(64), kannolo.RaBitQ(8)):
            # PQ(64) rescoring is bounded by PQ(64) itself (~0.84 as a Flat index on this data)
            floor = 0.7 if isinstance(rescore, kannolo.PQ) else 0.9
            run(f"HNSW {first} -> {rescore}", DENSE, DENSE_Q, ("dense", "l2"), floor,
                metric="l2", encoder=first, index=kannolo.HNSW(m=16, graph="permuted"),
                rerank=kannolo.Rerank(rescore),
                # 4-bit queries for the 1-bit codes, the paper's default
                search_kwargs={"k_candidates": 200, "ef_search": 200,
                               **({"query_bits": 4} if first == kannolo.RaBitQ(1) else {})})


def test_dense_residual_rescoring():
    # The rescoring stage scores the residual x - x̂ of the RaBitQ first stage. Under inner
    # product, the two scores add up. RaBitQ-ext computes <q, x̂> exactly, so the sum is the exact
    # inner product up to rounding. 1-bit RaBitQ also quantizes the query, so the sum keeps that
    # error.
    for first, floor, kw in ((kannolo.RaBitQ(1), 0.95, {"query_bits": 8}), (kannolo.RaBitQ(4), 0.99, {})):
        residual_rescoring(f"Flat {first} -> residual Plain(f32) {kw}", first,
                           kannolo.Rerank(kannolo.Plain("f32"), residual=True), floor, search_kwargs=kw)
        # 1-bit codes need precise queries for the sum to hold (query_bits=8)
        index = run(f"HNSW {first} -> residual RaBitQ(8)", DENSE, DENSE_Q, ("dense", "ip"), 0.85,
                    metric="ip", encoder=first, index=kannolo.HNSW(m=16),
                    rerank=kannolo.Rerank(kannolo.RaBitQ(8), residual=True),
                    search_kwargs={"k_candidates": 300, "ef_search": 300,
                                   **({"query_bits": 8} if first == kannolo.RaBitQ(1) else {})})
        # the residual stage has its own rotation, with the seed derived as in Rust's DenseRerankIndex
        assert index.spec.rerank[0].encoder.seed != first.seed


def test_dense_ivf():
    for metric in ("ip", "l2"):
        t = ("dense", metric)
        for centroids in (kannolo.Flat(), kannolo.HNSW(m=16, ef_construction=100)):
            for enc in (kannolo.Plain("f32"), kannolo.Plain("f16"), kannolo.PQ(32), kannolo.RaBitQ(1), kannolo.RaBitQ(4)):
                for residual in (False, True):
                    floor = {"PQ": 0.3, "RaBitQ": 0.05 if enc == kannolo.RaBitQ(1) else 0.5}.get(type(enc).__name__, 0.9)
                    run(f"IVF {type(centroids).__name__} centroids, {enc}, residual={residual}, {metric}",
                        DENSE, DENSE_Q, t, floor, metric=metric, encoder=enc,
                        index=kannolo.IVF(n_lists=64, residual=residual, centroids=centroids, seed=1),
                        search_kwargs={"nprobe": 16})


def test_dense_ivf_rescored():
    for enc in (kannolo.PQ(32), kannolo.RaBitQ(4)):
        run(f"IVF {enc} -> Plain(f32)", DENSE, DENSE_Q, ("dense", "l2"), 0.9, metric="l2", encoder=enc,
            index=kannolo.IVF(n_lists=64, residual=True, seed=1), rerank=kannolo.Rerank(kannolo.Plain("f32")),
            search_kwargs={"nprobe": 16, "k_candidates": 200})


# ----------------------------------------------------------------------------- sparse rows


SPARSE_ENCODERS = [kannolo.Plain("f32"), kannolo.Plain("f16"), kannolo.FixedPoint(8), kannolo.Scalar(8), kannolo.Blocked("f16"), kannolo.DotPacking8("fixed_u8")]


def test_sparse_flat():
    for enc in SPARSE_ENCODERS:
        run(f"sparse Flat {enc}", SPARSE, SPARSE_Q, ("sparse", "ip"), 0.9, encoder=enc, index=kannolo.Flat())
    run("sparse Flat Plain(f32) l2", SPARSE, SPARSE_Q, ("sparse", "l2"), 1.0, metric="l2", index=kannolo.Flat())


def test_sparse_ivf():
    for enc in (kannolo.Plain("f32"), kannolo.Plain("f16")):
        for centroids in (kannolo.Flat(), kannolo.HNSW(m=16)):
            run(f"sparse IVF {type(centroids).__name__} centroids, {enc}", SPARSE, SPARSE_Q, ("sparse", "ip"), 0.5,
                encoder=enc, index=kannolo.IVF(n_lists=32, centroids=centroids, seed=1, centroid_nnz_factor=4.0),
                search_kwargs={"nprobe": 8})


def test_sparse_rescored():
    for first in (kannolo.FixedPoint(8), kannolo.DotPacking8("scalar_4bit")):
        exact_rescoring(f"sparse Flat {first} -> Plain(f32)", SPARSE, SPARSE_Q, ("sparse", "ip"), first, kannolo.Rerank(kannolo.Plain("f32")))
    for index in (kannolo.HNSW(m=16), kannolo.IVF(n_lists=32, seed=1)):
        run(f"sparse {type(index).__name__} FixedPoint(8)->Plain(f32)" if not isinstance(index, kannolo.IVF) else "sparse IVF Plain(f16) -> Plain(f32)",
            SPARSE, SPARSE_Q, ("sparse", "ip"), 0.8,
            encoder=kannolo.FixedPoint(8) if isinstance(index, kannolo.HNSW) else kannolo.Plain("f16"), index=index,
            rerank=kannolo.Rerank(kannolo.Plain("f32")),
            search_kwargs={"k_candidates": 100, **({"nprobe": 8} if isinstance(index, kannolo.IVF) else {})})


def test_sparse_pruned_residual():
    # The first stage keeps the top 60% of each vector's L1 mass. The rescoring stage scores the
    # pruned part.
    for first in (kannolo.Plain("f16"), kannolo.Blocked("f16")):
        for rescore in (kannolo.Plain("f32"), kannolo.Blocked("f16"), kannolo.FixedPoint(16)):
            index = kannolo.build(SPARSE, encoder=first, index=kannolo.Flat(), transform=kannolo.L1Prune(60),
                             rerank=kannolo.Rerank(rescore, residual=True))
            ids = roundtrip(index, lambda ix: ix.search(SPARSE_Q, K, k_candidates=N))
            r = recall(ids, TRUTH[("sparse", "ip")])
            print(f"{'sparse Flat L1Prune(60) ' + str(first) + ' -> residual ' + str(rescore):<70} exact rescoring: recall@{K} {r:.3f}")
            assert r >= 0.97, r  # first stage and residual are each quantized; their sum is near-exact
        index = run(f"sparse HNSW L1Prune(60) {first} -> residual Plain(f16)", SPARSE, SPARSE_Q,
                    ("sparse", "ip"), 0.6, encoder=first, index=kannolo.HNSW(m=16), transform=kannolo.L1Prune(60),
                    rerank=kannolo.Rerank(kannolo.Plain("f16"), residual=True), search_kwargs={"k_candidates": 100})
    # Pruned first stage rescored with the full vectors.
    run("sparse HNSW L1Prune(60) Plain(f16) -> full Plain(f32)", SPARSE, SPARSE_Q, ("sparse", "ip"), 0.6,
        encoder=kannolo.Plain("f16"), index=kannolo.HNSW(m=16), transform=kannolo.L1Prune(60),
        rerank=kannolo.Rerank(kannolo.Plain("f32")), search_kwargs={"k_candidates": 100})


# ----------------------------------------------------------------------------- multi-vector rows


MV_ENCODERS = [kannolo.Plain("f32"), kannolo.Plain("f16"), kannolo.PQ(16), kannolo.TwoLevelPQ(16, n_coarse=256)]


def test_multivector_flat_and_hnsw():
    for enc in MV_ENCODERS:
        floor = 0.99 if isinstance(enc, kannolo.Plain) else 0.4
        run(f"multivec Flat {enc}", MV, MV_Q, ("multivec", "maxsim"), floor, encoder=enc, index=kannolo.Flat())
        run(f"multivec HNSW {enc}", MV, MV_Q, ("multivec", "maxsim"), floor - 0.1, encoder=enc,
            index=kannolo.HNSW(m=16, graph="permuted"), search_kwargs={"ef_search": 200})


def test_multivector_rescored():
    for first in (kannolo.PQ(16), kannolo.TwoLevelPQ(16, n_coarse=256)):
        for rescore in (kannolo.Plain("f32"), kannolo.PQ(32)):
            exact_rescoring(f"multivec Flat {first} -> {rescore}", MV, MV_Q, ("multivec", "maxsim"), first, kannolo.Rerank(rescore))


# ----------------------------------------------------------------------------- hybrid rows


def test_hybrid():
    # Sparse first stage, rescored with another representation of the same items.
    exact_rescoring("hybrid sparse Flat -> dense Plain(f32)", SPARSE, SPARSE_Q, ("dense", "ip"),
                    kannolo.Plain("f16"), kannolo.Rerank(kannolo.Plain("f32"), data=DENSE), rerank_queries=DENSE_Q)
    exact_rescoring("hybrid sparse Flat -> multivec Plain(f32)", SPARSE, SPARSE_Q, ("multivec", "maxsim"),
                    kannolo.Plain("f16"), kannolo.Rerank(kannolo.Plain("f32"), data=MV), rerank_queries=MV_Q)
    exact_rescoring("hybrid dense Flat -> multivec Plain(f32)", DENSE, DENSE_Q, ("multivec", "maxsim"),
                    kannolo.PQ(32), kannolo.Rerank(kannolo.Plain("f32"), data=MV), rerank_queries=MV_Q, metric="ip")
    # With an approximate first stage, only the wiring is checked: the pipeline must run end to
    # end. Recall depends on how well the two synthetic representations agree.
    index = kannolo.build(SPARSE, encoder=kannolo.DotPacking8(), index=kannolo.HNSW(m=16),
                     rerank=kannolo.Rerank(kannolo.TwoLevelPQ(16, n_coarse=256), data=MV))
    d, i = index.search(SPARSE_Q, K, rerank_queries=MV_Q, k_candidates=100)
    assert i.shape == (NQ, K) and (i >= 0).all()
    roundtrip(index, lambda ix: ix.search(SPARSE_Q, K, rerank_queries=MV_Q))
    # one query per representation
    d1, i1 = index.search((SPARSE_Q[0][:40], SPARSE_Q[1][:40]), K, rerank_queries=MV_Q[0], k_candidates=100)
    assert (i1 == i[0]).all()


# ----------------------------------------------------------------------------- cascade


def test_cascade():
    index = run("cascade HNSW RaBitQ(1) -> RaBitQ(8) -> Plain(f32)", DENSE, DENSE_Q, ("dense", "l2"), 0.9,
                metric="l2", encoder=kannolo.RaBitQ(1), index=kannolo.HNSW(m=16),
                rerank=[kannolo.Rerank(kannolo.RaBitQ(8)), kannolo.Rerank(kannolo.Plain("f32"))],
                search_kwargs={"k_candidates": [400, 50], "ef_search": 400, "query_bits": 4})
    exact_rescoring("cascade Flat PQ(32) -> PQ(64) -> Plain(f32)", DENSE, DENSE_Q, ("dense", "ip"),
                    kannolo.PQ(32), [kannolo.Rerank(kannolo.PQ(64)), kannolo.Rerank(kannolo.Plain("f32"))])


# ----------------------------------------------------------------------------- search options


def expect_error(fn, *needles, exc=ValueError):
    try:
        fn()
    except exc as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        return
    raise AssertionError("expected an error")


def test_search_parameters_are_checked():
    hnsw = kannolo.build(DENSE[:500])
    flat = kannolo.build(DENSE[:500], index=kannolo.Flat())
    ivf = kannolo.build(DENSE[:500], index=kannolo.IVF(n_lists=8))
    expect_error(lambda: hnsw.search(DENSE_Q, nprobe=4), "nprobe applies to IVF")
    expect_error(lambda: flat.search(DENSE_Q, ef_search=10), "ef_search applies")
    expect_error(lambda: ivf.search(DENSE_Q, ef_search=10), "ef_search applies")
    expect_error(lambda: hnsw.search(DENSE_Q, k_candidates=50), "rescoring stages")
    ivf_h = kannolo.build(DENSE[:500], index=kannolo.IVF(n_lists=8, centroids=kannolo.HNSW(m=8)))
    ivf_h.search(DENSE_Q, nprobe=4, ef_search=20)
    two = kannolo.build(DENSE[:500], encoder=kannolo.PQ(16), rerank=kannolo.Rerank())
    expect_error(lambda: two.search(DENSE_Q, k=20, k_candidates=10), "at least k")
    expect_error(lambda: two.search(DENSE_Q, k_candidates=[50, 50]), "entries")


def test_alpha_and_beta():
    index = kannolo.build(DENSE, metric="l2", encoder=kannolo.RaBitQ(1), index=kannolo.HNSW(m=16), rerank=kannolo.Rerank(kannolo.Plain("f32")))
    base = index.search(DENSE_Q, K, k_candidates=300, ef_search=300)[1]
    # beta larger than the candidate list never exits early; alpha=1 keeps every candidate
    assert (index.search(DENSE_Q, K, k_candidates=300, ef_search=300, beta=1000)[1] == base).all()
    assert (index.search(DENSE_Q, K, k_candidates=300, ef_search=300, alpha=1.0)[1] == base).all()
    r_base = recall(base, TRUTH[("dense", "l2")])
    r_beta = recall(index.search(DENSE_Q, K, k_candidates=300, ef_search=300, beta=20)[1], TRUTH[("dense", "l2")])
    print(f"alpha/beta: recall {r_base:.3f}, with beta=20 {r_beta:.3f}")
    assert r_beta <= r_base + 1e-9 and r_beta > 0.5


def test_invalid_compositions():
    expect_error(lambda: kannolo.check(representation=kannolo.MultiVector(), encoder=kannolo.RaBitQ()), "not available for multi-vector")
    expect_error(lambda: kannolo.check(representation=kannolo.MultiVector(), metric="ip"), "maxsim")
    expect_error(lambda: kannolo.check(representation=kannolo.MultiVector(), index=kannolo.IVF()), "IVF is not available")
    expect_error(lambda: kannolo.check(representation=kannolo.Sparse(), index=kannolo.IVF(residual=True)), "dense data only")
    expect_error(lambda: kannolo.check(representation=kannolo.Sparse(), encoder=kannolo.DotPacking8(), index=kannolo.IVF()), "Plain")
    expect_error(lambda: kannolo.check(encoder=kannolo.PQ(), rerank=kannolo.Rerank(residual=True)), "RaBitQ first-stage")
    expect_error(lambda: kannolo.check(metric="l2", encoder=kannolo.RaBitQ(), rerank=kannolo.Rerank(residual=True)), "inner product")
    expect_error(lambda: kannolo.check(representation=kannolo.Sparse(), rerank=kannolo.Rerank(residual=True)), "L1Prune")
    expect_error(lambda: kannolo.check(transform=kannolo.L1Prune(50)), "sparse data only")
    expect_error(lambda: kannolo.check(encoder=kannolo.RaBitQ(), rerank=[kannolo.Rerank(), kannolo.Rerank(residual=True)]), "first rescoring stage")
    # mismatched rescoring data size
    expect_error(lambda: kannolo.build(DENSE, rerank=kannolo.Rerank(data=DENSE[:10])), "same collection")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
