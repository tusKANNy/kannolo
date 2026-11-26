"""Filtered search (ACORN) and composing indexes from separately built parts.

Run with `pytest python/tests` or `python python/tests/test_filter_compose.py`.
"""

import os
import tempfile

import numpy as np

import kannolo

K = 10
RNG = np.random.default_rng(7)


def clustered(n, dim, seed, n_clusters=30):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    return (centers[rng.integers(n_clusters, size=n)] + 0.4 * rng.normal(size=(n, dim))).astype(np.float32)


N, NQ, DIM = 3000, 50, 64
DENSE = clustered(N, DIM, 0)
DENSE_Q = clustered(NQ, DIM, 1)


def sparse_csr(x, keep):
    x = x - x.min() + 0.01
    idx = np.sort(np.argsort(-x, axis=1)[:, :keep], axis=1)
    rows = np.arange(len(x))[:, None]
    dense = np.zeros_like(x)
    dense[rows, idx] = x[rows, idx]
    return (idx.ravel().astype(np.int64), dense[rows, idx].ravel(), np.arange(0, len(x) * keep + 1, keep)), dense


SPARSE, SPARSE_DENSE = sparse_csr(clustered(N, 500, 2), 30)
SPARSE_Q, SPARSE_Q_DENSE = sparse_csr(clustered(NQ, 500, 3), 30)
MV = [(DENSE[i] + 0.3 * RNG.normal(size=(int(RNG.integers(3, 8)), DIM))).astype(np.float32) for i in range(N)]
MV_Q = [(DENSE_Q[i] + 0.3 * RNG.normal(size=(4, DIM))).astype(np.float32) for i in range(NQ)]


def l2(data, queries):
    return (queries**2).sum(1)[:, None] - 2 * queries @ data.T + (data**2).sum(1)[None, :]


def exact(scores, allowed=None, k=K):
    """Top-k ids per row (smallest score), among `allowed` (a mask) if given."""
    s = np.array(scores, dtype=np.float64)
    if allowed is not None:
        s[:, ~allowed] = np.inf
    return np.argsort(s, axis=1, kind="stable")[:, :k]


def recall(ids, truth):
    return float(np.mean([len(set(a[a >= 0]) & set(b)) / len(b) for a, b in zip(ids, truth)]))


def expect_error(fn, needle, exc=ValueError):
    try:
        fn()
    except exc as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"expected an error containing {needle!r}")


# ----------------------------------------------------------------------------- filtered search

EVEN = np.arange(N) % 2 == 0
RARE = np.zeros(N, dtype=bool)
RARE[RNG.choice(N, N // 100, replace=False)] = True  # 1% of the collection


def test_hnsw_acorn_1_and_gamma():
    truth_even = exact(l2(DENSE, DENSE_Q), EVEN)
    truth_rare = exact(l2(DENSE, DENSE_Q), RARE)
    for graph in ("standard", "permuted", "compressed"):  # ACORN works on original ids
        for name, mask, truth in (("even", EVEN, truth_even), ("1%", RARE, truth_rare)):
            index = kannolo.build(DENSE, metric="l2", index=kannolo.HNSW(m=16, graph=graph))
            assert not index.has_acorn_gamma
            _, i = index.search(DENSE_Q, K, filter=mask, ef_search=200)
            assert mask[i[i >= 0]].all(), "a filtered-out id was returned"
            r1 = recall(i, truth)
            _, i_wide = index.search(DENSE_Q, K, filter=mask, ef_search=1000)
            r1_wide = recall(i_wide, truth)
            index.build_acorn_gamma(3)
            assert index.has_acorn_gamma
            _, ig = index.search(DENSE_Q, K, filter=mask, ef_search=200)
            assert mask[ig[ig >= 0]].all()
            rg = recall(ig, truth)
            _, i1 = index.search(DENSE_Q, K, filter=mask, ef_search=200, acorn_gamma=False)
            assert (i1 == i).all(), "acorn_gamma=False must be ACORN-1"
            print(f"HNSW {graph:<10} filter {name:<4}: ACORN-1 recall {r1:.3f} (ef 1000: {r1_wide:.3f}), ACORN-γ {rg:.3f}")
            assert rg >= 0.9, (graph, name, rg)
            assert r1_wide >= r1 - 1e-9, "a wider beam must not lose recall"
            # ACORN-1 degrades under highly selective filters. ACORN-γ is designed for this case.
            assert r1 >= (0.9 if name == "even" else 0.5), (graph, name, r1)


def test_filter_forms_agree():
    index = kannolo.build(DENSE, metric="l2", index=kannolo.HNSW(m=16))
    ids_even = np.flatnonzero(EVEN)
    by_mask = index.search(DENSE_Q, K, filter=EVEN)[1]
    assert (index.search(DENSE_Q, K, filter=ids_even)[1] == by_mask).all()
    assert (index.search(DENSE_Q, K, filter=list(ids_even))[1] == by_mask).all()
    assert (index.search(DENSE_Q, K, filter=lambda i: i % 2 == 0)[1] == by_mask).all()
    # one filter per query: query q may only return ids congruent to q mod 5
    per_query = [np.flatnonzero(np.arange(N) % 5 == q % 5) for q in range(NQ)]
    _, ids = index.search(DENSE_Q, K, filter=per_query, ef_search=200)
    for q in range(NQ):
        assert (ids[q] % 5 == q % 5).all()
    # the same filters as per-query masks: a list of masks, and one 2-D array
    masks = np.stack([np.arange(N) % 5 == q % 5 for q in range(NQ)])
    assert (index.search(DENSE_Q, K, filter=list(masks), ef_search=200)[1] == ids).all()
    assert (index.search(DENSE_Q, K, filter=masks, ef_search=200)[1] == ids).all()
    expect_error(lambda: index.search(DENSE_Q, K, filter=masks[:, :10]), "shape")
    truth = np.stack([exact(l2(DENSE, DENSE_Q[q:q + 1]), np.arange(N) % 5 == q % 5)[0] for q in range(NQ)])
    print(f"per-query filters: recall {recall(ids, truth):.3f}")
    assert recall(ids, truth) >= 0.9
    # single query
    d, i = index.search(DENSE_Q[0], K, filter=EVEN)
    assert i.shape == (K,) and EVEN[i].all()
    expect_error(lambda: index.search(DENSE_Q, K, filter=EVEN[:10]), "one entry per vector")
    expect_error(lambda: index.search(DENSE_Q, K, filter=[N + 5]), "ids must be in")
    expect_error(lambda: index.search(DENSE_Q, K, filter=per_query[:3]), "per-query filters")
    expect_error(lambda: index.search(DENSE_Q, K, filter=EVEN, acorn_gamma=True), "build_acorn_gamma")
    expect_error(lambda: index.search(DENSE_Q, K, acorn_gamma=True), "filter=")


def test_too_few_allowed_pads():
    index = kannolo.build(DENSE, index=kannolo.HNSW(m=16))
    d, i = index.search(DENSE_Q[:3], K, filter=[1, 2, 3])
    assert set(i[i >= 0].ravel()) <= {1, 2, 3} and (i[:, 3:] == -1).all()


def test_sparse_hnsw_filtered():
    truth = exact(-(SPARSE_Q_DENSE @ SPARSE_DENSE.T), EVEN)
    index = kannolo.build(SPARSE, encoder=kannolo.DotPacking8(), index=kannolo.HNSW(m=16))
    _, i = index.search(SPARSE_Q, K, filter=EVEN, ef_search=200)
    _, unfiltered = index.search(SPARSE_Q, K, ef_search=200)
    r, r_unfiltered = recall(i, truth), recall(unfiltered, exact(-(SPARSE_Q_DENSE @ SPARSE_DENSE.T)))
    print(f"sparse HNSW (DotPacking8) recall: filtered {r:.3f}, unfiltered {r_unfiltered:.3f}")
    # With a filter that keeps half the collection, ACORN loses little recall compared with the
    # same index without a filter.
    assert EVEN[i[i >= 0]].all() and r >= r_unfiltered - 0.1


def test_filtered_search_only_on_single_stage_hnsw():
    """For now, filtered search applies to HNSW first stages of single-stage indexes."""
    others = {
        "Flat": kannolo.build(DENSE, index=kannolo.Flat()),
        "IVF": kannolo.build(DENSE, index=kannolo.IVF(n_lists=16)),
        "two-stage": kannolo.build(DENSE, encoder=kannolo.RaBitQ(4), index=kannolo.HNSW(m=16), rerank=kannolo.Rerank(kannolo.Plain("f32"))),
    }
    for name, index in others.items():
        expect_error(lambda: index.search(DENSE_Q, K, filter=EVEN), "filtered search applies to HNSW first stages")
        expect_error(lambda: index.build_acorn_gamma(2), "ACORN-γ applies to HNSW first stages")
        assert not index.has_acorn_gamma, name


def test_acorn_gamma_is_not_saved():
    index = kannolo.build(DENSE, index=kannolo.HNSW(m=16))
    index.build_acorn_gamma(2)
    with tempfile.TemporaryDirectory() as tmp:
        index.save(os.path.join(tmp, "i"))
        assert not kannolo.load(os.path.join(tmp, "i")).has_acorn_gamma


# ----------------------------------------------------------------------------- compose


def results(index, queries, **kw):
    return index.search(queries, K, **kw)


def test_compose_equals_build():
    # The Flat first stage is deterministic, so compose must give the same results as build.
    for rescore in (kannolo.Plain("f32"), kannolo.Plain("f16"), kannolo.RaBitQ(8)):
        built = kannolo.build(DENSE, encoder=kannolo.RaBitQ(1), index=kannolo.Flat(), rerank=kannolo.Rerank(rescore))
        first = kannolo.build(DENSE, encoder=kannolo.RaBitQ(1), index=kannolo.Flat())
        composed = kannolo.compose(first, kannolo.Rerank(rescore, data=DENSE))
        assert composed.spec == built.spec
        a, b = results(built, DENSE_Q, k_candidates=100), results(composed, DENSE_Q, k_candidates=100)
        assert (a[1] == b[1]).all() and np.allclose(a[0], b[0], rtol=1e-5)
    # the composed index saves and loads like a built one
    with tempfile.TemporaryDirectory() as tmp:
        composed.save(os.path.join(tmp, "c"))
        loaded = kannolo.load(os.path.join(tmp, "c"))
        assert loaded.spec == composed.spec
        assert (results(loaded, DENSE_Q)[1] == results(composed, DENSE_Q)[1]).all()


def test_compose_three_by_three_from_six_files():
    """3 first stages and 3 rescoring datasets, stored as 6 files, composed into 9 indexes."""
    firsts = {
        "hnsw-rabitq1": dict(encoder=kannolo.RaBitQ(1), index=kannolo.HNSW(m=16)),
        "hnsw-pq": dict(encoder=kannolo.PQ(16), index=kannolo.HNSW(m=16, graph="compressed")),
        "ivf-rabitq4": dict(encoder=kannolo.RaBitQ(4), index=kannolo.IVF(n_lists=32, residual=True)),
    }
    rescorings = {"f32": kannolo.Plain("f32"), "f16": kannolo.Plain("f16"), "rabitq8": kannolo.RaBitQ(8)}
    truth = exact(l2(DENSE, DENSE_Q))
    with tempfile.TemporaryDirectory() as tmp:
        for name, kw in firsts.items():
            kannolo.build(DENSE, metric="l2", **kw).save(os.path.join(tmp, name + ".index"))
        np.save(os.path.join(tmp, "collection.npy"), DENSE)  # rescoring data, stored once per dataset
        data = np.load(os.path.join(tmp, "collection.npy"))
        for name in firsts:
            first = kannolo.load(os.path.join(tmp, name + ".index"))
            kw = {"nprobe": 8} if "ivf" in name else {"ef_search": 100}
            if name == "hnsw-rabitq1":
                kw["query_bits"] = 4
            before = results(first, DENSE_Q, **kw)
            # These are the first stage's 100 candidates, as the composed indexes see them. Exact
            # rescoring must find every true neighbor among them.
            candidates = first.search(DENSE_Q, 100, **kw)[1]
            coverage = recall(candidates, truth)
            for rname, enc in rescorings.items():
                index = kannolo.compose(first, kannolo.Rerank(enc, data=data))
                r = recall(results(index, DENSE_Q, k_candidates=100, **kw)[1], truth)
                print(f"compose {name:<13} + {rname:<8}: recall {r:.3f} (true top-10 among the candidates: {coverage:.3f})")
                if rname == "f32":
                    assert abs(r - coverage) < 1e-9, (name, r, coverage)
                else:
                    assert coverage - 0.05 <= r <= coverage + 1e-9, (name, rname, r, coverage)
            # compose shares the first stage and does not consume or change it
            after = results(first, DENSE_Q, **kw)
            assert (before[1] == after[1]).all()


def test_compose_hybrid_and_cascade():
    first = kannolo.build(SPARSE, encoder=kannolo.Blocked(), index=kannolo.HNSW(m=16))
    hybrid = kannolo.compose(first, kannolo.Rerank(kannolo.Plain("f16"), data=MV))
    assert isinstance(hybrid.spec.rerank[0].representation, kannolo.MultiVector)
    assert hybrid.spec.rerank[0].metric == "maxsim"
    _, i = hybrid.search(SPARSE_Q, K, rerank_queries=MV_Q, k_candidates=100)
    assert i.shape == (NQ, K) and (i >= 0).all()
    # compose onto a two-stage index: a cascade
    two = kannolo.build(DENSE, encoder=kannolo.RaBitQ(1), index=kannolo.HNSW(m=16), rerank=kannolo.Rerank(kannolo.RaBitQ(8)))
    cascade = kannolo.compose(two, kannolo.Rerank(kannolo.Plain("f32"), data=DENSE))
    assert len(cascade.spec.rerank) == 2
    truth = exact(-(DENSE_Q @ DENSE.T))
    r = recall(cascade.search(DENSE_Q, K, k_candidates=[300, 50], ef_search=300, query_bits=4)[1], truth)
    # The inner two-stage index passes 50 candidates to the f32 stage, which rescores them exactly.
    coverage = recall(two.search(DENSE_Q, 50, k_candidates=300, ef_search=300, query_bits=4)[1], truth)
    print(f"compose cascade RaBitQ(1) -> RaBitQ(8) -> f32: recall {r:.3f} (true top-10 among the candidates: {coverage:.3f})")
    assert abs(r - coverage) < 1e-9, (r, coverage)


def test_compose_residual():
    # Sparse: the first stage was built on 60% of the L1 mass. Like build, compose computes the
    # residual from the original collection.
    first = kannolo.build(SPARSE, encoder=kannolo.Plain("f16"), index=kannolo.Flat(), transform=kannolo.L1Prune(60))
    composed = kannolo.compose(first, kannolo.Rerank(kannolo.Plain("f32"), data=SPARSE, residual=True))
    built = kannolo.build(SPARSE, encoder=kannolo.Plain("f16"), index=kannolo.Flat(), transform=kannolo.L1Prune(60),
                     rerank=kannolo.Rerank(kannolo.Plain("f32"), residual=True))
    assert composed.spec == built.spec
    a = results(composed, SPARSE_Q, k_candidates=N)
    b = results(built, SPARSE_Q, k_candidates=N)
    assert (a[1] == b[1]).all()
    assert recall(a[1], exact(-(SPARSE_Q_DENSE @ SPARSE_DENSE.T))) >= 0.97
    # Dense: x - x̂ of the RaBitQ first stage, with its own rotation seed
    first = kannolo.build(DENSE, encoder=kannolo.RaBitQ(4), index=kannolo.Flat())
    composed = kannolo.compose(first, kannolo.Rerank(kannolo.Plain("f32"), data=DENSE, residual=True))
    r = recall(results(composed, DENSE_Q, k_candidates=N)[1], exact(-(DENSE_Q @ DENSE.T)))
    print(f"compose dense residual: recall {r:.3f}")
    assert r >= 0.99


def test_compose_errors():
    first = kannolo.build(DENSE, index=kannolo.HNSW(m=16))
    expect_error(lambda: kannolo.compose(first, kannolo.Rerank(kannolo.Plain())), "data=")
    expect_error(lambda: kannolo.compose(first, kannolo.Rerank(data=DENSE[:100])), "same collection")
    expect_error(lambda: kannolo.compose(first, kannolo.Rerank(data=DENSE, residual=True)), "RaBitQ first-stage")
    expect_error(lambda: kannolo.compose(first, []), "at least one")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
