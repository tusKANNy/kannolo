"""Tests for the composable API (kannolo.build / kannolo.load).

Run with `pytest python/tests` or `python python/tests/test_composable.py`, after building the
extension (`maturin develop --release`).
"""

import os
import struct
import tempfile

import numpy as np

import kannolo

RNG = np.random.default_rng(0)
K = 10


# ----------------------------------------------------------------------------- data


def clustered_dense(n, dim, n_clusters=50, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    x = centers[rng.integers(n_clusters, size=n)] + 0.3 * rng.normal(size=(n, dim)).astype(np.float32)
    return x.astype(np.float32)


def exact_topk(data, queries, metric, k=K):
    if metric == "ip":
        scores = -(queries @ data.T)
    else:
        scores = (queries**2).sum(1)[:, None] - 2 * queries @ data.T + (data**2).sum(1)[None, :]
    return np.argsort(scores, axis=1)[:, :k]


def recall(ids, truth):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(ids, truth)])


def random_sparse(n, vocab, nnz, seed=0, stride=1, n_topics=30):
    """CSR arrays plus the dense matrix (over compact ids) for ground truth. Documents share
    topics, so nearest neighbors are meaningful. `stride` spreads component ids apart (to
    exercise u32 components without a huge dense matrix)."""
    rng = np.random.default_rng(seed)
    topics = [rng.choice(vocab, size=3 * nnz, replace=False) for _ in range(n_topics)]
    comps, vals, offs = [], [], [0]
    dense = np.zeros((n, vocab), dtype=np.float32)
    for i in range(n):
        t = topics[rng.integers(n_topics)]
        c = np.unique(np.concatenate([rng.choice(t, size=nnz, replace=False),
                                      rng.choice(vocab, size=nnz // 4, replace=False)]))
        v = rng.random(c.size).astype(np.float32) + 0.1
        dense[i, c] = v
        rng.shuffle(c)  # unsorted on purpose: the API must sort
        comps.append(c * stride)
        vals.append(dense[i, c])
        offs.append(offs[-1] + c.size)
    return (np.concatenate(comps).astype(np.int64), np.concatenate(vals), np.array(offs)), dense


def write_seismic(path, csr):
    comps, vals, offs = csr
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(offs) - 1))
        for s, e in zip(offs[:-1], offs[1:]):
            order = np.argsort(comps[s:e])
            f.write(struct.pack("<I", e - s))
            f.write(comps[s:e][order].astype("<u4").tobytes())
            f.write(vals[s:e][order].astype("<f4").tobytes())


DENSE = clustered_dense(4000, 128)
DENSE_Q = clustered_dense(100, 128, seed=1)
SPARSE, SPARSE_DENSE = random_sparse(3000, 2000, 40)
SPARSE_Q, SPARSE_Q_DENSE = random_sparse(100, 2000, 40, seed=1)


def sparse_queries(csr):
    return csr


# ----------------------------------------------------------------------------- validation


def expect_error(fn, *needles, exc=ValueError):
    try:
        fn()
    except exc as e:
        msg = str(e)
        for needle in needles:
            assert needle in msg, f"{needle!r} not in {msg!r}"
        return msg
    raise AssertionError("expected an error")


def test_invalid_combinations_are_rejected_with_a_reason():
    expect_error(lambda: kannolo.check(representation=kannolo.Sparse(), encoder=kannolo.PQ()), "not available for sparse data")
    expect_error(lambda: kannolo.check(representation=kannolo.Dense(), encoder=kannolo.DotPacking8()), "not available for dense data")
    expect_error(lambda: kannolo.check(representation=kannolo.Sparse(), metric="l2", encoder=kannolo.DotPacking8()), "inner product")
    expect_error(lambda: kannolo.check(representation=kannolo.Sparse("u32"), encoder=kannolo.Blocked()), "u16")
    expect_error(lambda: kannolo.check(encoder=kannolo.PQ(subspaces=7)), "not compiled in", "RaBitQ")
    expect_error(lambda: kannolo.check(encoder=kannolo.PQ(subspaces=96)), "not compiled in", "RaBitQ")
    expect_error(lambda: kannolo.check(encoder=kannolo.RaBitQ(bits=3)), "1, 2, 4 or 8")
    expect_error(lambda: kannolo.check(index=kannolo.HNSW(m=200, graph="compressed")), "m <= 128")
    expect_error(lambda: kannolo.check(metric="cosine"), "'ip'")
    # u32 is fine where the encoder supports it
    kannolo.check(representation=kannolo.Sparse("u32"), encoder=kannolo.DotPacking8("scalar_u8"))


def test_dimension_dependent_checks_happen_at_build_time():
    data = clustered_dense(200, 96)
    expect_error(lambda: kannolo.build(data, encoder=kannolo.RaBitQ()), "multiple of 64")
    expect_error(lambda: kannolo.build(data, encoder=kannolo.PQ(subspaces=64)), "must divide")


# ----------------------------------------------------------------------------- dense


# Absolute recall depends on the data. Inner product on unnormalized vectors is maximum inner
# product search (MIPS), which is harder for HNSW, and PQ and 1-bit RaBitQ are lossy. The sweep
# therefore only checks that every combination works: miswired ids or a lost permutation give
# recall near 0 (random: ~0.003).
SANITY_RECALL = 0.3
# 1-bit RaBitQ alone is far less accurate on this data. It is meant to be followed by a
# rescoring stage. test_rabitq_accuracy_grows_with_bits checks its accuracy.
SANITY_RECALL_RABITQ1 = 0.05
DENSE_ENCODERS = [
    kannolo.Plain("f32"),
    kannolo.Plain("f16"),
    kannolo.PQ(subspaces=32),
    kannolo.PQ(subspaces=64),
    kannolo.RaBitQ(bits=1),
    kannolo.RaBitQ(bits=4),
]


def test_dense_every_encoder_metric_and_graph():
    for metric in ("ip", "l2"):
        truth = exact_topk(DENSE, DENSE_Q, metric)
        for encoder in DENSE_ENCODERS:
            for graph in ("standard", "permuted", "compressed"):
                index = kannolo.build(DENSE, metric=metric, encoder=encoder, index=kannolo.HNSW(m=16, graph=graph))
                dist, ids = index.search(DENSE_Q, k=K, ef_search=200)
                assert dist.shape == ids.shape == (len(DENSE_Q), K)
                r = recall(ids, truth)
                print(f"dense {metric} {encoder} {graph}: recall@{K} = {r:.3f}")
                floor = SANITY_RECALL_RABITQ1 if encoder == kannolo.RaBitQ(bits=1) else SANITY_RECALL
                assert r >= floor, (metric, encoder, graph, r)


def test_rabitq_accuracy_grows_with_bits():
    """Wiring check for RaBitQ: more bits per component must mean more accuracy, and 8-bit
    codes must come close to the uncompressed index."""
    for metric in ("ip", "l2"):
        truth = exact_topk(DENSE, DENSE_Q, metric)
        recalls = [
            recall(kannolo.build(DENSE, metric=metric, encoder=kannolo.RaBitQ(bits=b), index=kannolo.HNSW(m=16))
                   .search(DENSE_Q, ef_search=200)[1], truth)
            for b in (1, 2, 4, 8)
        ]
        plain = recall(kannolo.build(DENSE, metric=metric, index=kannolo.HNSW(m=16)).search(DENSE_Q, ef_search=200)[1], truth)
        print(f"rabitq {metric} bits 1/2/4/8: {recalls}, plain {plain:.3f}")
        assert all(a < b for a, b in zip(recalls, recalls[1:])), recalls
        assert recalls[-1] >= plain - 0.1


def test_rabitq_query_bits():
    index = kannolo.build(DENSE, encoder=kannolo.RaBitQ(bits=1))
    truth = exact_topk(DENSE, DENSE_Q, "ip")
    r1 = recall(index.search(DENSE_Q, ef_search=200, query_bits=1)[1], truth)
    r8 = recall(index.search(DENSE_Q, ef_search=200, query_bits=8)[1], truth)
    print(f"rabitq query_bits=1: {r1:.3f}, query_bits=8: {r8:.3f}")
    assert r8 >= r1 - 0.02
    expect_error(lambda: index.search(DENSE_Q, query_bits=9), "1..=8")
    plain = kannolo.build(DENSE[:500])
    expect_error(lambda: plain.search(DENSE_Q, query_bits=4), "RaBitQ(bits=1)")


def test_single_query_shape_and_threads():
    index = kannolo.build(DENSE, metric="l2")
    d1, i1 = index.search(DENSE_Q[0], k=5)
    assert d1.shape == i1.shape == (5,)
    d, i = index.search(DENSE_Q, k=5, num_threads=1)
    d4, i4 = index.search(DENSE_Q, k=5, num_threads=4)
    assert (i == i4).all() and (i[0] == i1).all()


def test_k_larger_than_collection_is_padded():
    index = kannolo.build(DENSE[:5])
    d, i = index.search(DENSE_Q[0], k=8)
    assert (i[5:] == -1).all() and np.isinf(d[5:]).all()


def test_dense_queries_against_sparse_index_are_rejected():
    index = kannolo.build(SPARSE)
    expect_error(lambda: index.search(DENSE_Q), "stores sparse vectors", exc=TypeError)


# ----------------------------------------------------------------------------- sparse


SPARSE_ENCODERS = [
    # (encoder, metrics)
    (kannolo.Plain("f32"), ("ip", "l2")),
    (kannolo.Plain("f16"), ("ip", "l2")),
    (kannolo.FixedPoint(8), ("ip", "l2")),
    (kannolo.FixedPoint(16), ("ip", "l2")),
    (kannolo.Scalar(8), ("ip", "l2")),
    (kannolo.Scalar(16), ("ip", "l2")),
    (kannolo.Blocked("f16"), ("ip",)),
    (kannolo.Blocked("scalar_u8"), ("ip",)),
    (kannolo.Blocked("scalar_u16"), ("ip",)),
    (kannolo.DotPacking8("fixed_u8"), ("ip",)),
    (kannolo.DotPacking8("scalar_u8"), ("ip",)),
    (kannolo.DotPacking8("scalar_u16"), ("ip",)),
    (kannolo.DotPacking8("f16"), ("ip",)),
    (kannolo.DotPacking8("scalar_4bit"), ("ip",)),
]


def test_sparse_every_encoder_metric_and_graph():
    for encoder, metrics in SPARSE_ENCODERS:
        for metric in metrics:
            truth = exact_topk(SPARSE_DENSE, SPARSE_Q_DENSE, metric)
            for graph in ("standard", "compressed"):
                index = kannolo.build(SPARSE, metric=metric, encoder=encoder, index=kannolo.HNSW(m=16, graph=graph))
                assert index.spec.representation == kannolo.Sparse("u16")
                _, ids = index.search(SPARSE_Q, k=K, ef_search=200)
                r = recall(ids, truth)
                print(f"sparse {metric} {encoder} {graph}: recall@{K} = {r:.3f}")
                assert r >= SANITY_RECALL, (metric, encoder, graph, r)


def test_sparse_u32_components():
    stride = 97  # ids up to ~194k: does not fit u16
    data, dense = random_sparse(2000, 2000, 40, stride=stride)
    queries, q_dense = random_sparse(50, 2000, 40, seed=1, stride=stride)
    truth = exact_topk(dense, q_dense, "ip")
    for encoder in (kannolo.Plain(), kannolo.FixedPoint(8), kannolo.DotPacking8("scalar_u8")):
        index = kannolo.build(data, encoder=encoder)
        assert index.spec.representation == kannolo.Sparse("u32"), index.spec
        r = recall(index.search(queries, ef_search=200)[1], truth)
        print(f"sparse u32 {encoder}: recall@{K} = {r:.3f}")
        assert r >= SANITY_RECALL
    expect_error(lambda: kannolo.build(data, representation=kannolo.Sparse("u16")), "u32")
    expect_error(lambda: kannolo.build(data, encoder=kannolo.Blocked()), "u16")


def test_sparse_single_query_and_out_of_vocabulary_components():
    index = kannolo.build(SPARSE)
    s, e = SPARSE_Q[2][0], SPARSE_Q[2][1]
    comps, vals = SPARSE_Q[0][s:e], SPARSE_Q[1][s:e]
    d, i = index.search((comps, vals), k=5)
    assert i.shape == (5,)
    # A component no document has cannot change inner products.
    d2, i2 = index.search((np.append(comps, 10**6), np.append(vals, 5.0)), k=5)
    assert (i == i2).all() and np.allclose(d, d2)
    expect_error(lambda: index.search((np.array([3, 3]), np.array([1.0, 1.0]))), "more than once")


def test_scipy_like_csr_input():
    class FakeCSR:  # anything with indptr/indices/data, like scipy.sparse.csr_matrix
        format = "csr"

        def __init__(self, csr):
            self.indices, self.data, self.indptr = csr

    index = kannolo.build(FakeCSR(SPARSE))
    a = index.search(FakeCSR(SPARSE_Q))[1]
    b = index.search(SPARSE_Q)[1]
    assert (a == b).all()


# ----------------------------------------------------------------------------- persistence


def test_save_load_roundtrip_is_self_describing():
    cases = [
        (DENSE, DENSE_Q, dict(metric="l2", encoder=kannolo.PQ(32), index=kannolo.HNSW(m=16, graph="compressed"))),
        (DENSE, DENSE_Q, dict(encoder=kannolo.RaBitQ(bits=4), index=kannolo.HNSW(m=16, graph="permuted"))),
        (SPARSE, SPARSE_Q, dict(encoder=kannolo.DotPacking8(), index=kannolo.HNSW(m=16, graph="compressed"))),
        (SPARSE, SPARSE_Q, dict(metric="l2", encoder=kannolo.Scalar(16))),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for n, (data, queries, kwargs) in enumerate(cases):
            index = kannolo.build(data, **kwargs)
            path = os.path.join(tmp, f"{n}.kannolo")
            index.save(path)
            loaded = kannolo.load(path)
            assert loaded.spec == index.spec
            assert len(loaded) == len(index) and loaded.dim == index.dim and loaded.nbytes == index.nbytes
            d0, i0 = index.search(queries)
            d1, i1 = loaded.search(queries)
            assert (i0 == i1).all() and np.array_equal(d0, d1)
        other = os.path.join(tmp, "other.bin")
        with open(other, "wb") as f:
            f.write(b"\0" * 64)
        expect_error(lambda: kannolo.load(other), "not an index written by kannolo.Index.save", exc=OSError)


def test_file_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        npy, binf = os.path.join(tmp, "d.npy"), os.path.join(tmp, "s.bin")
        np.save(npy, DENSE)
        write_seismic(binf, SPARSE)
        a = kannolo.build(npy, encoder=kannolo.PQ(32))
        assert isinstance(a.spec.representation, kannolo.Dense) and len(a) == len(DENSE)
        b = kannolo.build(binf, encoder=kannolo.FixedPoint(8))
        assert isinstance(b.spec.representation, kannolo.Sparse) and len(b) == SPARSE[2].size - 1
        truth = exact_topk(SPARSE_DENSE, SPARSE_Q_DENSE, "ip")
        assert recall(b.search(SPARSE_Q, ef_search=200)[1], truth) >= SANITY_RECALL


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
