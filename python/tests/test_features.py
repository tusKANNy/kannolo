"""Tests for the parts of the composable API that a build contains (Cargo features `dense`,
`sparse`, `multivec`, `ivf`). The tests run against any build: compiled parts must work, and
missing parts must raise an error that names the feature to enable.

Run with `pytest python/tests` or `python python/tests/test_features.py`.
"""

import numpy as np

import kannolo

FEATURES = kannolo.compiled_features()
RNG = np.random.default_rng(0)
DENSE = RNG.normal(size=(300, 64)).astype(np.float32)
SPARSE = (np.concatenate([RNG.choice(500, size=10, replace=False) for _ in range(300)]),
          RNG.random(300 * 10).astype(np.float32), np.arange(0, 3001, 10))
MULTIVEC = [RNG.normal(size=(int(RNG.integers(2, 6)), 32)).astype(np.float32) for _ in range(300)]


def expect_feature_error(fn, feature):
    try:
        fn()
    except ValueError as e:
        assert f"`{feature}` Cargo feature" in str(e), str(e)
        return
    raise AssertionError(f"expected an error naming the {feature!r} feature")


def test_compiled_features_is_a_subset():
    assert FEATURES <= {"dense", "sparse", "multivec", "ivf"}, FEATURES
    print("compiled:", sorted(FEATURES))


def test_each_representation_works_or_names_its_feature():
    cases = {
        "dense": (DENSE, DENSE[:3], kannolo.Dense()),
        "sparse": (SPARSE, (SPARSE[0][:30], SPARSE[1][:30], np.array([0, 10, 20, 30])), kannolo.Sparse()),
        "multivec": (MULTIVEC, MULTIVEC[:3], kannolo.MultiVector()),
    }
    for feature, (data, queries, rep) in cases.items():
        if feature in FEATURES:
            d, i = kannolo.build(data, index=kannolo.Flat()).search(queries, k=5)
            assert i.shape == (3, 5) and (i >= 0).all()
        else:
            expect_feature_error(lambda: kannolo.check(representation=rep), feature)
            expect_feature_error(lambda: kannolo.build(data, index=kannolo.Flat()), feature)


def test_ivf_works_or_names_its_feature():
    if "dense" not in FEATURES:
        return
    if "ivf" in FEATURES:
        kannolo.build(DENSE, index=kannolo.IVF(n_lists=8)).search(DENSE[:3], k=5, nprobe=4)
    else:
        expect_feature_error(lambda: kannolo.check(index=kannolo.IVF()), "ivf")


def test_a_rescoring_stage_needs_its_feature_too():
    if "dense" in FEATURES and "multivec" not in FEATURES:
        expect_feature_error(lambda: kannolo.check(rerank=kannolo.Rerank(representation=kannolo.MultiVector(), metric="maxsim")), "multivec")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
