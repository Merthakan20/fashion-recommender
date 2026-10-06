import numpy as np
import pytest

from fashion_rec.diversity import mmr


@pytest.fixture
def catalog(tmp_path):
    # a ve a_copy neredeyse aynı; b farklı yönde ama sorguya hâlâ yakın
    vectors = {"a": [1.0, 0.0, 0.0], "a_copy": [0.999, 0.045, 0.0], "b": [0.8, 0.6, 0.0], "c": [0.0, 0.0, 1.0]}
    paths = [tmp_path / f"{n}.jpg" for n in vectors]
    for p in paths:
        p.touch()
    feats = np.array([np.array(v) / np.linalg.norm(v) for v in vectors.values()])
    return paths, feats


def names(results):
    return [p.stem for p, _ in results]


def ranked(paths, feats, query):
    sims = feats @ query
    return [(paths[i], float(sims[i])) for i in np.argsort(-sims)]


def test_zero_diversity_keeps_order(catalog):
    paths, feats = catalog
    results = ranked(paths, feats, np.array([1.0, 0.0, 0.0]))
    assert mmr(results, paths, feats, 3, diversity=0) == results[:3]


def test_near_duplicate_is_pushed_down(catalog):
    paths, feats = catalog
    query = np.array([0.95, 0.31, 0.0])  # a ve a_copy en yakın, sonra b
    results = ranked(paths, feats, query)
    first = names(results)[0]
    assert set(names(results)[:2]) == {"a", "a_copy"}  # çeşitlilik olmadan iki kopya ilk sırada
    diverse = mmr(results, paths, feats, 2, diversity=0.5, query_vector=query / np.linalg.norm(query))
    assert names(diverse) == [first, "b"]  # ikinci kopya yerine farklı ama ilgili ürün


def test_scores_are_unchanged(catalog):
    paths, feats = catalog
    results = ranked(paths, feats, np.array([1.0, 0.0, 0.0]))
    original = dict(results)
    for p, s in mmr(results, paths, feats, 4, diversity=0.4):
        assert s == original[p]


def test_invalid_diversity(catalog):
    paths, feats = catalog
    with pytest.raises(ValueError):
        mmr(ranked(paths, feats, np.array([1.0, 0, 0])), paths, feats, 2, diversity=1.5)


def test_drop_near_duplicates_removes_copies_even_without_diversity(catalog):
    paths, feats = catalog
    results = ranked(paths, feats, np.array([0.95, 0.31, 0.0]))
    kept = names(mmr(results, paths, feats, 3, diversity=0, drop_near_duplicates=True))
    assert len({"a", "a_copy"} & set(kept)) == 1  # iki kopyadan yalnız biri
    assert "b" in kept
