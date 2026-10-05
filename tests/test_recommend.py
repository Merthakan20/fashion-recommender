import pytest

from fashion_rec.features import build_index
from fashion_rec.recommend import recommend

from .conftest import COLORS, make_image


@pytest.fixture
def index(data_dir, cache_path, model):
    return build_index(data_dir, cache_path, model=model)


def test_input_image_is_excluded(data_dir, index):
    paths, feats = index
    results = recommend(data_dir / "red.jpg", paths, feats, top_n=10)
    assert "red.jpg" not in [p.name for p, _ in results]


def test_results_sorted_by_descending_score(data_dir, index):
    paths, feats = index
    scores = [s for _, s in recommend(data_dir / "red.jpg", paths, feats, top_n=4)]
    assert scores == sorted(scores, reverse=True)


def test_most_similar_comes_first(data_dir, index):
    paths, feats = index
    assert recommend(data_dir / "red.jpg", paths, feats, top_n=1)[0][0].name == "orange.png"
    assert recommend(data_dir / "blue.jpeg", paths, feats, top_n=1)[0][0].name == "cyan.webp"


@pytest.mark.parametrize("top_n", [1, 2, 4])
def test_returns_exactly_top_n(data_dir, index, top_n):
    paths, feats = index
    assert len(recommend(data_dir / "red.jpg", paths, feats, top_n=top_n)) == top_n


def test_top_n_larger_than_dataset(data_dir, index):
    paths, feats = index
    assert len(recommend(data_dir / "red.jpg", paths, feats, top_n=100)) == len(COLORS) - 1


def test_dataset_image_does_not_need_model(data_dir, index, model):
    paths, feats = index
    calls = model.calls
    recommend(data_dir / "green.jpg", paths, feats, model=model)
    assert model.calls == calls


def test_relative_path_is_recognized(data_dir, index, monkeypatch):
    paths, feats = index
    monkeypatch.chdir(data_dir)
    results = recommend("red.jpg", paths, feats, top_n=10)
    assert "red.jpg" not in [p.name for p, _ in results]


def test_image_outside_dataset(tmp_path, index, model):
    paths, feats = index
    outside = tmp_path / "outside.jpg"
    make_image(outside, (250, 10, 0))  # kırmızıya çok yakın
    results = recommend(outside, paths, feats, top_n=2, model=model)
    assert model.calls > 0
    assert len(results) == 2
    assert results[0][0].name == "red.jpg"


def test_image_outside_dataset_with_custom_encoder(tmp_path, index):
    paths, feats = index
    outside = tmp_path / "outside.jpg"
    make_image(outside, (250, 10, 0))
    calls = []

    def encode(ps):
        calls.append(ps)
        return feats[[0]]  # katalogdaki ilk görselin vektörü

    results = recommend(outside, paths, feats, top_n=1, encode=encode)
    assert len(calls) == 1
    assert results[0][0] == paths[0]
