import numpy as np
import pytest

from fashion_rec.text_search import build_text_index, search

from .conftest import COLORS, make_image


@pytest.fixture
def text_index(data_dir, tmp_path, clip_model):
    return build_text_index(data_dir, tmp_path / "clip.npz", model=clip_model)


def test_text_features_are_l2_normalized(text_index):
    paths, feats = text_index
    assert len(paths) == len(feats) == len(COLORS)
    np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("query, expected", [("a red dress", "red.jpg"), ("blue", "blue.jpeg"),
                                             ("GREEN skirt", "green.jpg")])
def test_best_match_comes_first(text_index, clip_model, query, expected):
    paths, feats = text_index
    assert search(query, paths, feats, top_n=1, model=clip_model)[0][0].name == expected


def test_results_sorted_and_top_n(text_index, clip_model):
    paths, feats = text_index
    results = search("red", paths, feats, top_n=3, model=clip_model)
    scores = [s for _, s in results]
    assert len(results) == 3
    assert scores == sorted(scores, reverse=True)


def test_top_n_larger_than_dataset(text_index, clip_model):
    paths, feats = text_index
    assert len(search("red", paths, feats, top_n=100, model=clip_model)) == len(COLORS)


def test_text_index_uses_cache(data_dir, tmp_path, clip_model):
    cache = tmp_path / "clip.npz"
    build_text_index(data_dir, cache, model=clip_model)
    build_text_index(data_dir, cache)  # model yok: önbellek geçerli, load_clip_model çağrılmamalı
    assert clip_model.image_calls == 1

    make_image(data_dir / "yellow.jpg", (255, 255, 0))
    paths, feats = build_text_index(data_dir, cache, model=clip_model)
    assert clip_model.image_calls == 2
    assert len(paths) == len(feats) == len(COLORS) + 1
