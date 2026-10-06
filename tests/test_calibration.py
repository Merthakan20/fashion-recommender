import numpy as np
import pytest

from fashion_rec.calibration import PercentScale, combined_scale, image_scale, sample_paths, text_scale
from fashion_rec.color import build_color_index
from fashion_rec.features import build_index
from fashion_rec.recommend import recommend
from fashion_rec.text_search import build_text_index, search


def test_percent_scale_bounds_and_monotonic():
    scale = PercentScale([0.1, 0.2, 0.3, 0.4])
    assert scale(0.0) == 0
    assert scale(0.25) == 50
    assert scale(0.4) == 100
    assert scale(9.9) == 100
    values = [scale(x) for x in np.linspace(-1, 1, 50)]
    assert values == sorted(values)


def test_percent_scale_rejects_empty_reference():
    with pytest.raises(ValueError):
        PercentScale([])


def test_sample_paths_is_reproducible_and_bounded(data_dir):
    paths = sorted(data_dir.iterdir())
    assert sample_paths(paths, 100) == paths
    assert sample_paths(paths, 2) == sample_paths(paths, 2)
    assert len(sample_paths(paths, 2)) == 2


@pytest.fixture
def index(data_dir, cache_path, model):
    return build_index(data_dir, cache_path, model=model)


def test_image_scale_top_result_beats_typical_pairs(data_dir, index):
    paths, feats = index
    scale = image_scale(paths, feats)
    results = recommend(data_dir / "red.jpg", paths, feats, top_n=len(paths))
    best, worst = scale(results[0][1]), scale(results[-1][1])
    assert best > worst
    assert best >= 90  # kırmızı–turuncu katalogdaki en benzer çiftlerden


def test_image_scale_with_color_weight(data_dir, tmp_path, index):
    paths, feats = index
    _, colors = build_color_index(data_dir, tmp_path / "color.npz")
    scale = image_scale(paths, feats, color_features=colors, color_weight=0.5)
    results = recommend(data_dir / "red.jpg", paths, feats, top_n=3, color_features=colors, color_weight=0.5)
    assert 0 <= scale(results[0][1]) <= 100


def test_text_scale_and_combined_scale(data_dir, tmp_path, clip_model):
    paths, feats = build_text_index(data_dir, tmp_path / "clip.npz", model=clip_model)
    queries = ["red", "blue", "green"]
    scale = text_scale(paths, feats, clip_model, queries)
    good = search("red", paths, feats, top_n=1, model=clip_model)[0][1]
    bad = search("red", paths, feats, top_n=len(paths), model=clip_model)[-1][1]
    assert scale(good) > scale(bad)
    comb = combined_scale(paths, feats, clip_model, queries, sample=2)
    assert 0 <= comb(0.0) <= 100


def test_image_scale_with_subset_uses_only_subset(data_dir, index):
    paths, feats = index
    subset = [p for p in paths if p.name in {"red.jpg", "orange.png", "green.jpg"}]
    scale = image_scale(paths, feats, subset=subset)
    # referans yalnız alt kümedeki çiftlerden: 3 sorgu × 2 eşleşme
    assert len(scale.reference) == 6
