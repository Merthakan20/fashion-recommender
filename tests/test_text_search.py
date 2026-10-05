import numpy as np
import pytest

from fashion_rec.text_search import build_text_index, combined_search, search

from .conftest import COLORS, FakeClipModel, make_image


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


def names(results):
    return [p.name for p, _ in results]


def test_combined_weight_zero_is_image_only(data_dir, text_index, clip_model):
    paths, feats = text_index
    results = combined_search(data_dir / "blue.jpeg", "red", paths, feats, top_n=1, text_weight=0, model=clip_model)
    assert names(results) == ["cyan.webp"]


def test_combined_weight_one_is_text_only(data_dir, text_index, clip_model):
    paths, feats = text_index
    results = combined_search(data_dir / "blue.jpeg", "green", paths, feats, top_n=1, text_weight=1, model=clip_model)
    assert names(results) == ["green.jpg"]


def test_combined_finds_image_with_both_traits(data_dir, tmp_path, clip_model):
    # macenta kırmızıya en benzeyen değil (turuncu), "blue" metnine en uyan da değil (mavi);
    # ama ikisini birden taşıyan tek görsel o
    make_image(data_dir / "magenta.jpg", (255, 0, 255))
    paths, feats = build_text_index(data_dir, tmp_path / "clip.npz", model=clip_model)
    red = data_dir / "red.jpg"
    assert names(combined_search(red, "blue", paths, feats, top_n=1, text_weight=0, model=clip_model)) == ["orange.png"]
    assert names(combined_search(red, "blue", paths, feats, top_n=1, text_weight=1, model=clip_model)) == ["blue.jpeg"]
    assert names(combined_search(red, "blue", paths, feats, top_n=1, model=clip_model)) == ["magenta.jpg"]


class CompressedClipModel(FakeClipModel):
    """Gerçek CLIP gibi metin benzerlikleri görsel benzerliklerinden çok daha dar aralıkta.

    Metin vektörüne görsellerde olmayan büyük bir 4. bileşen eklenir; normalize
    edilince renk kısmı küçülür ve tüm metin benzerlikleri ~10 kat sıkışır.
    """

    def encode_images(self, paths):
        feats = super().encode_images(paths)
        return np.hstack([feats, np.zeros((len(feats), 1), dtype=np.float32)])

    def encode_text(self, texts):
        feats = super().encode_text(texts)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return np.hstack([feats, 10 * norms])


def test_combined_balances_different_similarity_scales(data_dir, tmp_path):
    # z-skoru olmadan sıkışık metin benzerlikleri görsel tarafından bastırılırdı (sonuç: turuncu)
    model = CompressedClipModel()
    make_image(data_dir / "magenta.jpg", (255, 0, 255))
    paths, feats = build_text_index(data_dir, tmp_path / "clip.npz", model=model)
    results = combined_search(data_dir / "red.jpg", "blue", paths, feats, top_n=1, model=model)
    assert names(results) == ["magenta.jpg"]


def test_combined_excludes_query_and_sorts(data_dir, text_index, clip_model):
    paths, feats = text_index
    results = combined_search(data_dir / "red.jpg", "red", paths, feats, top_n=10, model=clip_model)
    assert "red.jpg" not in names(results)
    assert len(results) == len(COLORS) - 1
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_combined_image_outside_dataset(tmp_path, text_index, clip_model):
    paths, feats = text_index
    outside = tmp_path / "outside.jpg"
    make_image(outside, (250, 10, 0))
    calls = clip_model.image_calls
    results = combined_search(outside, "red", paths, feats, top_n=1, model=clip_model)
    assert clip_model.image_calls == calls + 1
    assert names(results) == ["red.jpg"]


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_combined_rejects_invalid_weight(data_dir, text_index, clip_model, weight):
    paths, feats = text_index
    with pytest.raises(ValueError):
        combined_search(data_dir / "red.jpg", "red", paths, feats, text_weight=weight, model=clip_model)
