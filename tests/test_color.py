import numpy as np
import pytest

from fashion_rec.color import build_color_index, color_histogram
from fashion_rec.features import build_index
from fashion_rec.recommend import recommend

from .conftest import COLORS, make_image


def test_histogram_is_normalized_and_peaks_for_solid_color(data_dir):
    v = color_histogram(data_dir / "red.jpg")
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, rtol=1e-6)
    assert np.count_nonzero(v) == 1  # düz renk tek bir kutuya düşer


def test_histogram_ignores_border(tmp_path):
    # kenarlar beyaz, orta bölge kırmızı: histogram düz kırmızıyla aynı olmalı
    from PIL import Image

    img = Image.new("RGB", (100, 100), (255, 255, 255))
    img.paste((255, 0, 0), (25, 15, 75, 85))
    img.save(tmp_path / "framed.png")
    make_image(tmp_path / "solid.png", (255, 0, 0))
    np.testing.assert_allclose(color_histogram(tmp_path / "framed.png"), color_histogram(tmp_path / "solid.png"))


def test_color_index_uses_cache(data_dir, tmp_path, monkeypatch):
    import fashion_rec.color as color_module

    cache = tmp_path / "color.npz"
    paths, feats = build_color_index(data_dir, cache)
    assert len(paths) == len(feats) == len(COLORS)
    monkeypatch.setattr(color_module, "extract_color_features", lambda p: pytest.fail("önbellek kullanılmalıydı"))
    _, again = build_color_index(data_dir, cache)
    np.testing.assert_array_equal(feats, again)


@pytest.fixture
def red_family(data_dir, tmp_path, model):
    """Sahte VGG16 (piksel yönü) koyu kırmızıyı, renk histogramı (ton/doygunluk/parlaklık) açık kırmızıyı seçer."""
    make_image(data_dir / "darkred.jpg", (128, 0, 0))     # piksel yönü kırmızıyla aynı, parlaklığı farklı
    make_image(data_dir / "lightred.jpg", (255, 60, 60))  # piksel yönü biraz farklı, HSV kutusu aynı
    paths, feats = build_index(data_dir, tmp_path / "vgg.npz", model=model)
    _, colors = build_color_index(data_dir, tmp_path / "color.npz")
    return paths, feats, colors


def top(results):
    return results[0][0].name


def order(results):
    return [p.name for p, _ in results]


def test_color_weight_changes_ranking(data_dir, red_family):
    paths, feats, colors = red_family
    red = data_dir / "red.jpg"
    plain = order(recommend(red, paths, feats, top_n=10))
    colored = order(recommend(red, paths, feats, top_n=10, color_features=colors, color_weight=1))
    assert plain[0] == "darkred.jpg"
    assert plain.index("darkred.jpg") < plain.index("lightred.jpg")
    assert colored.index("lightred.jpg") < colored.index("darkred.jpg")


def test_zero_color_weight_keeps_plain_scores(data_dir, red_family):
    paths, feats, colors = red_family
    red = data_dir / "red.jpg"
    assert recommend(red, paths, feats, color_features=colors, color_weight=0) == recommend(red, paths, feats)


def test_color_weight_with_image_outside_dataset(tmp_path, red_family, model):
    paths, feats, colors = red_family
    outside = tmp_path / "outside.jpg"
    make_image(outside, (250, 50, 50))
    results = recommend(outside, paths, feats, top_n=1, model=model, color_features=colors, color_weight=1)
    assert top(results) in {"red.jpg", "lightred.jpg"}  # ikisi de aynı HSV kutusunda


def test_color_weight_validation(data_dir, red_family):
    paths, feats, colors = red_family
    with pytest.raises(ValueError):
        recommend(data_dir / "red.jpg", paths, feats, color_weight=0.5)  # color_features yok
    with pytest.raises(ValueError):
        recommend(data_dir / "red.jpg", paths, feats, color_features=colors, color_weight=1.5)
