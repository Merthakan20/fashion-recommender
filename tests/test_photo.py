from PIL import Image

from fashion_rec.photo import PARTIAL_THRESHOLD, border_ratio, build_photo_index, partial_names


def test_full_product_on_background_vs_cropped(tmp_path):
    full = Image.new("RGB", (100, 150), (240, 240, 240))
    full.paste((20, 20, 120), (25, 30, 75, 120))           # ürün ortada, kenarlar fon
    full.save(tmp_path / "full.png")
    Image.new("RGB", (100, 150), (20, 20, 120)).save(tmp_path / "crop.png")  # ürün tüm kareyi kaplıyor
    assert border_ratio(tmp_path / "full.png") < 0.05
    assert border_ratio(tmp_path / "crop.png") > PARTIAL_THRESHOLD


def test_photo_index_is_cached_and_flags_partial(tmp_path, monkeypatch):
    import fashion_rec.photo as photo

    Image.new("RGB", (100, 150), (240, 240, 240)).save(tmp_path / "a.png")
    Image.new("RGB", (100, 150), (200, 30, 30)).save(tmp_path / "b.png")
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    ratios = build_photo_index(paths, tmp_path / "photo.npz")
    assert partial_names(paths, ratios) == {"b.png"}
    monkeypatch.setattr(photo, "border_ratio", lambda p: (_ for _ in ()).throw(AssertionError("önbellek kullanılmalı")))
    assert list(build_photo_index(paths, tmp_path / "photo.npz")) == list(ratios)
