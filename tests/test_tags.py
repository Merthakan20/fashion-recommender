from pathlib import Path

import pytest

import fashion_rec.tags as tags_module
from fashion_rec.tags import build_tag_index, filter_results, tags_by_name
from fashion_rec.text_search import build_text_index

# sahte CLIP modeli metindeki renk kelimesine baktığı için istemler renk kelimeleri
TEST_GROUPS = {
    "Renk": {"kırmızı": "red", "mavi": "blue", "yeşil": "green"},
    "Ton": {"sıcak": "orange", "soğuk": "cyan"},
}


@pytest.fixture(autouse=True)
def test_groups(monkeypatch):
    monkeypatch.setattr(tags_module, "TAG_GROUPS", TEST_GROUPS)


@pytest.fixture
def clip_index(data_dir, tmp_path, clip_model):
    return build_text_index(data_dir, tmp_path / "clip.npz", model=clip_model)


def test_each_image_gets_closest_label(clip_index, tmp_path, clip_model):
    paths, feats = clip_index
    lookup = tags_by_name(paths, build_tag_index(paths, feats, tmp_path / "tags.npz", model=clip_model))
    assert lookup["red.jpg"] == {"Renk": "kırmızı", "Ton": "sıcak"}
    assert lookup["orange.png"]["Renk"] == "kırmızı"
    assert lookup["cyan.webp"] == {"Renk": "mavi", "Ton": "soğuk"}
    assert lookup["green.jpg"]["Renk"] == "yeşil"


def test_tags_are_cached(clip_index, tmp_path, clip_model):
    paths, feats = clip_index
    cache = tmp_path / "tags.npz"
    first = build_tag_index(paths, feats, cache, model=clip_model)
    # model yok: önbellek geçerli olduğu için load_clip_model çağrılmamalı
    assert build_tag_index(paths, feats, cache) == first


def test_changing_tag_groups_invalidates_cache(clip_index, tmp_path, clip_model, monkeypatch):
    paths, feats = clip_index
    cache = tmp_path / "tags.npz"
    build_tag_index(paths, feats, cache, model=clip_model)
    monkeypatch.setattr(tags_module, "TAG_GROUPS", {"Renk": {"kırmızı": "red", "yeşil": "green"}})
    tags = build_tag_index(paths, feats, cache, model=clip_model)
    assert list(tags) == ["Renk"]
    assert set(tags["Renk"]) <= {"kırmızı", "yeşil"}


LOOKUP = {
    "a.jpg": {"Renk": "kırmızı", "Tür": "elbise"},
    "b.jpg": {"Renk": "mavi", "Tür": "elbise"},
    "c.jpg": {"Renk": "kırmızı", "Tür": "etnik"},
}
RESULTS = [(Path(n), s) for n, s in (("c.jpg", 0.9), ("b.jpg", 0.8), ("a.jpg", 0.7))]


def names(results):
    return [p.name for p, _ in results]


def test_filter_without_selection_keeps_everything():
    assert filter_results(RESULTS, LOOKUP, {"Renk": [], "Tür": []}) == RESULTS


def test_filter_keeps_order_and_ands_groups():
    assert names(filter_results(RESULTS, LOOKUP, {"Renk": ["kırmızı"]})) == ["c.jpg", "a.jpg"]
    assert names(filter_results(RESULTS, LOOKUP, {"Renk": ["kırmızı"], "Tür": ["elbise"]})) == ["a.jpg"]


def test_filter_ors_labels_within_group():
    assert names(filter_results(RESULTS, LOOKUP, {"Renk": ["kırmızı", "mavi"], "Tür": ["elbise"]})) == ["b.jpg", "a.jpg"]


def test_filter_can_return_nothing():
    assert filter_results(RESULTS, LOOKUP, {"Tür": ["pantolonlu"]}) == []
