import pytest

from fashion_rec.glossary import translate, turkish_lower
from fashion_rec.text_search import build_text_index, combined_search, search


@pytest.mark.parametrize("query, expected", [
    ("kemer", "belt"),
    ("erkek gömleği", "men shirt"),               # ekli hal: gömlek → gömleği
    ("kadın çantası", "women bag"),
    ("kız çocuk elbisesi", "girls dress"),        # çok kelimeli ifade önce
    ("dizüstü bilgisayar çantası", "laptop bag"),
    ("güneş gözlüğü", "sunglasses"),
    ("gümüş rengi elbise", "silver dress"),       # "rengi" atılır
    ("KIRMIZI ELBİSE", "red dress"),              # Türkçe büyük harf: I → ı, İ → i
    ("sarı tişört", "yellow t-shirt"),
    ("sari", "saree"),                            # sarı (renk) ile karışmaz
    ("titan kol saati", "titan wrist watch"),     # sözlükte olmayan kelime kalır
])
def test_translate(query, expected):
    assert translate(query) == expected


@pytest.mark.parametrize("query", ["botanik", "farklı", "red dress", "Titan watch", ""])
def test_translate_returns_none_without_known_terms(query):
    # kısa kökler ("bot", "far") uzun kelimelerde eşleşmez
    assert translate(query) is None


def test_turkish_lower():
    assert turkish_lower("IŞIK İPEK") == "ışık ipek"


def test_search_uses_glossary(data_dir, tmp_path, clip_model):
    # sahte model yalnız İngilizce renk kelimelerini tanır: Türkçe sorgu sözlük sayesinde çalışır
    paths, feats = build_text_index(data_dir, tmp_path / "clip.npz", model=clip_model)
    assert search("kırmızı", paths, feats, top_n=1, model=clip_model)[0][0].name == "red.jpg"
    assert search("yeşil elbise", paths, feats, top_n=1, model=clip_model)[0][0].name == "green.jpg"
    results = combined_search(data_dir / "blue.jpeg", "yeşil", paths, feats, top_n=1, text_weight=1, model=clip_model)
    assert results[0][0].name == "green.jpg"
