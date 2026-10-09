"""Ürün feed'i kaynağı testleri: ağa bağlanmaz (feed'ler bellekte / geçici dosyada)."""

import csv
from pathlib import Path

import pytest

from fashion_rec.bigquery_source import write_metadata
from fashion_rec.catalog import load_metadata, metadata_tags, segment_of
from fashion_rec.feed_source import (FIELDS, Feed, best_per_brand, category, colour, convert, gender,
                                     load_settings, parse, parse_csv, parse_google, to_metadata)

EXAMPLE = Path(__file__).resolve().parent.parent / "config" / "feeds.example.toml"

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">
  <channel>
    <title>Marka A</title>
    <item>
      <g:id>A-1</g:id>
      <title>Kemerli Midi Elbise</title>
      <link>https://marka-a.example/elbise-a1</link>
      <g:image_link>https://cdn.marka-a.example/a1.jpg</g:image_link>
      <g:price>899.99 TRY</g:price>
      <g:sale_price>649.99 TRY</g:sale_price>
      <g:product_type>Kadın &gt; Giyim &gt; Elbise</g:product_type>
      <g:color>Lacivert</g:color>
    </item>
    <item>
      <g:id>A-2</g:id>
      <title>Basic Tişört</title>
      <link>https://marka-a.example/tisort-a2</link>
      <g:image_link>https://cdn.marka-a.example/a2.jpg</g:image_link>
      <g:gender>male</g:gender>
      <g:age_group>kids</g:age_group>
      <g:product_type>Çocuk &gt; Üst Giyim</g:product_type>
      <g:color>BEYAZ</g:color>
    </item>
    <item>
      <g:id>A-3</g:id>
      <title>Dekoratif Yastık</title>
      <g:image_link>https://cdn.marka-a.example/a3.jpg</g:image_link>
      <g:product_type>Ev &gt; Dekorasyon</g:product_type>
    </item>
  </channel>
</rss>""".encode()

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:g="http://base.google.com/ns/1.0">
  <entry>
    <g:id>B-1</g:id>
    <title>Leather Belt</title>
    <link href="https://brand-b.example/belt"/>
    <g:image_link>https://cdn.brand-b.example/b1.jpg</g:image_link>
    <g:google_product_category>Apparel &amp; Accessories &gt; Clothing Accessories &gt; Belts</g:google_product_category>
  </entry>
</feed>""".encode()


def test_parse_google_rss_reads_g_fields_and_plain_title_link():
    items = parse_google(RSS)
    assert [i["id"] for i in items] == ["A-1", "A-2", "A-3"]
    assert items[0]["title"] == "Kemerli Midi Elbise" and items[0]["link"] == "https://marka-a.example/elbise-a1"
    assert items[0]["product_type"] == "Kadın > Giyim > Elbise" and items[0]["sale_price"] == "649.99 TRY"
    assert items[2]["link"] == ""  # eksik alan boş kalır


def test_parse_google_atom_reads_link_href():
    [item] = parse_google(ATOM)
    assert item["id"] == "B-1" and item["link"] == "https://brand-b.example/belt"


def test_parse_csv_maps_columns():
    data = "urun_kodu;ad;gorsel;kategori\nC-1;Kot Etek;https://c.example/1.jpg;Kadın / Etek\n".encode("utf-8-sig")
    [item] = parse_csv(data, {"id": "urun_kodu", "title": "ad", "image_link": "gorsel", "product_type": "kategori"},
                       delimiter=";")
    assert item["id"] == "C-1" and item["product_type"] == "Kadın / Etek" and item["color"] == ""


@pytest.mark.parametrize("raw, expected", [
    ({"product_type": "Kadın > Giyim > Elbise", "title": "Kemerli Midi Elbise"}, ("Apparel", "Dress", "Dresses")),
    ({"title": "Kot Ceket"}, ("Apparel", "Topwear", "Jackets")),           # kot ama ceket
    ({"title": "Yüksek Bel Kot Etek"}, ("Apparel", "Bottomwear", "Skirts")),
    ({"title": "Slim Fit Kot Pantolon"}, ("Apparel", "Bottomwear", "Jeans")),
    ({"title": "Oversize T-Shirt"}, ("Apparel", "Topwear", "Tshirts")),     # "shirt" içerir ama gömlek değil
    ({"title": "Keten Gömlek"}, ("Apparel", "Topwear", "Shirts")),
    ({"title": "Takım Elbise"}, ("Apparel", "Apparel Set", "Suits")),       # "takı" değil
    ({"title": "Deri Kemer"}, ("Accessories", "Belts", "Belts")),
    ({"title": "Şalvar Pantolon"}, ("Apparel", "Bottomwear", "Trousers")),  # "şal" değil
    ({"product_type": "Ayakkabı > Sneaker"}, ("Footwear", "Shoes", "Sports Shoes")),
    ({"google_product_category": "Apparel & Accessories > Handbags"}, ("Accessories", "Bags", "Handbags")),
    ({"product_type": "Giyim > Kadın > Ceket", "title": "Kemerli Trençkot"}, ("Apparel", "Topwear", "Jackets")),
    ({"title": "ELBİSE"}, ("Apparel", "Dress", "Dresses")),                 # Türkçe büyük harf
    ({"title": "Ribbed Vest Top"}, ("Apparel", "Topwear", "Tops")),        # İngiltere'de "vest" = atlet
    ({"title": "Keten Tunik"}, ("Apparel", "Topwear", "Tunics")),
    ({"title": "Denim Salopet"}, ("Apparel", "Dress", "Jumpsuit")),          # "denim" ama kot pantolon değil
    ({"title": "Kapri Pantolon"}, ("Apparel", "Bottomwear", "Trousers")),
    ({"title": "Eşofman Takımı"}, ("Apparel", "Apparel Set", "Sets")),
    ({"title": "Gümüş Yüzük"}, ("Accessories", "Jewellery", "Jewellery")),
    ({"title": "Earring Set"}, ("Accessories", "Jewellery", "Jewellery")),   # takı seti, giyim takımı değil
])
def test_category_rules(raw, expected):
    assert category(raw) == expected


def test_category_none_for_non_fashion():
    assert category({"product_type": "Ev > Dekorasyon", "title": "Dekoratif Yastık"}) is None


@pytest.mark.parametrize("raw, expected", [
    ({"gender": "female"}, "Women"),
    ({"gender": "male", "age_group": "kids"}, "Boys"),
    ({"product_type": "Kız Çocuk > Elbise"}, "Girls"),
    ({"product_type": "Erkek Çocuk > Tişört"}, "Boys"),
    ({"product_type": "Bebek > Tulum"}, "Kids"),
    ({"product_type": "Erkek > Gömlek"}, "Men"),
    ({"title": "Kadın Bebek Mavisi Gömlek"}, "Women"),                     # renk, çocuk ürünü değil
    ({"title": "Unisex Hoodie", "gender": "unisex"}, "Unisex"),
    ({"title": "Gömlek"}, ""),
])
def test_gender_rules(raw, expected):
    assert gender(raw) == expected


def test_colour_from_field_then_title():
    assert colour({"color": "Koyu Lacivert"}) == "Navy Blue"
    assert colour({"color": "KIRMIZI"}) == "Red"
    assert colour({"title": "Siyah Deri Çanta"}) == "Black"
    assert colour({"color": "Çok renkli"}) == ""


def test_to_metadata_builds_app_row():
    raws = parse_google(RSS)
    row = to_metadata(raws[0], "Marka A")
    assert row["file"] == "marka-a_A-1.jpg" and row["brand"] == "Marka A"
    assert (row["gender"], row["masterCategory"], row["subCategory"], row["baseColour"]) == \
        ("Women", "Apparel", "Dress", "Navy Blue")
    assert row["price"] == "649.99 TRY"  # indirimli fiyat öncelikli
    assert segment_of(row) == "Kadın"
    assert to_metadata(raws[1], "Marka A")["gender"] == "Boys"
    assert to_metadata(raws[2], "Marka A") is None  # moda dışı


def test_to_metadata_skips_unknown_gender_apparel_but_keeps_accessories():
    base = {"id": "1", "image_link": "https://x.example/1.jpg"}
    assert to_metadata({**base, "title": "Gömlek"}, "X") is None
    belt = to_metadata({**base, "title": "Deri Kemer"}, "X")
    assert belt["gender"] == "Unisex" and segment_of(belt) == "Aksesuar"


def test_to_metadata_rejects_non_http_image_and_link():
    raw = {"id": "1", "title": "Kadın Elbise", "image_link": "file:///etc/passwd"}
    assert to_metadata(raw, "X") is None
    raw = {"id": "1", "title": "Kadın Elbise", "image_link": "https://x.example/1.jpg", "link": "javascript:alert(1)"}
    assert to_metadata(raw, "X")["url"] == ""


def test_convert_dedupes_and_limits():
    raws = parse_google(RSS)
    rows, skipped = convert(Feed("Marka A", "x"), raws + raws)
    assert [r["file"] for r in rows] == ["marka-a_A-1.jpg", "marka-a_A-2.jpg"] and skipped == 4
    rows, _ = convert(Feed("Marka A", "x", limit=1), raws)
    assert len(rows) == 1


def test_parse_dispatches_on_format():
    assert len(parse(Feed("A", "x"), RSS)) == 3
    feed = Feed("C", "x", format="csv", columns={"id": "id", "title": "t", "image_link": "i"})
    assert parse(feed, b"id,t,i\n1,Elbise,https://c.example/1.jpg\n")[0]["title"] == "Elbise"


def test_example_settings_are_valid():
    feeds = load_settings(EXAMPLE)
    assert [f.format for f in feeds] == ["google", "csv"]
    assert feeds[1].columns["image_link"] == "gorsel_url" and "gender" not in feeds[1].columns


@pytest.mark.parametrize("bad", [
    dict(brand=""),
    dict(format="json"),
    dict(source=""),
    dict(format="csv", columns={"id": "a", "title": "b"}),                         # image_link eksik
    dict(format="csv", columns={"id": "a", "title": "b", "image_link": "c", "renk": "d"}),  # bilinmeyen alan
])
def test_invalid_feed_settings(bad):
    with pytest.raises(ValueError):
        Feed(**{"brand": "A", "source": "x", **bad}).validate()


def test_written_metadata_has_brand_tags(tmp_path):
    rows, _ = convert(Feed("Marka A", "x"), parse_google(RSS))
    for row in rows:
        (tmp_path / row["file"]).write_bytes(b"")
    path = write_metadata(rows, tmp_path, FIELDS)
    with open(path, newline="", encoding="utf-8") as f:
        assert csv.DictReader(f).fieldnames == ["file", *FIELDS, "image"]
    metadata = load_metadata(tmp_path)
    tags = metadata_tags(sorted(tmp_path.glob("*.jpg")), metadata)
    assert tags["marka-a_A-1.jpg"]["Marka"] == "Marka A"


def test_best_per_brand_groups_in_rank_order():
    results = [(Path(n), s) for n, s in [("a1", .9), ("b1", .8), ("a2", .7), ("a3", .6), ("b2", .5), ("a4", .4)]]
    brands = {"a1": "A", "a2": "A", "a3": "A", "a4": "A", "b1": "B", "b2": "B"}
    groups = best_per_brand(results, brands, per_brand=2)
    assert list(groups) == ["A", "B"]
    assert [p.name for p, _ in groups["A"]] == ["a1", "a2"] and [p.name for p, _ in groups["B"]] == ["b1", "b2"]
