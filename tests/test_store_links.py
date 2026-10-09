"""Mağaza arama bağlantıları testleri (ağa bağlanmaz)."""

from urllib.parse import parse_qs, urlparse

import pytest

from fashion_rec.store_links import STORES, article_tr, search_query, store_links


@pytest.mark.parametrize("row, expected", [
    ({"gender": "Women", "masterCategory": "Apparel", "articleType": "Dress", "baseColour": "Black"},
     "kadın siyah elbise"),
    ({"gender": "Boys", "masterCategory": "Footwear", "articleType": "Sneakers", "baseColour": "Navy Blue"},
     "erkek çocuk lacivert spor ayakkabı"),
    ({"gender": "Men", "masterCategory": "Apparel", "articleType": "Tshirts", "baseColour": "Metallic"},
     "erkek tişört"),                                                          # karşılığı olmayan renk atlanır
    ({"gender": "Women", "masterCategory": "Accessories", "articleType": "Handbags", "baseColour": "Brown"},
     "kahverengi çanta"),                                                      # aksesuarda cinsiyet yok
    ({"gender": "Women", "masterCategory": "Accessories", "articleType": "Earring", "baseColour": "Gold"},
     "altın küpe"),
    ({"gender": "Women", "masterCategory": "Apparel", "articleType": "Vest top", "baseColour": "White"},
     "kadın beyaz atlet"),
    ({"gender": "Women", "masterCategory": "Apparel", "articleType": "Bra", "baseColour": "Black"},
     "kadın siyah sütyen"),
    ({"gender": "Women", "masterCategory": "Apparel", "articleType": "Cardigan", "baseColour": "Beige"},
     "kadın bej hırka"),
    ({"gender": "Men", "masterCategory": "Apparel", "subCategory": "Bottomwear", "articleType": "Jeans"},
     "erkek jean"),
])
def test_search_query_from_labels(row, expected):
    assert search_query(row) == expected


def test_search_query_empty_when_article_unknown():
    assert search_query({"gender": "Women", "articleType": "Sarees", "baseColour": "Red"}) == ""
    assert search_query({}) == ""


def test_article_falls_back_to_sub_category():
    assert article_tr({"articleType": "Something", "subCategory": "Shoes"}) == "ayakkabı"
    assert article_tr({"articleType": "String bag"}) == "çanta"   # "ring" kelime başında değil: yüzük değil


def test_store_links_encode_query():
    links = dict(store_links("  kadın siyah elbise "))
    assert list(links) == [store for store, _, _ in STORES]
    assert links["Zara"] == "https://www.zara.com/tr/tr/search?searchTerm=kad%C4%B1n%20siyah%20elbise"
    assert parse_qs(urlparse(links["LCW"]).query)["q"] == ["kadın siyah elbise"]
    assert parse_qs(urlparse(links["Koton"]).query)["search_text"] == ["kadın siyah elbise"]
    assert store_links("   ") == []


def test_store_links_cannot_inject_parameters():
    links = dict(store_links("elbise&q=x)#a"))
    assert parse_qs(urlparse(links["LCW"]).query) == {"q": ["elbise&q=x)#a"]}
    assert ")" not in links["Zara"] and "#" not in links["Zara"]
