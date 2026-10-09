"""Mağazalarda ara: ürünün Türkçe tarifiyle markaların kendi arama sayfalarına bağlantılar.

Sitelerden veri çekilmez (kullanım koşulları kazımayı yasaklar, görseller
markalara aittir); yalnızca ürünün etiketlerinden "kadın siyah elbise" gibi
bir arama metni kurulur ve her mağazanın arama sayfasına bağlantı verilir.
Sonuçları mağazanın kendi araması getirir, yani görsel benzerlik değil metin
eşleşmesidir. Görsel benzerlikle diğer markalardan öneri için ürün feed'leri:
fashion_rec/feed_source.py.

Arama adresleri tarayıcıda denenmiştir (Ekim 2026); siteler adreslerini
değiştirirse STORES güncellenmelidir.
"""

import re
from urllib.parse import quote, quote_plus

from .feed_source import category

# (mağaza, arama adresi şablonu, sorgu kodlaması): {q} sorguyla doldurulur
STORES = [
    ("Zara", "https://www.zara.com/tr/tr/search?searchTerm={q}", quote),
    ("LCW", "https://www.lcw.com/arama?q={q}", quote_plus),
    ("DeFacto", "https://www.defacto.com.tr/arama?q={q}", quote_plus),
    ("Koton", "https://www.koton.com/list/?search_text={q}", quote_plus),
    ("Mavi", "https://www.mavi.com/search?text={q}", quote_plus),
    ("H&M", "https://www2.hm.com/tr_tr/search-results.html?q={q}", quote_plus),
    ("Trendyol", "https://www.trendyol.com/sr?q={q}", quote_plus),
]

GENDER_TR = {"Women": "kadın", "Men": "erkek", "Girls": "kız çocuk", "Boys": "erkek çocuk", "Kids": "çocuk"}
COLOUR_TR = {
    "Black": "siyah", "White": "beyaz", "Off White": "ekru", "Cream": "krem", "Beige": "bej", "Grey": "gri",
    "Grey Melange": "gri", "Charcoal": "antrasit", "Navy Blue": "lacivert", "Blue": "mavi", "Teal": "petrol",
    "Turquoise": "turkuaz", "Turquoise Blue": "turkuaz", "Red": "kırmızı", "Maroon": "bordo", "Burgundy": "bordo",
    "Pink": "pembe", "Peach": "somon", "Purple": "mor", "Lavender": "lila", "Magenta": "fuşya", "Green": "yeşil",
    "Olive": "haki", "Sea Green": "yeşil", "Yellow": "sarı", "Mustard": "hardal", "Orange": "turuncu",
    "Rust": "kiremit", "Brown": "kahverengi", "Coffee Brown": "kahverengi", "Tan": "taba", "Taupe": "vizon",
    "Khaki": "haki", "Gold": "altın", "Silver": "gümüş", "Copper": "bakır", "Bronze": "bronz",
}
# feed_source.category'nin ürün türü adı → Türkçe arama kelimesi
ARTICLE_TR = {
    "Sports Shoes": "spor ayakkabı", "Boots": "bot", "Heels": "topuklu ayakkabı", "Sandals": "sandalet",
    "Flip Flops": "terlik", "Casual Shoes": "ayakkabı", "Backpacks": "sırt çantası", "Handbags": "çanta",
    "Wallets": "cüzdan", "Cufflinks": "kol düğmesi", "Belts": "kemer", "Sunglasses": "güneş gözlüğü", "Watches": "saat", "Jewellery": "takı",
    "Caps": "şapka", "Scarves": "atkı", "Ties": "kravat", "Gloves": "eldiven", "Innerwear": "iç giyim",
    "Socks": "çorap", "Swimwear": "mayo", "Suits": "takım elbise", "Sets": "takım", "Jumpsuit": "tulum",
    "Dresses": "elbise", "Jackets": "ceket", "Coats": "mont", "Sweaters": "kazak", "Sweatshirts": "sweatshirt",
    "Tshirts": "tişört", "Shirts": "gömlek", "Tops": "bluz", "Tunics": "tunik", "Skirts": "etek", "Shorts": "şort", "Jeans": "jean",
    "Leggings": "tayt", "Track Pants": "eşofman altı", "Trousers": "pantolon",
}
# ortak türün genel kelimesinden ("takı", "iç giyim", "kazak") daha net arananlar; kelime başında eşleşir
_SPECIFIC_TR = {"earring": "küpe", "necklace": "kolye", "pendant": "kolye", "ring": "yüzük", "bracelet": "bileklik",
                "bangle": "bileklik", "bra": "sütyen", "bikini": "bikini", "pyjama": "pijama", "night": "gecelik",
                "brief": "külot", "boxer": "boxer", "trunk": "boxer", "cardigan": "hırka", "blazer": "blazer ceket",
                "polo": "polo tişört", "hoodie": "kapüşonlu sweatshirt", "loafer": "loafer", "ballerina": "babet",
                "slipper": "ev terliği", "robe": "sabahlık", "waistcoat": "yelek", "vest top": "atlet"}


def article_tr(row: dict[str, str]) -> str:
    """Ürün türünün Türkçe arama kelimesi; veri setlerinin farklı adları ("Vest top", "Tshirts", "Casual Shoes")
    feed eşlemesiyle ortak türe çevrilir. Bulunamazsa ''."""
    for text in (row.get("articleType", ""), row.get("subCategory", "")):
        for word, turkish in _SPECIFIC_TR.items():
            if re.search(rf"\b{word}", text, re.IGNORECASE):
                return turkish
        cat = category({"title": text}) if text else None
        if cat:
            return ARTICLE_TR.get(cat[2], "")
    return ""


def search_query(row: dict[str, str]) -> str:
    """Ürün etiketlerinden Türkçe arama metni ("kadın siyah elbise"); ürün türü bilinmiyorsa ''.

    Aksesuarlarda cinsiyet yazılmaz (mağazalar çanta / kemeri çoğu zaman cinsiyetsiz listeler).
    """
    article = article_tr(row)
    if not article:
        return ""
    gender = "" if row.get("masterCategory") == "Accessories" else GENDER_TR.get(row.get("gender", ""), "")
    return " ".join(w for w in (gender, COLOUR_TR.get(row.get("baseColour", ""), ""), article) if w)


def store_links(query: str) -> list[tuple[str, str]]:
    """(mağaza, arama adresi) çiftleri; boş sorguda boş liste."""
    query = query.strip()
    if not query:
        return []
    return [(store, template.format(q=encode(query))) for store, template, encode in STORES]
