"""Marka ürün feed'lerinden (affiliate / Google Shopping) çok markalı katalog.

Markalar ürünlerini affiliate ağları (Admitad, Awin, Trendyol / Hepsiburada
affiliate programları…) ve Google Merchant Center üzerinden ürün feed'i
olarak yayınlar: her ürün için kimlik, ad, görsel, mağaza bağlantısı, fiyat,
kategori, cinsiyet ve renk. Bu modül iki yaygın biçimi okur:

- Google Shopping XML (RSS 2.0 `<item>` veya Atom `<entry>`, `g:` alanları),
- CSV (sütun adları ayar dosyasında eşlenir).

Ürünler uygulamanın `metadata.csv` biçimine çevrilir; kategori ve renk
Türkçe / İngilizce anahtar kelimelerle eşlenir ("Kadın > Giyim > Elbise" →
Women / Apparel / Dress). Moda dışı ürünler (ev, kozmetik…) ve cinsiyeti
bulunamayan giyim ürünleri alınmaz. `scripts/sync_feeds.py` feed'leri
indirir, görselleri `bigquery_source.download_images` ile çeker ve "feed"
veri setini yazar. Ayarlar: config/feeds.toml (örnek: config/feeds.example.toml).

Yalnızca kullanım hakkınız olan feed'leri kullanın (affiliate sözleşmesi vb.);
mağaza sitelerinden kazıma (scraping) yapılmaz.
"""

import csv
import io
import re
import tomllib
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .bigquery_source import FIELDS as BASE_FIELDS

# metadata.csv alanları: temel alanlar + marka, mağaza bağlantısı, fiyat
FIELDS = [*BASE_FIELDS, "brand", "url", "price"]
# feed'den okunan ham alanlar (Google Shopping adlarıyla)
RAW_FIELDS = ["id", "title", "link", "image_link", "price", "sale_price", "brand", "gender", "age_group",
              "color", "product_type", "google_product_category"]
FORMATS = ("google", "csv")
_G = "{http://base.google.com/ns/1.0}"
_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class Feed:
    brand: str
    source: str                                   # https://… adresi veya yerel dosya yolu
    format: str = "google"
    columns: dict[str, str] = field(default_factory=dict)   # yalnız csv: ham alan -> CSV sütunu
    limit: int = 0                                # 0 = tümü
    delimiter: str = ","

    def validate(self) -> None:
        if not self.brand.strip():
            raise ValueError("her feed'in bir 'brand' adı olmalı")
        if self.format not in FORMATS:
            raise ValueError(f"{self.brand}: format {' / '.join(FORMATS)} olmalı, {self.format!r} değil")
        if not self.source:
            raise ValueError(f"{self.brand}: 'source' (adres veya dosya yolu) zorunlu")
        if self.format == "csv":
            missing = [k for k in ("id", "title", "image_link") if not self.columns.get(k)]
            if missing:
                raise ValueError(f"{self.brand}: csv feed'inde [columns] içinde {', '.join(missing)} zorunlu")
            unknown = set(self.columns) - set(RAW_FIELDS)
            if unknown:
                raise ValueError(f"{self.brand}: bilinmeyen alan(lar): {', '.join(sorted(unknown))}")


def load_settings(path: Path) -> list[Feed]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    feeds = [Feed(brand=f.get("brand", ""), source=f.get("source", ""), format=f.get("format", "google"),
                  columns={k: v for k, v in f.get("columns", {}).items() if v}, limit=int(f.get("limit", 0)),
                  delimiter=f.get("delimiter", ","))
             for f in data.get("feeds", [])]
    if not feeds:
        raise ValueError("ayar dosyasında [[feeds]] yok")
    for feed in feeds:
        feed.validate()
    return feeds


def read_source(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(source, headers={"User-Agent": "fashion-recommender feed sync"})
        with urllib.request.urlopen(request, timeout=120) as r:
            return r.read()
    return Path(source).expanduser().read_bytes()


def parse_google(data: bytes) -> list[dict[str, str]]:
    """Google Shopping XML → ham ürünler. RSS'te ad / bağlantı düz `title` / `link`, Atom'da `link href`."""
    root = ET.fromstring(data)
    items = root.iter("item") if root.tag == "rss" or root.find("channel") is not None else root.iter(f"{_ATOM}entry")
    products = []
    for item in items:
        def text(name):
            for tag in (f"{_G}{name}", name, f"{_ATOM}{name}"):
                el = item.find(tag)
                if el is not None:
                    return (el.text or el.get("href") or "").strip()
            return ""
        products.append({f: text(f) for f in RAW_FIELDS})
    return products


def parse_csv(data: bytes, columns: dict[str, str], delimiter: str = ",") -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter=delimiter)
    return [{f: (row.get(columns.get(f, ""), "") or "").strip() for f in RAW_FIELDS} for row in reader]


def parse(feed: Feed, data: bytes) -> list[dict[str, str]]:
    return parse_google(data) if feed.format == "google" else parse_csv(data, feed.columns, feed.delimiter)


# Kategori eşlemesi: ilk eşleşen kural kazanır, bu yüzden daha özel kelimeler önce
# (ör. "kot ceket" pantolon değil ceket). (desen, masterCategory, subCategory, articleType)
_CATEGORY_RULES = [
    (r"sneaker|spor ayakkab|trainers?", "Footwear", "Shoes", "Sports Shoes"),
    (r"\bbot\b|\bbotlar|\bçizme|\bboots?\b", "Footwear", "Shoes", "Boots"),
    (r"topuklu|stiletto|\bheels?\b", "Footwear", "Shoes", "Heels"),
    (r"sandalet|sandals?", "Footwear", "Sandal", "Sandals"),
    (r"terlik|slippers?|flip ?flops?", "Footwear", "Flip Flops", "Flip Flops"),
    (r"ayakkab|loafer|babet|ballerina|\bshoes?\b", "Footwear", "Shoes", "Casual Shoes"),
    (r"sırt çanta|backpack", "Accessories", "Bags", "Backpacks"),
    (r"\bçanta|\bbags?\b|handbag|clutch", "Accessories", "Bags", "Handbags"),
    (r"cüzdan|wallets?", "Accessories", "Wallets", "Wallets"),
    (r"\bkemer\b|\bbelts?\b", "Accessories", "Belts", "Belts"),     # "kemerli elbise" elbisedir
    (r"güneş gözlü|gözlük|sunglasses|eyewear", "Accessories", "Eyewear", "Sunglasses"),
    (r"\bsaat\b|\bsaatler|\bwatch(es)?\b", "Accessories", "Watches", "Watches"),
    (r"kolye|küpe|bileklik|yüzük|\btakı\b|necklace|earrings?|bracelets?|\brings?\b|jewel", "Accessories", "Jewellery",
     "Jewellery"),                                                           # "takım" takı değil
    (r"şapka|\bbere\b|kasket|\bhats?\b|\bcaps?\b|beanie", "Accessories", "Headwear", "Caps"),
    (r"\batkı|\bşal\b|eşarp|fular|scarf|scarves|\bstoles?\b|mufflers?", "Accessories", "Scarves", "Scarves"),   # "şalvar" değil
    (r"kravat|\bties?\b", "Accessories", "Ties", "Ties"),
    (r"kol düğme|cufflinks?", "Accessories", "Cufflinks", "Cufflinks"),
    (r"eldiven|\bgloves?\b", "Accessories", "Gloves", "Gloves"),
    (r"iç giyim|sütyen|külot|boxer|pijama|sabahlık|bornoz|underwear|lingerie|\bbra\b|nightwear|pyjama|\brobe\b"
     r"|\bbriefs\b|\btrunks?\b|camisole", "Apparel", "Innerwear",
     "Innerwear"),
    (r"çorap|\bsocks?\b", "Apparel", "Socks", "Socks"),
    (r"mayo|bikini|swim", "Apparel", "Swimwear", "Swimwear"),
    (r"takım elbise|\bsuits?\b", "Apparel", "Apparel Set", "Suits"),
    (r"\btakımı?\b|garment set|\bsets?\b", "Apparel", "Apparel Set", "Sets"),   # eşofman takımı, alt-üst set
    (r"tulum|salopet|jumpsuit|overall|dungarees", "Apparel", "Dress", "Jumpsuit"),
    (r"elbise|\bdress(es)?\b", "Apparel", "Dress", "Dresses"),
    (r"ceket|blazer|yelek|waistcoat|jackets?|\bvests?\b(?! tops?)", "Apparel", "Topwear", "Jackets"),
    (r"\bmont|kaban|parka|trençkot|\bcoats?\b", "Apparel", "Topwear", "Coats"),
    (r"kazak|hırka|triko|sweaters?|cardigans?|knitwear", "Apparel", "Topwear", "Sweaters"),
    (r"sweatshirt|hoodie|kapüşonlu", "Apparel", "Topwear", "Sweatshirts"),
    (r"t-?shirt|tişört|tisort|\bpolo\b", "Apparel", "Topwear", "Tshirts"),   # gömlekten önce: "t-shirt" ⊃ "shirt"
    (r"gömlek|\bshirts?\b", "Apparel", "Topwear", "Shirts"),
    (r"tunik|tunics?|\bkurt[ai]s?\b", "Apparel", "Topwear", "Tunics"),
    (r"\bbluz|blouses?|\btops?\b|\bcrop\b|atlet|büstiyer|\btank\b|\bbody\b|bodysuit", "Apparel", "Topwear",
     "Tops"),
    (r"\betek|skirts?", "Apparel", "Bottomwear", "Skirts"),               # kottan önce: "kot etek" etektir
    (r"\bşort|\bshorts\b|bermuda", "Apparel", "Bottomwear", "Shorts"),
    (r"\bkot\b|\bjeans?\b|denim", "Apparel", "Bottomwear", "Jeans"),
    (r"\btayt|leggings?", "Apparel", "Bottomwear", "Leggings"),
    (r"eşofman alt|jogger|track ?pants?", "Apparel", "Bottomwear", "Track Pants"),
    (r"pantolon|trousers?|\bpants\b|chino|kapri|capris?", "Apparel", "Bottomwear", "Trousers"),
]
_CATEGORY_PATTERNS = [(re.compile(p, re.IGNORECASE), m, s, a) for p, m, s, a in _CATEGORY_RULES]

_COLOURS = {
    "siyah": "Black", "black": "Black", "beyaz": "White", "white": "White", "ekru": "Off White",
    "krem": "Cream", "cream": "Cream", "bej": "Beige", "beige": "Beige", "gri": "Grey", "grey": "Grey",
    "gray": "Grey", "antrasit": "Charcoal", "lacivert": "Navy Blue", "navy": "Navy Blue", "mavi": "Blue",
    "blue": "Blue", "indigo": "Blue", "turkuaz": "Turquoise", "turquoise": "Turquoise", "kırmızı": "Red",
    "red": "Red", "bordo": "Maroon", "burgundy": "Maroon", "maroon": "Maroon", "pembe": "Pink", "pink": "Pink",
    "pudra": "Pink", "mor": "Purple", "purple": "Purple", "lila": "Lavender", "lilac": "Lavender",
    "yeşil": "Green", "green": "Green", "haki": "Olive", "khaki": "Olive", "olive": "Olive",
    "sarı": "Yellow", "yellow": "Yellow", "hardal": "Mustard", "mustard": "Mustard", "turuncu": "Orange",
    "orange": "Orange", "kahverengi": "Brown", "kahve": "Brown", "brown": "Brown", "camel": "Brown",
    "taba": "Tan", "vizon": "Taupe", "altın": "Gold", "gold": "Gold", "gümüş": "Silver", "silver": "Silver",
}

_BABY_COLOURS = re.compile(r"bebek (mavisi|pembesi)|baby (blue|pink)")  # renk, çocuk değil
_KIDS = re.compile(r"çocuk|bebek|kids?|children|baby|infant|toddler", re.IGNORECASE)
_GIRL = re.compile(r"kız|girls?", re.IGNORECASE)
_BOY = re.compile(r"erkek çocuk|oğlan|boys?", re.IGNORECASE)
_WOMEN = re.compile(r"kadın|bayan|women|woman|ladies|female", re.IGNORECASE)
_MEN = re.compile(r"erkek|\bmen\b|\bman\b|male", re.IGNORECASE)


def lower_tr(text: str) -> str:
    """Türkçe küçük harf: "I" → "ı", "İ" → "i" (str.lower "İ"yi "i̇" yapar, "KIRMIZI"yı "kirmizi")."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def category(raw: dict[str, str]) -> tuple[str, str, str] | None:
    """(masterCategory, subCategory, articleType); önce kategori yolu, olmazsa ürün adı. Moda dışıysa None."""
    for text in (raw.get("product_type", ""), raw.get("google_product_category", ""), raw.get("title", "")):
        text = lower_tr(text)
        # kategori yolunun en özel (son) parçasından başla: "Giyim > Kadın > Ceket" → "ceket"
        for part in reversed(re.split(r"\s*[>/|]\s*", text)):
            for pattern, master, sub, article in _CATEGORY_PATTERNS:
                if pattern.search(part):
                    return master, sub, article
    return None


def gender(raw: dict[str, str]) -> str:
    """Women / Men / Girls / Boys / Kids / Unisex; bulunamazsa ''.

    Önce feed'in kendi alanları (g:gender, g:age_group), sonra kategori yolu ve ad.
    """
    g, age = raw.get("gender", "").lower(), raw.get("age_group", "").lower()
    text = lower_tr(" ".join((raw.get("product_type", ""), raw.get("google_product_category", ""),
                              raw.get("title", ""))))
    text = _BABY_COLOURS.sub("", text)
    kids = age in ("kids", "infant", "toddler", "newborn") or bool(_KIDS.search(text))
    if kids:
        if g == "female" or _GIRL.search(text):
            return "Girls"
        if g == "male" or _BOY.search(text):
            return "Boys"
        return "Kids"
    if g == "female":
        return "Women"
    if g == "male":
        return "Men"
    if g == "unisex":
        return "Unisex"
    if _WOMEN.search(text):
        return "Women"
    if _MEN.search(text):
        return "Men"
    return ""


def colour(raw: dict[str, str]) -> str:
    """Feed'in renk alanı (yoksa ad) → İngilizce temel renk; ilk tanınan kelime. Bulunamazsa ''."""
    for text in (raw.get("color", ""), raw.get("title", "")):
        for word in re.findall(r"\w+", lower_tr(text)):
            if word in _COLOURS:
                return _COLOURS[word]
    return ""


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", lower_tr(text).translate(str.maketrans("çğıöşü", "cgiosu"))).strip("-")


def to_metadata(raw: dict[str, str], brand: str) -> dict[str, str] | None:
    """Ham ürün → metadata.csv satırı; görseli / kimliği olmayan, moda dışı veya cinsiyeti belirsiz giyimse None."""
    if not raw.get("id") or not raw.get("image_link", "").startswith(("http://", "https://")):
        return None
    cat = category(raw)
    if cat is None:
        return None
    master, sub, article = cat
    g = gender(raw)
    if not g and master != "Accessories":  # aksesuarlar cinsiyetten bağımsız bölümde
        return None
    return {
        "file": f"{slug(brand)}_{re.sub(r'[^A-Za-z0-9_-]', '_', raw['id'])}.jpg",
        "name": raw.get("title") or raw["id"],
        "gender": g or "Unisex",
        "masterCategory": master,
        "subCategory": sub,
        "articleType": article,
        "baseColour": colour(raw),
        "season": "",
        "usage": "Sports" if re.search(r"spor|sport|athletic", lower_tr(raw.get("product_type", ""))) else "Casual",
        "brand": raw.get("brand") or brand,
        "url": raw.get("link", "") if raw.get("link", "").startswith(("http://", "https://")) else "",
        "price": raw.get("sale_price") or raw.get("price", ""),
        "image": raw["image_link"],
    }


def convert(feed: Feed, raws: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    """Feed'in ürünlerini satırlara çevirir: (satırlar, atlanan sayısı). Aynı dosya adı bir kez alınır."""
    rows, seen = [], set()
    for raw in raws:
        row = to_metadata(raw, feed.brand)
        if row and row["file"] not in seen:
            seen.add(row["file"])
            rows.append(row)
            if feed.limit and len(rows) >= feed.limit:
                break
    return rows, len(raws) - len(rows)


def best_per_brand(results: list[tuple[Path, float]], brand_by_name: dict[str, str],
                   per_brand: int = 3) -> dict[str, list[tuple[Path, float]]]:
    """Sıralı sonuçlardan her markanın en iyi `per_brand` ürünü; markalar en iyi sonuçlarına göre sıralı."""
    grouped: dict[str, list[tuple[Path, float]]] = {}
    for path, score in results:
        items = grouped.setdefault(brand_by_name.get(path.name) or "—", [])
        if len(items) < per_brand:
            items.append((path, score))
    return grouped
