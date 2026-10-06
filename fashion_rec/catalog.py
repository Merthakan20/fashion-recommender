"""Ürün bilgileri: görünen ad ve (varsa) veri setinin kendi etiketleri.

Görsellerle aynı klasörde `metadata.csv` varsa (ilk sütun `file`) ürün adları
ve etiketleri oradan okunur; yoksa dosya adı ürün adı olarak kullanılır
(ör. kadın veri setinde `black floral saree.jpg`).
"""

import csv
import re
from pathlib import Path

from .config import DATA_DIR, METADATA_FILE

Metadata = dict[str, dict[str, str]]  # dosya adı -> {sütun: değer}


def load_metadata(data_dir: Path = DATA_DIR) -> Metadata:
    path = Path(data_dir) / METADATA_FILE
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def display_name(path: Path, metadata: Metadata) -> str:
    """Ürünün okunabilir adı: metadata'daki `name`, yoksa uzantısız dosya adı (ilk harfi büyük)."""
    row = metadata.get(Path(path).name)
    if row and row.get("name"):
        return row["name"]
    stem = Path(path).stem
    return stem[:1].upper() + stem[1:]


# Filtre olarak gösterilecek metadata sütunları: grup adı -> (sütun, Türkçe karşılıklar)
LABEL_GROUPS = {
    "Cinsiyet": ("gender", {"Men": "Erkek", "Women": "Kadın", "Boys": "Erkek Çocuk", "Girls": "Kız Çocuk",
                            "Unisex": "Unisex"}),
    "Kategori": ("masterCategory", {"Apparel": "Giyim", "Accessories": "Aksesuar", "Footwear": "Ayakkabı",
                                    "Personal Care": "Kişisel Bakım", "Free Items": "Hediye",
                                    "Sporting Goods": "Spor Malzemesi", "Home": "Ev"}),
    "Ürün türü": ("articleType", {}),
    "Renk": ("baseColour", {}),
}


_GIRL = re.compile(r"\bgirls?\b|\bgirl's\b", re.IGNORECASE)
_BOY = re.compile(r"\bboys?\b|\bboy's\b", re.IGNORECASE)


def corrected_gender(row: dict[str, str]) -> str:
    """Cinsiyet etiketi; adı açıkça çocuk ürünü diyorsa ona göre düzeltilir.

    Fashion Product Images'ta adında "girl" geçen 185 ürün "Women", "boy"
    geçen 210 ürün "Men" etiketli (ör. "Doodle Girl Little Princess Pink
    Tops"); bu, çocuk ürünlerinin ~%23'ü demek. "boyfriend" gibi kelimeler
    kelime sınırı sayesinde etkilenmez.
    """
    gender = row.get("gender", "")
    name = row.get("name", "")
    if gender in ("Women", "Men", ""):
        if _GIRL.search(name):
            return "Girls"
        if _BOY.search(name):
            return "Boys"
    return gender


def metadata_tags(paths: list[Path], metadata: Metadata) -> dict[str, dict[str, str]]:
    """Dosya adı -> {grup: etiket}; veri setinin kendi etiketlerinden (yoksa boş sözlük).

    `tags.tags_by_name` ile aynı biçimdedir, böylece filtreler iki kaynakla da çalışır.
    """
    if not metadata:
        return {}
    lookup = {}
    for p in paths:
        row = dict(metadata.get(p.name, {}))
        if row:
            row["gender"] = corrected_gender(row)
        lookup[p.name] = {group: translations.get(row.get(column, ""), row.get(column, "")) or "—"
                          for group, (column, translations) in LABEL_GROUPS.items()}
    return lookup


# Bölümler: aksesuarlar cinsiyetten bağımsız tek bölüm, diğer ürünler (giyim, ayakkabı, kişisel bakım) cinsiyete göre
SEGMENTS = ["Kadın", "Erkek", "Kız Çocuk", "Erkek Çocuk", "Aksesuar", "Unisex"]


def segment_of(row: dict[str, str]) -> str | None:
    """Ürünün bölümü; metadata'sı yoksa None."""
    if not row:
        return None
    if row.get("masterCategory") == "Accessories":
        return "Aksesuar"
    return LABEL_GROUPS["Cinsiyet"][1].get(corrected_gender(row)) or None


def segments(paths: list[Path], metadata: Metadata) -> list[str | None]:
    return [segment_of(metadata.get(p.name, {})) for p in paths]


def majority_segment(results: list[tuple[Path, float]], segment_by_name: dict[str, str | None], k: int = 20) -> str | None:
    """Katalog dışı bir görselin bölümü: en yakın `k` komşunun en sık bölümü."""
    from collections import Counter

    votes = Counter(segment_by_name.get(p.name) for p, _ in results[:k])
    votes.pop(None, None)
    return votes.most_common(1)[0][0] if votes else None
