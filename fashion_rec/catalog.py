"""Ürün bilgileri: görünen ad ve (varsa) veri setinin kendi etiketleri.

Görsellerle aynı klasörde `metadata.csv` varsa (ilk sütun `file`) ürün adları
ve etiketleri oradan okunur; yoksa dosya adı ürün adı olarak kullanılır
(ör. kadın veri setinde `black floral saree.jpg`).
"""

import csv
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


def metadata_tags(paths: list[Path], metadata: Metadata) -> dict[str, dict[str, str]]:
    """Dosya adı -> {grup: etiket}; veri setinin kendi etiketlerinden (yoksa boş sözlük).

    `tags.tags_by_name` ile aynı biçimdedir, böylece filtreler iki kaynakla da çalışır.
    """
    if not metadata:
        return {}
    lookup = {}
    for p in paths:
        row = metadata.get(p.name, {})
        lookup[p.name] = {group: translations.get(row.get(column, ""), row.get(column, "")) or "—"
                          for group, (column, translations) in LABEL_GROUPS.items()}
    return lookup
