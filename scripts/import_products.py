"""Fashion Product Images veri setini proje formatına dönüştürür.

    python scripts/import_products.py

Kaynak: Hugging Face `benitomartin/fashion-product-images-small-384x512`
(Kaggle `paramaggarwal/fashion-product-images-small`, MIT; görseller Myntra'dan).
`data/products/raw/*.parquet` dosyalarını okuyup her ürünü
`data/products/images/<id>.jpg` olarak yazar ve etiketleri
`data/products/images/metadata.csv` dosyasına kaydeder. Var olan görseller
yeniden yazılmaz; script tekrar çalıştırılabilir.

İndirme:
    for i in 0 1 2 3 4; do curl -L -o data/products/raw/train-0000$i-of-00005.parquet \\
      https://huggingface.co/datasets/benitomartin/fashion-product-images-small-384x512/resolve/main/data/train-0000$i-of-00005.parquet; done
"""

import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "products" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "products" / "images"
COLUMNS = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year", "usage"]


def clean(value) -> str:
    """CSV için metin: eksik değer (None / NaN) boş, yıl gibi tam sayı float'lar tam sayı."""
    if value is None or value != value:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def main():
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        sys.exit(f"Hata: {RAW_DIR} altında parquet dosyası yok (indirme komutu için dosyanın başına bakın)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, written = [], 0
    for file in files:
        table = pq.read_table(file, columns=["id", "productDisplayName", "image", *COLUMNS])
        for record in table.to_pylist():
            name = f"{record['id']}.jpg"
            target = OUT_DIR / name
            if not target.exists():
                target.write_bytes(record["image"]["bytes"])
                written += 1
            rows.append({"file": name, "name": clean(record["productDisplayName"]),
                         **{c: clean(record[c]) for c in COLUMNS}})
        print(f"{file.name}: toplam {len(rows)} ürün")

    rows.sort(key=lambda r: r["file"])
    with open(OUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "name", *COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} ürün, {written} yeni görsel yazıldı → {OUT_DIR}")


if __name__ == "__main__":
    main()
