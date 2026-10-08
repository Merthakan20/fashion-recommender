"""Fashion Product Images veri setini proje formatına dönüştürür.

    python scripts/import_products.py [--keep-raw]

Kaynak: Hugging Face `benitomartin/fashion-product-images-small-384x512`
(Kaggle `paramaggarwal/fashion-product-images-small`, MIT; görseller Myntra'dan).
`data/products/raw/*.parquet` dosyalarını okuyup her ürünü
`data/products/images/<id>.jpg` olarak yazar ve etiketleri
`data/products/images/metadata.csv` dosyasına kaydeder. Var olan görseller
yeniden yazılmaz; script tekrar çalıştırılabilir.

Parquet dosyaları (5 × ~400 MB) yoksa kendisi indirir ve iş bitince siler
(`--keep-raw` ile tutulur).
"""

import argparse
import csv
import shutil
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "products" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "products" / "images"
URL = ("https://huggingface.co/datasets/benitomartin/fashion-product-images-small-384x512/resolve/main/data/"
       "train-0000{}-of-00005.parquet")
COLUMNS = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year", "usage"]


def clean(value) -> str:
    """CSV için metin: eksik değer (None / NaN) boş, yıl gibi tam sayı float'lar tam sayı."""
    if value is None or value != value:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        target = RAW_DIR / f"train-0000{i}-of-00005.parquet"
        if not target.exists():
            print(f"indiriliyor: {target.name} (~400 MB)", flush=True)
            urllib.request.urlretrieve(URL.format(i), target.with_suffix(".part"))
            target.with_suffix(".part").rename(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-raw", action="store_true", help="indirilen parquet dosyalarını silme")
    args = parser.parse_args()
    download()
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        sys.exit(f"Hata: {RAW_DIR} altında parquet dosyası yok")
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
    if not args.keep_raw:
        shutil.rmtree(RAW_DIR)  # görseller çıkarıldı; 2 GB'lık ham dosyalara gerek yok


if __name__ == "__main__":
    main()
