"""BigQuery ürün tablosunu "bq" veri seti olarak indirir (yalnız okuma).

    python scripts/sync_bigquery.py --dry-run          # sorgular + tarayacakları veri (ücretsiz deneme)
    python scripts/sync_bigquery.py [--max-gb 20]      # ürünler + görseller (+ satış varsa birlikte alınma)
    python scripts/setup_data.py bq                    # eşitle + vektörler + bölüm ayarları

Ayarlar: config/bigquery.toml (örnek: config/bigquery.example.toml).
Kimlik: `gcloud auth application-default login`. BigQuery taranan veri
miktarına göre ücretlendirir; sorgular çalışmadan önce tahmini taranacak
veri hesaplanır ve `--max-gb` sınırını aşarsa iptal edilir.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scipy import sparse  # noqa: E402

from fashion_rec import config  # noqa: E402
from fashion_rec.bigquery_source import (download_images, estimate_bytes, fetch_products,  # noqa: E402
                                         fetch_transactions, load_settings, products_query, transactions_query,
                                         write_metadata)
from fashion_rec.copurchase import baskets_matrix, copurchase_matrix  # noqa: E402
from fashion_rec.data import list_images  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "bigquery.toml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="sorguları ve tarayacakları veriyi göster, çalıştırma")
    parser.add_argument("--max-gb", type=float, default=20.0, help="bir sorgunun tarayabileceği en fazla veri (GB)")
    args = parser.parse_args()
    if not args.config.exists():
        sys.exit(f"Hata: {args.config} yok; config/bigquery.example.toml dosyasını kopyalayıp doldurun")

    from google.cloud import bigquery

    settings = load_settings(args.config)
    client = bigquery.Client(project=settings.project)
    queries = {"ürünler": products_query(settings)}
    if settings.transactions_table:
        queries["satışlar"] = transactions_query(settings)

    for label, sql in queries.items():
        gb = estimate_bytes(client, sql) / 1e9
        print(f"\n[{label}] tahmini taranacak veri: {gb:.2f} GB\n{sql}")
        if gb > args.max_gb:
            sys.exit(f"Hata: {gb:.1f} GB > --max-gb {args.max_gb}; sorguyu daraltın (limit / where / "
                     f"transactions_since) veya sınırı bilerek yükseltin")
    if args.dry_run:
        return

    config.set_dataset("bq")
    out_dir = config.DATA_DIR
    rows = fetch_products(client, settings)
    print(f"\n{len(rows)} ürün okundu; görseller indiriliyor → {out_dir}", flush=True)
    ok = download_images(rows, out_dir)
    if not ok:
        sys.exit("Hata: hiçbir görsel indirilemedi (görsel sütununu ve erişim izinlerini kontrol edin)")
    write_metadata(ok, out_dir)
    print(f"{len(ok)} / {len(rows)} ürünün görseli indi; metadata.csv yazıldı")

    if settings.transactions_table:
        paths = list_images(out_dir)
        customer, day, item = fetch_transactions(client, settings, {p.name: i for i, p in enumerate(paths)})
        print(f"katalog ürünlerini içeren satış satırı: {len(item):,}", flush=True)
        if len(item):
            cooc = copurchase_matrix(baskets_matrix(customer, day, item, len(paths)))
            config.COPURCHASE_PATH.parent.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(config.COPURCHASE_PATH, cooc)
            print(f"birlikte alınma: {cooc.nnz // 2:,} ürün çifti → {config.COPURCHASE_PATH}")


if __name__ == "__main__":
    main()
