"""Marka ürün feed'lerini "feed" veri seti olarak indirir.

    python scripts/sync_feeds.py --dry-run     # feed'leri okur, marka başına alınacak / atlanacak ürünleri gösterir
    python scripts/sync_feeds.py               # ürünler + görseller → data/feed/images
    python scripts/setup_data.py feed          # eşitle + vektörler + bölüm ayarları

Ayarlar: config/feeds.toml (örnek: config/feeds.example.toml). Feed biçimleri
ve kategori / renk eşlemesi: fashion_rec/feed_source.py. Bir feed okunamazsa
diğerleriyle devam edilir.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fashion_rec import config  # noqa: E402
from fashion_rec.bigquery_source import download_images, write_metadata  # noqa: E402
from fashion_rec.data import list_images  # noqa: E402
from fashion_rec.feed_source import FIELDS, convert, load_settings, parse, read_source  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "feeds.toml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="feed'leri oku ve özetle, görsel indirme")
    args = parser.parse_args()
    if not args.config.exists():
        sys.exit(f"Hata: {args.config} yok; config/feeds.example.toml dosyasını kopyalayıp doldurun")

    rows = []
    for feed in load_settings(args.config):
        try:
            raws = parse(feed, read_source(feed.source))
        except Exception as e:  # bir markanın feed'i bozuksa diğerleri yine alınsın
            print(f"[{feed.brand}] okunamadı: {e}", flush=True)
            continue
        brand_rows, skipped = convert(feed, raws)
        sections = Counter(f"{r['gender']}/{r['subCategory']}" for r in brand_rows)
        print(f"[{feed.brand}] {len(raws)} ürün → {len(brand_rows)} alınacak, {skipped} atlandı "
              f"(moda dışı, cinsiyeti belirsiz, görselsiz, tekrar veya limit üstü)\n    en sık: "
              + ", ".join(f"{k} {n}" for k, n in sections.most_common(8)), flush=True)
        rows += brand_rows
    if not rows:
        sys.exit("Hata: hiçbir feed'den ürün alınamadı")
    if args.dry_run:
        return

    config.set_dataset("feed")
    out_dir = config.DATA_DIR
    print(f"\n{len(rows)} ürünün görseli indiriliyor → {out_dir}", flush=True)
    ok = download_images(rows, out_dir)
    if not ok:
        sys.exit("Hata: hiçbir görsel indirilemedi (feed'deki görsel adreslerini kontrol edin)")
    write_metadata(ok, out_dir, FIELDS)
    # feed'den kalkan ürünlerin görselleri silinir: katalog = metadata.csv'deki ürünler
    kept = {r["file"] for r in ok}
    stale = [p for p in list_images(out_dir) if p.name not in kept]
    for p in stale:
        p.unlink()
    print(f"{len(ok)} / {len(rows)} ürünün görseli indi; metadata.csv yazıldı"
          + (f"; feed'den kalkan {len(stale)} ürün silindi" if stale else ""))


if __name__ == "__main__":
    main()
