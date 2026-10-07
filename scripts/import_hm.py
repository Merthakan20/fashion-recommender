"""H&M ürünlerinden dengeli bir alt küme indirip proje formatına dönüştürür (öğrenme amaçlı).

    python scripts/import_hm.py [--limit 25000]

Kaynaklar (Hugging Face):
- ürün bilgileri: `Qdrant/hm_ecommerce_products` (H&M Kaggle yarışması
  articles.csv'nin temizlenmiş hali). Bu kopyadaki görsel bağlantıları ölü
  (S3 deposu kaldırılmış), yalnız etiketler kullanılır.
- görseller: `microsoft/hnm-search-data` deposundaki `data/raw/images/…`
  (orijinal yarışma görselleri, 1166×1750).
Haklar H&M'e aittir; Kaggle yarışma kuralları gereği yalnızca ticari olmayan
öğrenme / araştırma için kullanın.

Bölümlere (Kadın / Erkek / Kız Çocuk / Erkek Çocuk / Aksesuar) dengeli bir
alt küme seçilir; aynı ürünün en fazla 2 renk çeşidi alınır. Görseller 512
piksel yüksekliğe küçültülüp `data/hm/images/<article_id>.jpg` olarak yazılır,
etiketler 44 bin ürünlük veri setiyle aynı biçimde `metadata.csv`'ye
(gender / masterCategory / subCategory / articleType / baseColour / usage)
eşlenir, böylece uygulamanın geri kalanı değişmeden çalışır. Script tekrar
çalıştırılabilir; inmiş görseller atlanır.
"""

import argparse
import csv
import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "hm" / "raw" / "hm_ecommerce_products_enriched.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "hm" / "images"
META_URL = "https://huggingface.co/datasets/Qdrant/hm_ecommerce_products/resolve/main/hm_ecommerce_products_enriched.parquet"
IMAGE_URL = "https://huggingface.co/datasets/microsoft/hnm-search-data/resolve/main/data/raw/images/{}/{}.jpg"
TARGETS = {"Kadın": 0.32, "Erkek": 0.24, "Kız Çocuk": 0.14, "Erkek Çocuk": 0.12, "Aksesuar": 0.18}
MAX_PER_PRODUCT, IMAGE_HEIGHT, WORKERS, SEED = 2, 512, 8, 0

ACCESSORY_SUB = {
    "Bag": "Bags", "Earring": "Jewellery", "Necklace": "Jewellery", "Ring": "Jewellery", "Bracelet": "Jewellery",
    "Hat/beanie": "Headwear", "Cap/peaked": "Headwear", "Hat/brim": "Headwear", "Beanie": "Headwear",
    "Scarf": "Scarves", "Sunglasses": "Eyewear", "Belt": "Belts", "Tie": "Ties", "Wallet": "Wallets",
    "Watch": "Watches", "Gloves": "Gloves",
}
GROUP_SUB = {
    "Garment Upper body": "Topwear", "Garment Lower body": "Bottomwear", "Garment Full body": "Dress",
    "Shoes": "Shoes", "Underwear": "Innerwear", "Underwear/nightwear": "Innerwear", "Swimwear": "Swimwear",
    "Socks & Tights": "Socks", "Nightwear": "Loungewear and Nightwear", "Cosmetic": "Makeup",
}
MASTER = {"Accessories": "Accessories", "Bags": "Accessories", "Shoes": "Footwear", "Cosmetic": "Personal Care"}
COLOUR = {"Khaki green": "Olive", "Lilac Purple": "Lavender", "Mole": "Taupe", "Metal": "Metallic",
          "Unknown": "", "undefined": ""}


def gender(row) -> str:
    group, section = row.index_group_name, row.section_name
    if group in ("Ladieswear", "Divided") or (group == "Sport" and section.startswith("Ladies")):
        return "Women"
    if group == "Menswear" or (group == "Sport" and section.startswith("Men")):
        return "Men"
    if group == "Baby/Children":
        if "Girl" in section:
            return "Girls"
        if "Boy" in section:
            return "Boys"
    return ""  # cinsiyeti belirsiz çocuk ürünleri (bebek temel ürünleri, çocuk dış giyim…) alınmaz


def to_metadata(row) -> dict[str, str] | None:
    group = row.product_group_name
    if group in ("Accessories", "Bags"):
        sub = ACCESSORY_SUB.get(row.product_type_name)
    else:
        sub = GROUP_SUB.get(group)
    g = gender(row)
    if not sub or not g:
        return None
    article = row.product_type_name
    denim = "denim" in row.garment_group_name.lower() or "jeans" in row.prod_name.lower()
    if denim and article in ("Trousers", "Shorts"):  # kot: renk uyumunda nötr sayılır
        article = "Jeans" if article == "Trousers" else "Denim Shorts"
    colour = COLOUR.get(row.perceived_colour_master_name, row.perceived_colour_master_name)
    return {
        "file": f"{row.article_id}.jpg",
        "name": f"{row.prod_name} - {row.colour_group_name}",
        "gender": g,
        "masterCategory": MASTER.get(group, "Apparel"),
        "subCategory": sub,
        "articleType": article,
        "baseColour": colour,
        "season": "",
        "year": "",
        "usage": "Sports" if row.index_group_name == "Sport" else "Casual",
    }


def segment(meta) -> str:
    if meta["masterCategory"] == "Accessories":
        return "Aksesuar"
    return {"Women": "Kadın", "Men": "Erkek", "Girls": "Kız Çocuk", "Boys": "Erkek Çocuk"}[meta["gender"]]


def select(limit: int) -> list[dict[str, str]]:
    columns = ["article_id", "product_code", "prod_name", "product_type_name", "product_group_name",
               "colour_group_name", "perceived_colour_master_name", "index_group_name", "section_name",
               "garment_group_name"]
    df = pd.read_parquet(RAW, columns=columns).sample(frac=1, random_state=SEED)
    df = df.groupby("product_code").head(MAX_PER_PRODUCT)  # aynı ürünün en fazla 2 renk çeşidi
    rows = [m for m in (to_metadata(r) for r in df.itertuples()) if m]
    chosen = []
    for name, share in TARGETS.items():
        pool = [m for m in rows if segment(m) == name]
        chosen += pool[:int(limit * share)]
        print(f"{name}: {min(len(pool), int(limit * share))} / {len(pool)} uygun ürün")
    return chosen


def download(meta) -> bool:
    target = OUT_DIR / meta["file"]
    if target.exists():
        return True
    article = meta["file"][:-4]
    url = IMAGE_URL.format(article[:3], article)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
            img.thumbnail((IMAGE_HEIGHT, IMAGE_HEIGHT))  # 1166×1750 → ~341×512
            img.save(target, quality=90)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(2 ** attempt * (10 if e.code == 429 else 1))  # hız sınırında daha uzun bekle
        except Exception:
            time.sleep(2 ** attempt)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25000)
    args = parser.parse_args()

    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print("ürün bilgileri indiriliyor (247 MB)...")
        urllib.request.urlretrieve(META_URL, RAW)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chosen = select(args.limit)

    ok, start = [], time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(download, m): m for m in chosen}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                ok.append(futures[future])
            if i % 1000 == 0 or i == len(chosen):
                print(f"görseller: {i}/{len(chosen)} ({len(ok)} başarılı, {time.time() - start:.0f} sn)",
                      file=sys.stderr, flush=True)

    ok.sort(key=lambda m: m["file"])
    with open(OUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ok[0]))
        writer.writeheader()
        writer.writerows(ok)
    print(f"{len(ok)} ürün → {OUT_DIR}")


if __name__ == "__main__":
    main()
