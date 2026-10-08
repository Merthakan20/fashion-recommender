"""BigQuery'den ürün kataloğu ve satın alma kayıtlarını okuma (yalnız okuma).

Uygulama veriyi dosyalardan okur (`metadata.csv` + görseller). Bu modül bir
BigQuery ürün tablosunu o biçime çevirir: `scripts/sync_bigquery.py` ürünleri
ve görsel bağlantılarını okur, görselleri indirir, `data/bq/images/metadata.csv`
yazar; uygulama bunu "bq" veri seti olarak açar. Testler gerçek BigQuery'ye
bağlanmaz (sahte istemci).

Tablo ve sütun adları şirketten şirkete değişir; hepsi bir TOML ayar
dosyasında eşlenir (örnek: config/bigquery.example.toml). Değerler de
eşlenebilir, ör. cinsiyet sütunundaki "Kadın" → uygulamanın beklediği "Women".
Ayar dosyasındaki adlar sorguya eklenmeden önce doğrulanır.
"""

import csv
import io
import re
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# uygulamanın metadata.csv alanları (id ve image zorunlu, gerisi isteğe bağlı)
FIELDS = ["name", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "usage"]
TRANSACTION_FIELDS = ["customer", "day", "product"]
IMAGE_HEIGHT = 512

_TABLE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_$-]+$")   # proje.veri_seti.tablo
_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class BigQuerySettings:
    project: str
    products_table: str
    columns: dict[str, str]                      # uygulama alanı -> BigQuery sütunu
    values: dict[str, dict[str, str]] = field(default_factory=dict)   # alan -> {BigQuery değeri: uygulama değeri}
    where: str = ""                              # isteğe bağlı ek filtre (ör. "is_active = TRUE")
    limit: int = 0
    transactions_table: str = ""
    transaction_columns: dict[str, str] = field(default_factory=dict)
    transactions_since: str = ""                 # YYYY-AA-GG; boşsa tümü (maliyete dikkat)

    def validate(self) -> None:
        for name, table in (("products_table", self.products_table), ("transactions_table", self.transactions_table)):
            if table and not _TABLE.match(table):
                raise ValueError(f"{name} 'proje.veri_seti.tablo' biçiminde olmalı: {table!r}")
        for key in ("id", "image"):
            if not self.columns.get(key):
                raise ValueError(f"[columns] içinde '{key}' sütunu zorunlu")
        for group in (self.columns, self.transaction_columns):
            for app_field, column in group.items():
                if column and not _COLUMN.match(column):
                    raise ValueError(f"geçersiz sütun adı ({app_field}): {column!r}")
        if self.transactions_table and any(not self.transaction_columns.get(k) for k in TRANSACTION_FIELDS):
            raise ValueError(f"[transaction_columns] içinde {', '.join(TRANSACTION_FIELDS)} zorunlu")
        if self.transactions_since and not re.match(r"^\d{4}-\d{2}-\d{2}$", self.transactions_since):
            raise ValueError("transactions_since YYYY-AA-GG biçiminde olmalı")
        if ";" in self.where:
            raise ValueError("where tek bir koşul olmalı (';' içeremez)")


def load_settings(path: Path) -> BigQuerySettings:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    bq = data.get("bigquery", {})
    settings = BigQuerySettings(
        project=bq.get("project", ""),
        products_table=bq.get("products_table", ""),
        columns={k: v for k, v in data.get("columns", {}).items() if v},
        values=data.get("values", {}),
        where=bq.get("where", ""),
        limit=int(bq.get("limit", 0)),
        transactions_table=bq.get("transactions_table", ""),
        transaction_columns={k: v for k, v in data.get("transaction_columns", {}).items() if v},
        transactions_since=bq.get("transactions_since", ""),
    )
    settings.validate()
    return settings


def products_query(settings: BigQuerySettings) -> str:
    """Ürün tablosundan yalnız gereken sütunları, uygulama alan adlarıyla seçen sorgu."""
    selected = [f"CAST(`{settings.columns['id']}` AS STRING) AS id", f"`{settings.columns['image']}` AS image"]
    selected += [f"CAST(`{settings.columns[f]}` AS STRING) AS {f}" for f in FIELDS if f in settings.columns]
    sql = f"SELECT {', '.join(selected)}\nFROM `{settings.products_table}`\nWHERE `{settings.columns['image']}` IS NOT NULL"
    if settings.where:
        sql += f" AND ({settings.where})"
    if settings.limit:
        sql += f"\nLIMIT {settings.limit}"
    return sql


def transactions_query(settings: BigQuerySettings) -> str:
    c = settings.transaction_columns
    sql = (f"SELECT CAST(`{c['customer']}` AS STRING) AS customer, DATE(`{c['day']}`) AS day, "
           f"CAST(`{c['product']}` AS STRING) AS product\nFROM `{settings.transactions_table}`")
    if settings.transactions_since:
        sql += f"\nWHERE DATE(`{c['day']}`) >= DATE '{settings.transactions_since}'"
    return sql


def safe_file_name(product_id: str) -> str:
    """Ürün kimliğinden dosya adı (yalnız harf, rakam, '-', '_')."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", product_id) + ".jpg"


def to_metadata(row: dict, settings: BigQuerySettings) -> dict[str, str]:
    """BigQuery satırını metadata.csv satırına çevirir (değer eşlemeleri uygulanır)."""
    out = {"file": safe_file_name(str(row["id"])), "image": row["image"]}
    for f in FIELDS:
        value = row.get(f)
        value = "" if value is None else str(value).strip()
        out[f] = settings.values.get(f, {}).get(value, value)
    if not out["name"]:
        out["name"] = str(row["id"])
    return out


def fetch_products(client, settings: BigQuerySettings) -> list[dict[str, str]]:
    return [to_metadata(dict(row.items()), settings) for row in client.query(products_query(settings)).result()]


def estimate_bytes(client, sql: str) -> int:
    """Sorgunun tarayacağı veri miktarı (bayt); BigQuery ücreti buna göre (deneme sorgusu, ücretsiz)."""
    from google.cloud import bigquery

    job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    return int(job.total_bytes_processed)


def read_image(uri: str, storage_client=None) -> bytes:
    """https:// veya gs:// adresindeki görselin baytları."""
    if uri.startswith("gs://"):
        bucket, _, blob = uri[5:].partition("/")
        if storage_client is None:
            from google.cloud import storage
            storage_client = storage.Client()
        return storage_client.bucket(bucket).blob(blob).download_as_bytes()
    if uri.startswith(("http://", "https://")):
        with urllib.request.urlopen(uri, timeout=30) as r:
            return r.read()
    raise ValueError(f"desteklenmeyen görsel adresi: {uri!r} (https:// veya gs:// olmalı)")


def download_images(rows: list[dict[str, str]], out_dir: Path, reader=read_image, workers: int = 8) -> list[dict]:
    """Görselleri indirip 512 piksel yüksekliğe küçültür; inenlerin satırlarını döndürür (inmiş olanlar atlanır)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(row):
        target = out_dir / row["file"]
        if target.exists():
            return True
        for attempt in range(3):
            try:
                img = Image.open(io.BytesIO(reader(row["image"]))).convert("RGB")
                img.thumbnail((IMAGE_HEIGHT, IMAGE_HEIGHT))
                img.save(target, quality=90)
                return True
            except Exception:
                time.sleep(2 ** attempt * 0.1)
        return False

    ok = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        for future in as_completed(futures):
            if future.result():
                ok.append(futures[future])
    return sorted(ok, key=lambda r: r["file"])


def write_metadata(rows: list[dict[str, str]], out_dir: Path) -> Path:
    path = out_dir / "metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", *FIELDS, "image"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def fetch_transactions(client, settings: BigQuerySettings, file_index: dict[str, int]):
    """Satın alma kayıtları → (müşteri, gün, ürün sırası) dizileri; katalogda olmayan ürünler atlanır."""
    import numpy as np

    customers, days, items, codes = [], [], [], {}
    for row in client.query(transactions_query(settings)).result():
        i = file_index.get(safe_file_name(str(row["product"])))
        if i is None:
            continue
        customers.append(codes.setdefault(row["customer"], len(codes)))
        days.append(row["day"].toordinal())
        items.append(i)
    return np.array(customers, dtype=np.int64), np.array(days, dtype=np.int64), np.array(items, dtype=np.int64)
