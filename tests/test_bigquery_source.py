"""BigQuery kaynağı testleri: gerçek BigQuery'ye bağlanmaz (sahte istemci)."""

import datetime
import io
from pathlib import Path

import pytest
from PIL import Image

from fashion_rec.bigquery_source import (BigQuerySettings, download_images, fetch_products, fetch_transactions,
                                         load_settings, products_query, safe_file_name, to_metadata,
                                         transactions_query, write_metadata)
from fashion_rec.catalog import load_metadata

EXAMPLE = Path(__file__).resolve().parent.parent / "config" / "bigquery.example.toml"


class FakeClient:
    """client.query(sql).result() → satırlar; gönderilen sorguları kaydeder."""

    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        return self

    def result(self):
        return [dict(r) for r in self.rows]


def settings(**extra):
    base = dict(project="p", products_table="p.ds.products",
                columns={"id": "urun_id", "image": "gorsel", "name": "ad", "gender": "cinsiyet"},
                values={"gender": {"Kadın": "Women"}})
    base.update(extra)
    s = BigQuerySettings(**base)
    s.validate()
    return s


def test_example_settings_are_valid():
    s = load_settings(EXAMPLE)
    assert s.products_table == "proje-adi.veri_seti.urunler"
    assert s.columns["id"] == "urun_id" and "season" not in s.columns  # boş alanlar atlanır
    assert s.values["gender"]["Kadın"] == "Women"


def test_products_query_selects_only_mapped_columns():
    sql = products_query(settings(limit=100, where="aktif = TRUE"))
    assert "FROM `p.ds.products`" in sql
    assert "CAST(`urun_id` AS STRING) AS id" in sql and "`gorsel` AS image" in sql and "AS gender" in sql
    assert "`gorsel` IS NOT NULL AND (aktif = TRUE)" in sql and sql.endswith("LIMIT 100")
    assert "articleType" not in sql


@pytest.mark.parametrize("bad", [
    dict(products_table="ürünler"),                                    # proje.veri_seti.tablo değil
    dict(products_table="p.ds.t; DROP TABLE x"),
    dict(columns={"id": "urun id", "image": "g"}),                     # boşluklu sütun adı
    dict(columns={"id": "urun_id"}),                                   # image zorunlu
    dict(where="1=1; DELETE FROM t"),
    dict(transactions_table="p.ds.sales", transaction_columns={"customer": "m"}),  # eksik satış sütunları
])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        settings(**bad)


def test_to_metadata_maps_values_and_names():
    row = to_metadata({"id": "A/12 3", "image": "https://x/a.jpg", "name": None, "gender": "Kadın"}, settings())
    assert row["file"] == "A_12_3.jpg"           # güvenli dosya adı
    assert row["gender"] == "Women"              # değer eşlemesi
    assert row["name"] == "A/12 3"               # ad yoksa kimlik
    assert safe_file_name("../../etc") == "______etc.jpg"


def test_fetch_products_runs_one_query():
    client = FakeClient([{"id": "1", "image": "https://x/1.jpg", "name": "Elbise", "gender": "Kadın"}])
    rows = fetch_products(client, settings())
    assert len(client.queries) == 1 and rows[0]["gender"] == "Women"


def png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (600, 900), color).save(buf, "PNG")
    return buf.getvalue()


def test_download_images_resizes_skips_existing_and_drops_failures(tmp_path):
    rows = [{"file": "a.jpg", "image": "ok"}, {"file": "b.jpg", "image": "bozuk"}]
    calls = []

    def reader(uri):
        calls.append(uri)
        if uri == "bozuk":
            raise OSError("404")
        return png_bytes((200, 0, 0))

    ok = download_images(rows, tmp_path, reader=reader, workers=2)
    assert [r["file"] for r in ok] == ["a.jpg"]
    assert Image.open(tmp_path / "a.jpg").size[1] == 512        # 512 piksel yüksekliğe küçültüldü
    calls.clear()
    download_images(rows[:1], tmp_path, reader=reader)
    assert calls == []                                           # inmiş görsel tekrar indirilmez


def test_metadata_file_is_readable_by_the_app(tmp_path):
    rows = [to_metadata({"id": "7", "image": "https://x/7.jpg", "name": "Gömlek", "gender": "Kadın"}, settings())]
    write_metadata(rows, tmp_path)
    assert load_metadata(tmp_path)["7.jpg"]["name"] == "Gömlek"


def test_transactions_query_and_fetch():
    s = settings(transactions_table="p.ds.sales", transactions_since="2025-01-01",
                 transaction_columns={"customer": "musteri", "day": "tarih", "product": "urun"})
    sql = transactions_query(s)
    assert "FROM `p.ds.sales`" in sql and "DATE '2025-01-01'" in sql
    day = datetime.date(2025, 3, 1)
    client = FakeClient([{"customer": "c1", "day": day, "product": "1"},
                         {"customer": "c1", "day": day, "product": "2"},
                         {"customer": "c2", "day": day, "product": "999"}])   # katalogda yok → atlanır
    customer, days, items = fetch_transactions(client, s, {"1.jpg": 0, "2.jpg": 1})
    assert list(items) == [0, 1] and list(customer) == [0, 0] and days[0] == day.toordinal()
