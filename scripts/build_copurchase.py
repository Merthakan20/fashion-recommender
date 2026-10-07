"""H&M satın alma kayıtlarından "birlikte alınanlar" matrisini çıkarır.

    FASHION_DATASET=hm python scripts/build_copurchase.py [--cutoff 2020-06-22]

Kaynak: HF `microsoft/hnm-search-data` → data/processed/transactions_train.parquet
(H&M Kaggle yarışması, 2018-09 – 2020-09, ~31,8 milyon satır). Yalnız
katalogdaki (data/hm/images/metadata.csv) ürünleri içeren satırlar tutulur.

Sepet = aynı müşterinin aynı gün aldıkları (kombin gibi birlikte seçilen
ürünlere en yakın tanım). Ürün çiftlerinin aynı sepette geçme sayısı seyrek
matris çarpımıyla (sepet × ürün)ᵀ(sepet × ürün) bulunur.

- `--cutoff` verilirse yalnız o tarihten önceki sepetler kullanılır ve
  sonraki sepetler değerlendirme için ayrı kaydedilir (notebooks/evaluate_copurchase.py).
- Çıktı: cache/hm/copurchase[_train].npz (ürün × ürün eş-satın alma sayıları,
  satırlar paths sırasıyla) ve cache/hm/test_baskets.npz.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import sparse  # noqa: E402

from fashion_rec.config import CACHE_DIR, DATASET  # noqa: E402
from fashion_rec.data import list_images  # noqa: E402

URL = "https://huggingface.co/datasets/microsoft/hnm-search-data/resolve/main/data/processed/transactions_train.parquet"
RAW = PROJECT_ROOT / "data" / "hm" / "raw" / "transactions_train.parquet"
MAX_BASKET = 30  # toptan alımlar gibi çok büyük sepetler çiftleri şişirmesin


def baskets_matrix(customer, day, item, n_items):
    """(sepet × ürün) ikili seyrek matris; sepet = (müşteri, gün)."""
    keys = np.unique(np.stack([customer, day], axis=1), axis=0, return_inverse=True)[1].ravel()
    m = sparse.csr_matrix((np.ones(len(item), dtype=np.float32), (keys, item)), shape=(keys.max() + 1, n_items))
    m.data[:] = 1  # aynı sepette aynı ürün birden çok kez alınmış olabilir
    sizes = np.asarray(m.sum(axis=1)).ravel()
    return m[(sizes >= 2) & (sizes <= MAX_BASKET)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", help="bu tarihten (YYYY-AA-GG) önceki sepetlerle öğren, sonrasını test için ayır")
    args = parser.parse_args()
    assert DATASET == "hm", "FASHION_DATASET=hm ile çalıştırın"

    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print("satın alma kayıtları indiriliyor...", flush=True)
        urllib.request.urlretrieve(URL, RAW)

    paths = list_images()
    index = {p.stem: i for i, p in enumerate(paths)}
    table = pq.read_table(RAW, columns=["t_dat", "customer_id", "article_id"])
    article = pc.cast(table["article_id"], "string")
    # article_id başındaki sıfır kaybolmuş olabilir: dosya adlarıyla aynı 10 haneye tamamla
    article = pc.utf8_lpad(article, 10, "0")
    keep = pc.is_in(article, value_set=pc.cast(list(index), "string"))
    table, article = table.filter(keep), article.filter(keep)
    print(f"katalog ürünlerini içeren satır: {table.num_rows:,}", flush=True)

    item = np.array([index[a] for a in article.to_pylist()], dtype=np.int64)
    customer = pc.dictionary_encode(table["customer_id"]).combine_chunks().indices.to_numpy()
    day = pc.cast(pc.cast(table["t_dat"], "timestamp[s]"), "int64").to_numpy() // 86400

    def build(mask, name):
        b = baskets_matrix(customer[mask], day[mask], item[mask], len(paths))
        cooc = (b.T @ b).tocsr()
        cooc.setdiag(0)
        cooc.eliminate_zeros()
        sparse.save_npz(CACHE_DIR / name, cooc)
        print(f"{name}: {b.shape[0]:,} sepet, {cooc.nnz // 2:,} ürün çifti", flush=True)
        return b

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.cutoff:
        cut = np.datetime64(args.cutoff).astype("datetime64[D]").astype(np.int64)
        build(day < cut, "copurchase_train.npz")
        test = baskets_matrix(customer[day >= cut], day[day >= cut], item[day >= cut], len(paths))
        sparse.save_npz(CACHE_DIR / "test_baskets.npz", test)
        print(f"test sepetleri ({args.cutoff} ve sonrası): {test.shape[0]:,}")
    else:
        build(np.ones(len(item), dtype=bool), "copurchase.npz")


if __name__ == "__main__":
    main()
