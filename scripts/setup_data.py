"""Bir veri setini tek komutla kullanıma hazırlar: indirme + vektörler + ayarlar.

    python scripts/setup_data.py women | products | hm

Görseller imaja / depoya konmaz; her kullanıcı kaynağından indirir (H&M ve
Myntra görsellerinin hakları markalara aittir, yeniden dağıtılmaz).

- women:    git'teki 93 görsel; VGG16 + CLIP vektörleri (~1 dk)
- products: Fashion Product Images 44 bin ürün (~2 GB indirme, ~4 GB disk);
            CLIP vektörleri + bölüm ayarları
- hm:       H&M ~26 bin ürün (~1 saat indirme, ~650 MB disk); CLIP vektörleri,
            bölüm ayarları, satın alma kayıtlarından birlikte alınma matrisi
            (774 MB indirilir, işlenince silinir)

Apple GPU (MPS) yoksa CLIP vektörleri işlemcide hesaplanır: 26 bin görsel
için ~15–20 dk, 44 bin için ~30 dk. Her adım tekrar çalıştırılabilir;
tamamlanmış işler (inmiş görseller, geçerli önbellekler) atlanır.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEPS = {
    "women": [["-m", "fashion_rec", "index", "--text"]],
    "products": [["scripts/import_products.py"], ["-m", "fashion_rec", "index"], ["scripts/tune_segments.py"]],
    "hm": [["scripts/import_hm.py", "--limit", "25000"], ["-m", "fashion_rec", "index"],
           ["scripts/tune_segments.py"], ["scripts/build_copurchase.py"]],
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        sys.exit(f"Kullanım: python scripts/setup_data.py {' | '.join(STEPS)}")
    dataset = sys.argv[1]
    env = {**os.environ, "FASHION_DATASET": dataset}
    for step in STEPS[dataset]:
        print(f"\n▶ {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, *step], cwd=PROJECT_ROOT, env=env, check=True)
    raw = PROJECT_ROOT / "data" / dataset / "raw"
    if raw.exists():  # satın alma kayıtları vb.: işlendi, yer kaplamasın
        shutil.rmtree(raw)
    print(f"\n✓ {dataset} hazır. Uygulamada kenar çubuğundaki \"Veri seti\" seçicisinden açılabilir.")


if __name__ == "__main__":
    main()
