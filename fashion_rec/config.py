"""Proje genelindeki yollar ve model ayarları.

Veri setine bağlı değerler (DATA_DIR, önbellek yolları, görsel modeli…)
`set_dataset()` ile çalışma sırasında değiştirilebilir (arayüzdeki veri seti
seçici). Bu yüzden diğer modüller bu değerleri içe aktararak sabitlemez,
çağrı anında `config.X` olarak okur.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "women": PROJECT_ROOT / "data" / "women fashion",       # 93 kadın kıyafeti (git'te)
    "products": PROJECT_ROOT / "data" / "products" / "images",  # Fashion Product Images, 44 bin ürün (git dışında)
    "hm": PROJECT_ROOT / "data" / "hm" / "images",              # H&M, ~25 bin ürünlük alt küme (git dışında)
}
DATASET_LABELS = {"women": "Kadın kıyafetleri (93)", "products": "Fashion Product Images (44 bin)",
                  "hm": "H&M (25 bin, satın alma verili)"}
# Görselden öneride kullanılan model: "vgg16" (25.088 boyut, küçük kataloglar) veya "clip" (512 boyut).
# VGG16 vektörleri 44 bin üründe ~4,4 GB tutar; karşılaştırma: notebooks/evaluate_products.py
IMAGE_BACKBONES = {"women": "vgg16", "products": "clip", "hm": "clip"}
METADATA_FILE = "metadata.csv"  # varsa görsellerle aynı klasörde: dosya adı -> ürün adı ve etiketler

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
IMAGE_SIZE = (224, 224)  # VGG16 giriş boyutu
BATCH_SIZE = 16

# Çok dilli (Türkçe dahil) CLIP; karşılaştırma: notebooks/compare_multilingual.py
CLIP_MODEL = "xlm-roberta-base-ViT-B-32"
CLIP_PRETRAINED = "laion5b_s13b_b90k"

# Renk duyarlı benzerlik; karşılaştırma: notebooks/compare_color.py
COLOR_BINS = (12, 3, 3)  # HSV: ton, doygunluk, parlaklık
COLOR_CROP = (0.25, 0.15, 0.75, 0.85)  # kıyafetin çoğunlukla durduğu orta bölge (sol, üst, sağ, alt)


def available_datasets() -> list[str]:
    """Görselleri indirilmiş veri setleri (kadın veri seti git'te, diğerleri metadata.csv ile)."""
    return [name for name, path in DATASETS.items()
            if path.is_dir() and (name == "women" or (path / METADATA_FILE).exists())]


def set_dataset(name: str) -> None:
    """Veri setine bağlı tüm yolları `name` için ayarlar."""
    if name not in DATASETS:
        raise ValueError(f"Bilinmeyen veri seti: {name!r} (seçenekler: {', '.join(DATASETS)})")
    global DATASET, DATA_DIR, IMAGE_BACKBONE, CACHE_DIR, CACHE_PATH, CLIP_CACHE_PATH, CLIP_ADAPTER_PATH
    global COLOR_CACHE_PATH, TAG_CACHE_PATH, ROLE_CACHE_PATH, COPURCHASE_PATH
    DATASET, DATA_DIR, IMAGE_BACKBONE = name, DATASETS[name], IMAGE_BACKBONES[name]
    # önbellekler veri setine özel: veri setleri arasında geçiş yapınca yeniden hesaplanmaz
    CACHE_DIR = PROJECT_ROOT / "cache" / name
    CACHE_PATH = CACHE_DIR / "features.npz"                 # VGG16
    CLIP_CACHE_PATH = CACHE_DIR / "clip_features.npz"       # metinden arama / CLIP görsel modeli
    CLIP_ADAPTER_PATH = CACHE_DIR / "clip_adapter.npz"      # isteğe bağlı eğitilmiş adaptör (scripts/train_adapter.py)
    COLOR_CACHE_PATH = CACHE_DIR / "color_features.npz"
    TAG_CACHE_PATH = CACHE_DIR / "tags.npz"                 # CLIP otomatik etiketleri (notebooks/compare_tags.py)
    ROLE_CACHE_PATH = CACHE_DIR / "visual_roles.npz"        # kombin önerisi: görsellerin CLIP rol tahmini
    COPURCHASE_PATH = CACHE_DIR / "copurchase.npz"          # birlikte alınma (scripts/build_copurchase.py)


# başlangıç: FASHION_DATASET, yoksa 44 bin ürünlük veri seti indirilmişse o, değilse git'teki kadın veri seti
set_dataset(os.environ.get("FASHION_DATASET")
            or ("products" if (DATASETS["products"] / METADATA_FILE).exists() else "women"))
