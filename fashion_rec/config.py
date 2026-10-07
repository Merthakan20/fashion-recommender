"""Proje genelindeki yollar ve model ayarları."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Veri seti FASHION_DATASET ortam değişkeniyle seçilir, ör. `FASHION_DATASET=women streamlit run app.py`
DATASETS = {
    "women": PROJECT_ROOT / "data" / "women fashion",       # 93 kadın kıyafeti (git'te)
    "products": PROJECT_ROOT / "data" / "products" / "images",  # Fashion Product Images, 44 bin ürün (git dışında)
    "hm": PROJECT_ROOT / "data" / "hm" / "images",              # H&M, ~25 bin ürünlük alt küme (git dışında)
}
# varsayılan: 44 bin ürünlük veri seti indirilmişse o, değilse git'teki kadın veri seti
DATASET = os.environ.get("FASHION_DATASET") or ("products" if (DATASETS["products"] / "metadata.csv").exists() else "women")
if DATASET not in DATASETS:
    raise ValueError(f"Bilinmeyen FASHION_DATASET: {DATASET!r} (seçenekler: {', '.join(DATASETS)})")
DATA_DIR = DATASETS[DATASET]
# Görselden öneride kullanılan model: "vgg16" (25.088 boyut, küçük kataloglar) veya "clip" (512 boyut).
# VGG16 vektörleri 44 bin üründe ~4,4 GB tutar; karşılaştırma: notebooks/evaluate_products.py
IMAGE_BACKBONE = {"women": "vgg16", "products": "clip", "hm": "clip"}[DATASET]
METADATA_FILE = "metadata.csv"  # varsa görsellerle aynı klasörde: dosya adı -> ürün adı ve etiketler

# Önbellekler veri setine özel: veri setleri arasında geçiş yapınca yeniden hesaplanmaz
CACHE_DIR = PROJECT_ROOT / "cache" / DATASET
CACHE_PATH = CACHE_DIR / "features.npz"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
IMAGE_SIZE = (224, 224)  # VGG16 giriş boyutu
BATCH_SIZE = 16

# Metinden arama (CLIP)
CLIP_CACHE_PATH = CACHE_DIR / "clip_features.npz"
# Çok dilli (Türkçe dahil) CLIP; karşılaştırma: notebooks/compare_multilingual.py
CLIP_MODEL = "xlm-roberta-base-ViT-B-32"
CLIP_PRETRAINED = "laion5b_s13b_b90k"
# Veri setine özel eğitilmiş adaptör (scripts/train_adapter.py); dosya yoksa CLIP olduğu gibi kullanılır
CLIP_ADAPTER_PATH = CACHE_DIR / "clip_adapter.npz"

# Renk duyarlı benzerlik; karşılaştırma: notebooks/compare_color.py
COLOR_CACHE_PATH = CACHE_DIR / "color_features.npz"
COLOR_BINS = (12, 3, 3)  # HSV: ton, doygunluk, parlaklık
COLOR_CROP = (0.25, 0.15, 0.75, 0.85)  # kıyafetin çoğunlukla durduğu orta bölge (sol, üst, sağ, alt)

# Otomatik etiketler (CLIP, sıfır örnekli); karşılaştırma: notebooks/compare_tags.py
TAG_CACHE_PATH = CACHE_DIR / "tags.npz"

# Kombin önerisi: görsellerin CLIP rol tahmini (fashion_rec/outfit.py)
ROLE_CACHE_PATH = CACHE_DIR / "visual_roles.npz"
# Birlikte alınma matrisi (scripts/build_copurchase.py; şimdilik yalnız H&M)
COPURCHASE_PATH = CACHE_DIR / "copurchase.npz"
