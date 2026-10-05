"""Proje genelindeki yollar ve model ayarları."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "women fashion"
CACHE_PATH = PROJECT_ROOT / "features.npz"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
IMAGE_SIZE = (224, 224)  # VGG16 giriş boyutu
BATCH_SIZE = 16

# Metinden arama (CLIP)
CLIP_CACHE_PATH = PROJECT_ROOT / "clip_features.npz"
# Çok dilli (Türkçe dahil) CLIP; karşılaştırma: notebooks/compare_multilingual.py
CLIP_MODEL = "xlm-roberta-base-ViT-B-32"
CLIP_PRETRAINED = "laion5b_s13b_b90k"

# Renk duyarlı benzerlik; karşılaştırma: notebooks/compare_color.py
COLOR_CACHE_PATH = PROJECT_ROOT / "color_features.npz"
COLOR_BINS = (12, 3, 3)  # HSV: ton, doygunluk, parlaklık
COLOR_CROP = (0.25, 0.15, 0.75, 0.85)  # kıyafetin çoğunlukla durduğu orta bölge (sol, üst, sağ, alt)

# Otomatik etiketler (CLIP, sıfır örnekli); karşılaştırma: notebooks/compare_tags.py
TAG_CACHE_PATH = PROJECT_ROOT / "tags.npz"
