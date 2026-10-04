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
CLIP_MODEL = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
