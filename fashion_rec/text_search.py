"""CLIP ile metinden kıyafet arama."""

from pathlib import Path

import numpy as np

from .config import BATCH_SIZE, CLIP_CACHE_PATH, CLIP_MODEL, CLIP_PRETRAINED, DATA_DIR
from .data import list_images
from .features import cached_features


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class ClipModel:
    """open_clip modelini görsel ve metin kodlama için saran sınıf."""

    def __init__(self, arch: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED):
        import open_clip  # torch yavaş açıldığı için import burada

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        batches = []
        with torch.no_grad():
            for start in range(0, len(paths), BATCH_SIZE):
                images = torch.stack([self.preprocess(Image.open(p).convert("RGB"))
                                      for p in paths[start:start + BATCH_SIZE]])
                batches.append(self.model.encode_image(images).numpy())
        return np.concatenate(batches)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.model.encode_text(self.tokenizer(texts)).numpy()


def load_clip_model() -> ClipModel:
    return ClipModel()


def build_text_index(data_dir: Path = DATA_DIR, cache_path: Path = CLIP_CACHE_PATH,
                     model=None, force: bool = False) -> tuple[list[Path], np.ndarray]:
    """Veri setinin (görsel yolları, L2 normalize CLIP vektörleri) ikilisini döndürür.

    Önbellek kuralları `build_index` ile aynıdır.
    """
    paths = list_images(data_dir)
    return paths, cached_features(paths, cache_path,
                                  lambda: normalize((model or load_clip_model()).encode_images(paths)),
                                  force, key=f"{CLIP_MODEL}/{CLIP_PRETRAINED}")


def search(query: str, paths: list[Path], features: np.ndarray,
           top_n: int = 5, model=None) -> list[tuple[Path, float]]:
    """Metne en çok uyan `top_n` görseli (yol, skor) olarak, azalan skorla döndürür."""
    text = normalize((model or load_clip_model()).encode_text([query]))[0]
    similarities = features @ text
    ranked = np.argsort(similarities)[::-1][:top_n]
    return [(paths[i], float(similarities[i])) for i in ranked]
