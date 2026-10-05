"""CLIP ile metinden kıyafet arama."""

from pathlib import Path

import numpy as np

from .adapter import current_adapter
from .config import BATCH_SIZE, CLIP_CACHE_PATH, CLIP_MODEL, CLIP_PRETRAINED, DATA_DIR
from .data import catalog_index, list_images
from .features import cached_features, report_progress
from .recommend import zscore


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def clip_key() -> str:
    """CLIP vektör önbelleklerinin anahtarı: model + (varsa) adaptör imzası."""
    adapter = current_adapter()
    return f"{CLIP_MODEL}/{CLIP_PRETRAINED}" + (f"/adapter-{adapter.signature}" if adapter else "")


class ClipModel:
    """open_clip modelini görsel ve metin kodlama için saran sınıf.

    `adapter="auto"` veri setinin eğitilmiş adaptörünü (varsa) uygular;
    `adapter=None` ham CLIP vektörleri verir (eğitim için).
    """

    def __init__(self, arch: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED, adapter="auto"):
        import open_clip  # torch yavaş açıldığı için import burada

        import torch

        # Apple GPU (MPS) varsa görsel kodlama çok daha hızlı; yoksa işlemci
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.adapter = current_adapter() if adapter == "auto" else adapter

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        batches = []
        with torch.no_grad():
            for start in range(0, len(paths), BATCH_SIZE):
                report_progress(start + BATCH_SIZE, len(paths), "CLIP")
                images = torch.stack([self.preprocess(Image.open(p).convert("RGB"))
                                      for p in paths[start:start + BATCH_SIZE]]).to(self.device)
                batches.append(self.model.encode_image(images).float().cpu().numpy())
        raw = np.concatenate(batches)
        return self.adapter.image(raw) if self.adapter else raw

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            raw = self.model.encode_text(self.tokenizer(texts).to(self.device)).float().cpu().numpy()
        return self.adapter.text(raw) if self.adapter else raw


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
                                  force, key=clip_key())


def search(query: str, paths: list[Path], features: np.ndarray,
           top_n: int = 5, model=None) -> list[tuple[Path, float]]:
    """Metne en çok uyan `top_n` görseli (yol, skor) olarak, azalan skorla döndürür."""
    text = normalize((model or load_clip_model()).encode_text([query]))[0]
    similarities = features @ text
    ranked = np.argsort(similarities)[::-1][:top_n]
    return [(paths[i], float(similarities[i])) for i in ranked]


def combined_search(image_path: Path, text: str, paths: list[Path], features: np.ndarray,
                    top_n: int = 5, text_weight: float = 0.5, model=None) -> list[tuple[Path, float]]:
    """Görsele benzeyen ve metne uyan `top_n` görseli (yol, skor) olarak döndürür ("bunun gibi ama kırmızı").

    Görsel–görsel ve görsel–metin benzerlikleri farklı ölçeklerde olduğundan
    ikisi de z-skoruna çevrilip `text_weight` ile ağırlıklı toplanır
    (0 = yalnız görsel, 1 = yalnız metin). Girdi görseli sonuçlardan çıkarılır.
    """
    if not 0 <= text_weight <= 1:
        raise ValueError("text_weight 0 ile 1 arasında olmalı")

    model = model or load_clip_model()
    query_index = catalog_index(paths)[1].get(Path(image_path).resolve())

    if query_index is not None:
        image = features[query_index]
    else:
        image = normalize(model.encode_images([image_path]))[0]
    text_vec = normalize(model.encode_text([text]))[0]

    scores = (1 - text_weight) * zscore(features @ image) + text_weight * zscore(features @ text_vec)
    ranked = np.argsort(scores)[::-1]
    results = [(paths[i], float(scores[i])) for i in ranked if i != query_index]
    return results[:top_n]
