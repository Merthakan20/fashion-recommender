"""CLIP vektörleri üzerine eğitilmiş doğrusal adaptör (scripts/train_adapter.py ile eğitilir).

CLIP'in kendisi değişmez; görsel ve metin vektörleri birer doğrusal katmandan
geçirilir: v' = normalize(W·v + b). Adaptör dosyası (`config.CLIP_ADAPTER_PATH`)
varsa `ClipModel` onu otomatik kullanır ve CLIP önbelleklerinin anahtarı
adaptörün imzasını içerir; adaptör değişince vektörler yeniden hesaplanır.
"""

import hashlib
from pathlib import Path

import numpy as np

from .config import CLIP_ADAPTER_PATH


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Adapter:
    def __init__(self, w_image, b_image, w_text, b_text, signature: str = ""):
        self.w_image, self.b_image = np.asarray(w_image, np.float32), np.asarray(b_image, np.float32)
        self.w_text, self.b_text = np.asarray(w_text, np.float32), np.asarray(b_text, np.float32)
        self.signature = signature

    def image(self, x: np.ndarray) -> np.ndarray:
        return _normalize(_normalize(x) @ self.w_image.T + self.b_image)

    def text(self, x: np.ndarray) -> np.ndarray:
        return _normalize(_normalize(x) @ self.w_text.T + self.b_text)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w_image=self.w_image, b_image=self.b_image, w_text=self.w_text, b_text=self.b_text)

    @classmethod
    def load(cls, path: Path) -> "Adapter":
        data = np.load(path)
        signature = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
        return cls(data["w_image"], data["b_image"], data["w_text"], data["b_text"], signature)


def current_adapter(path: Path = CLIP_ADAPTER_PATH) -> Adapter | None:
    """Veri setinin adaptörü varsa onu, yoksa None (CLIP olduğu gibi kullanılır)."""
    return Adapter.load(path) if Path(path).exists() else None
