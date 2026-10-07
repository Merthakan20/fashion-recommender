"""Görselden öneride kullanılan modeli (`config.IMAGE_BACKBONE`) tek yerden seçer."""

from pathlib import Path

import numpy as np

from . import config
from .features import build_index
from .text_search import build_text_index, normalize


def build_image_index(force: bool = False, clip_model=None) -> tuple[list[Path], np.ndarray]:
    """Görselden öneri için (yollar, vektörler): VGG16 veya CLIP görsel vektörleri."""
    if config.IMAGE_BACKBONE == "clip":
        return build_text_index(model=clip_model, force=force)
    return build_index(force=force)


def clip_encoder(clip_model):
    """Katalog dışı görseller için `recommend(encode=...)` fonksiyonu (CLIP backbone'unda)."""
    return lambda paths: normalize(clip_model.encode_images(paths))
