"""Görsellerden CNN özellik vektörü çıkarma ve önbellekleme."""

import sys
from pathlib import Path

import numpy as np

from .config import BATCH_SIZE, CACHE_PATH, DATA_DIR, IMAGE_SIZE
from .data import list_images


def load_model():
    """ImageNet ağırlıklı, sınıflandırma katmanı olmayan VGG16'yı yükler.

    TensorFlow yavaş açıldığı için import burada yapılır; önbellek
    geçerliyse model hiç yüklenmez.
    """
    from tensorflow.keras.applications import VGG16

    return VGG16(weights="imagenet", include_top=False)


def preprocess_image(path: Path) -> np.ndarray:
    """Görseli VGG16 girişine uygun (224, 224, 3) diziye çevirir."""
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.utils import img_to_array, load_img

    img = load_img(path, target_size=IMAGE_SIZE, color_mode="rgb")
    return preprocess_input(img_to_array(img))


def report_progress(done: int, total: int, label: str) -> None:
    """Büyük kataloglarda (1000+ görsel) her ~%10'da bir ilerlemeyi stderr'e yazar."""
    step = max(BATCH_SIZE, total // 10)
    if total >= 1000 and (done % step < BATCH_SIZE or done >= total):
        print(f"{label}: {min(done, total)}/{total}", file=sys.stderr, flush=True)


def extract_features(model, paths: list[Path]) -> np.ndarray:
    """Her görsel için L2 normalize edilmiş, düzleştirilmiş özellik vektörü döndürür."""
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        report_progress(start + BATCH_SIZE, len(paths), "VGG16")
        batch = np.stack([preprocess_image(p) for p in paths[start:start + BATCH_SIZE]])
        out = model.predict(batch, verbose=0)
        batches.append(out.reshape(len(batch), -1))
    features = np.concatenate(batches)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def build_index(data_dir: Path = DATA_DIR, cache_path: Path = CACHE_PATH,
                model=None, force: bool = False) -> tuple[list[Path], np.ndarray]:
    """Veri setinin (görsel yolları, özellik matrisi) ikilisini döndürür.

    Önbellekteki dosya adları güncel görsel listesiyle aynıysa önbellek
    kullanılır; değilse (veya force=True ise) özellikler yeniden çıkarılır.
    """
    paths = list_images(data_dir)
    return paths, cached_features(paths, cache_path,
                                  lambda: extract_features(model or load_model(), paths), force)


def cached_features(paths: list[Path], cache_path: Path, compute, force: bool = False,
                    key: str = "") -> np.ndarray:
    """Önbellek `paths` ve `key` ile uyuşuyorsa önbelleği, değilse `compute()` sonucunu döndürür.

    `key` vektörleri üreten modeli tanımlar; model değişince aynı boyutlu
    eski vektörlerin sessizce kullanılmasını önler. Yeniden hesaplanan
    özellikler önbelleğe yazılır.
    """
    names = [p.name for p in paths]

    if not force and Path(cache_path).exists():
        cache = np.load(cache_path)
        cached_key = str(cache["key"]) if "key" in cache.files else ""
        if list(cache["names"]) == names and cached_key == key:
            return cache["features"]

    features = compute()
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, names=np.array(names), key=np.array(key))
    return features
