"""CLIP ile sıfır örnekli (zero-shot) renk / tür etiketleri ve etikete göre filtreleme."""

import json
from pathlib import Path

import numpy as np

from . import config
from .features import cached_features
from .text_search import clip_key, load_clip_model, normalize

# grup -> {etiket: CLIP'e verilen tarif}; istemler notebooks/compare_tags.py ile seçildi
# (doğruluk: renk 0,87, tür 0,85)
TAG_GROUPS = {
    "Renk": {
        "siyah": "a black outfit",
        "beyaz": "a white outfit",
        "kırmızı": "a red outfit",
        "mavi": "a blue outfit",
        "yeşil": "a green outfit",
        "pembe": "a pink outfit",
        "sarı": "a yellow outfit",
        "mor": "a purple outfit",
        "bej / kahverengi": "a beige or brown outfit",
        "gri / gümüş": "a grey or silver outfit",
    },
    "Tür": {
        "elbise": "a photo of a woman wearing a dress",
        "etnik": "a photo of a woman wearing traditional Indian clothing such as a saree, kurta or anarkali suit",
        "pantolonlu": "a photo of a woman wearing an outfit with trousers",
        "etekli": "a photo of a woman wearing a top with a skirt",
        "ceket / üst": "a photo of a woman wearing a jacket, coat, blazer or sweater",
    },
}

Tags = dict[str, list[str]]  # grup -> görsel başına etiket (paths ile aynı sırada)


def build_tag_index(paths: list[Path], clip_features: np.ndarray, cache_path: Path | None = None,
                    model=None, force: bool = False) -> Tags:
    """Her görsele her gruptan en uygun etiketi verir; sonuç önbelleğe yazılır.

    Önbellek geçerliyse CLIP modeli yüklenmez. Etiket listesi veya CLIP modeli
    değişince önbellek geçersiz olur.
    """
    def compute():
        clip = model or load_clip_model()
        columns = []
        for prompts in TAG_GROUPS.values():
            text = normalize(clip.encode_text(list(prompts.values())))
            columns.append(np.argmax(clip_features @ text.T, axis=1))
        return np.stack(columns, axis=1)

    key = clip_key() + "/" + json.dumps(TAG_GROUPS, ensure_ascii=False, sort_keys=True)
    indices = cached_features(paths, cache_path or config.TAG_CACHE_PATH, compute, force, key=key)
    return {group: [list(labels)[i] for i in indices[:, g]] for g, (group, labels) in enumerate(TAG_GROUPS.items())}


def tags_by_name(paths: list[Path], tags: Tags) -> dict[str, dict[str, str]]:
    """Dosya adı -> {grup: etiket} sözlüğü."""
    return {p.name: {group: labels[i] for group, labels in tags.items()} for i, p in enumerate(paths)}


def filter_results(results: list[tuple[Path, float]], lookup: dict[str, dict[str, str]],
                   selected: dict[str, list[str]]) -> list[tuple[Path, float]]:
    """Seçili etiketlere uyan sonuçları sırayı bozmadan döndürür.

    Bir grup içindeki etiketler "veya", gruplar arası "ve" ile birleşir;
    boş seçimli grup filtre uygulamaz.
    """
    active = {group: set(labels) for group, labels in selected.items() if labels}
    return [(path, score) for path, score in results
            if all(lookup[path.name][group] in labels for group, labels in active.items())]
