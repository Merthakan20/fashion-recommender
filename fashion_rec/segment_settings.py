"""Bölüm bazında öneri ayarları (scripts/tune_segments.py ile ölçülüp seçilir).

`segment_settings.json` veri seti -> bölüm -> ayarlar biçimindedir, ör.
{"products": {"Kadın": {"color_weight": 0.2}, ...}}. Ayarı olmayan bölüm
veya veri seti için varsayılanlar kullanılır.
"""

import json
from pathlib import Path

from .config import DATASET

SETTINGS_PATH = Path(__file__).resolve().parent / "segment_settings.json"
DEFAULTS = {"color_weight": 0.0}


def load_settings(path: Path = SETTINGS_PATH, dataset: str = DATASET) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8")).get(dataset, {})


def setting(segment: str | None, key: str, settings: dict[str, dict]) -> float:
    """Bölümün ayarı; bölüm veya ayar yoksa varsayılan."""
    return settings.get(segment or "", {}).get(key, DEFAULTS[key])
