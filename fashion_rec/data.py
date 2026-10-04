"""Veri setindeki görselleri bulma."""

from pathlib import Path

from .config import DATA_DIR, IMAGE_EXTENSIONS


def list_images(data_dir: Path = DATA_DIR) -> list[Path]:
    """Klasördeki desteklenen görselleri isme göre sıralı döndürür."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {data_dir}")
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
