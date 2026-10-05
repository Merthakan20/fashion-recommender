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


def resolve_all(paths: list[Path]) -> list[Path]:
    """Yolların mutlak hallerini döndürür; klasörleri bir kez çözümler.

    Her dosya için ayrı `resolve()` büyük kataloglarda (44 bin görsel) her
    sorguda on binlerce dosya sistemi çağrısı demektir. Klasörler zaten
    çözümlenmiş haldeyse (`list_images` çıktısı gibi) liste olduğu gibi döner.
    """
    paths = list(map(Path, paths))
    parents = {p.parent for p in paths}
    resolved = {parent: parent.resolve() for parent in parents}
    if all(parent == real for parent, real in resolved.items()):
        return paths
    return [resolved[p.parent] / p.name for p in paths]


_INDEX_CACHE: dict[int, tuple[list[Path], list[Path], dict[Path, int]]] = {}


def catalog_index(paths: list[Path]) -> tuple[list[Path], dict[Path, int]]:
    """(çözümlenmiş yollar, çözümlenmiş yol -> sıra) ikilisi; aynı liste nesnesi için bir kez hesaplanır.

    Arama fonksiyonları her sorguda girdinin katalogda olup olmadığına bakar;
    44 bin üründe bunu her seferinde baştan yapmak sorgu başına ~0,1 sn eder.
    Liste yerinde değiştirilirse (aynı uzunlukta) önbellek bunu fark etmez;
    katalog listeleri değiştirilmeden kullanılmalıdır.
    """
    entry = _INDEX_CACHE.get(id(paths))
    if entry is None or entry[0] is not paths or len(entry[1]) != len(paths):
        resolved = resolve_all(paths)
        entry = (paths, resolved, {p: i for i, p in enumerate(resolved)})
        if len(_INDEX_CACHE) >= 8:
            _INDEX_CACHE.clear()
        _INDEX_CACHE[id(paths)] = entry
    return entry[1], entry[2]
