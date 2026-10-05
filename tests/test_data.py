import pytest

from fashion_rec.data import list_images

from .conftest import COLORS, make_image


def test_lists_supported_images_sorted(data_dir):
    names = [p.name for p in list_images(data_dir)]
    assert names == sorted(COLORS)


def test_extension_is_case_insensitive(data_dir):
    make_image(data_dir / "upper.JPG", (10, 10, 10))
    assert "upper.JPG" in [p.name for p in list_images(data_dir)]


def test_ignores_other_files_and_folders(data_dir):
    (data_dir / "notes.txt").write_text("x")
    (data_dir / ".DS_Store").write_bytes(b"")
    (data_dir / "sub.jpg").mkdir()
    assert len(list_images(data_dir)) == len(COLORS)


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "yok")


def test_resolve_all_handles_relative_and_symlinked_dirs(data_dir, tmp_path, monkeypatch):
    from fashion_rec.data import resolve_all

    link = tmp_path / "link"
    link.symlink_to(data_dir)
    monkeypatch.chdir(data_dir)
    paths = [link / "red.jpg", data_dir / "blue.jpeg"]
    from pathlib import Path
    paths.append(Path("green.jpg"))
    assert resolve_all(paths) == [p.resolve() for p in paths]
    absolute = sorted(p.resolve() for p in data_dir.iterdir())
    assert resolve_all(absolute) == absolute


def test_catalog_index_is_cached_per_list(data_dir):
    from fashion_rec.data import catalog_index

    paths = sorted(data_dir.iterdir())
    resolved, index = catalog_index(paths)
    assert index[resolved[2]] == 2
    assert catalog_index(paths)[1] is index  # aynı liste: yeniden hesaplanmaz
    shorter = paths[:-1]
    assert len(catalog_index(shorter)[0]) == len(paths) - 1  # farklı liste: kendi indeksi
