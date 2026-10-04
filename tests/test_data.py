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
