import csv

from fashion_rec.catalog import display_name, load_metadata


def write_metadata(directory, rows):
    with open(directory / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "name", "gender"])
        writer.writeheader()
        writer.writerows(rows)


def test_without_metadata_file_name_is_used(data_dir):
    metadata = load_metadata(data_dir)
    assert metadata == {}
    assert display_name(data_dir / "red.jpg", metadata) == "Red"


def test_metadata_names_and_fields(data_dir):
    write_metadata(data_dir, [{"file": "red.jpg", "name": "Kırmızı Elbise", "gender": "Women"},
                              {"file": "blue.jpeg", "name": "", "gender": "Men"}])
    metadata = load_metadata(data_dir)
    assert display_name(data_dir / "red.jpg", metadata) == "Kırmızı Elbise"
    assert metadata["red.jpg"]["gender"] == "Women"
    assert display_name(data_dir / "blue.jpeg", metadata) == "Blue"  # boş ad: dosya adına düşer
    assert display_name(data_dir / "green.jpg", metadata) == "Green"  # metadata'da yok


def test_metadata_csv_is_not_listed_as_image(data_dir):
    from fashion_rec.data import list_images

    write_metadata(data_dir, [{"file": "red.jpg", "name": "x", "gender": "Women"}])
    assert all(p.suffix != ".csv" for p in list_images(data_dir))


def test_metadata_tags_translate_and_fill_missing(data_dir):
    from fashion_rec.catalog import metadata_tags

    with open(data_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "name", "gender", "masterCategory", "articleType", "baseColour"])
        writer.writeheader()
        writer.writerow({"file": "red.jpg", "name": "x", "gender": "Girls", "masterCategory": "Accessories",
                         "articleType": "Handbags", "baseColour": "Red"})
    paths = sorted(p for p in data_dir.iterdir() if p.suffix != ".csv")
    lookup = metadata_tags(paths, load_metadata(data_dir))
    assert lookup["red.jpg"] == {"Cinsiyet": "Kız Çocuk", "Kategori": "Aksesuar", "Ürün türü": "Handbags",
                                 "Renk": "Red"}
    assert lookup["blue.jpeg"]["Cinsiyet"] == "—"  # metadata'da olmayan ürün
    assert metadata_tags(paths, {}) == {}
