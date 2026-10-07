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


def test_segment_of_rules():
    from fashion_rec.catalog import segment_of

    assert segment_of({"gender": "Women", "masterCategory": "Apparel"}) == "Kadın"
    assert segment_of({"gender": "Boys", "masterCategory": "Footwear"}) == "Erkek Çocuk"
    assert segment_of({"gender": "Men", "masterCategory": "Accessories"}) == "Aksesuar"  # aksesuar cinsiyetten önce
    assert segment_of({"gender": "Unisex", "masterCategory": "Apparel"}) == "Unisex"
    assert segment_of({}) is None


def test_majority_segment():
    from pathlib import Path

    from fashion_rec.catalog import majority_segment

    results = [(Path(n), 0.0) for n in ("a", "b", "c", "d")]
    lookup = {"a": "Kadın", "b": "Erkek", "c": "Kadın", "d": None}
    assert majority_segment(results, lookup) == "Kadın"
    assert majority_segment(results[3:], lookup) is None


def test_corrected_gender_fixes_kids_items_labelled_as_adults():
    from fashion_rec.catalog import corrected_gender, segment_of

    assert corrected_gender({"gender": "Women", "name": "Doodle Girl Little Princess Pink Tops"}) == "Girls"
    assert corrected_gender({"gender": "Men", "name": "Doodle Boy D Print Red Tshirts"}) == "Boys"
    assert corrected_gender({"gender": "Women", "name": "Palm Tree Girl's Cr Blossom Drs Pink Dresses"}) == "Girls"
    assert corrected_gender({"gender": "Women", "name": "Levis Women Boyfriend Jeans"}) == "Women"  # "boyfriend" değil
    assert corrected_gender({"gender": "Unisex", "name": "Kids Girls Cap"}) == "Unisex"  # yalnız yetişkin etiketleri düzeltilir
    assert segment_of({"gender": "Women", "masterCategory": "Apparel", "name": "Gini and Jony Kids Girls Tops"}) == "Kız Çocuk"


def test_kids_items_fit_both_kids_segments():
    from fashion_rec.catalog import compatible_genders, compatible_segments, segment_of

    assert segment_of({"gender": "Kids", "masterCategory": "Footwear"}) == "Çocuk"
    assert "Çocuk" in compatible_segments("Kız Çocuk") and "Çocuk" in compatible_segments("Erkek Çocuk")
    assert compatible_segments("Kadın") == {"Kadın"}                      # yetişkine çocuk ürünü gelmez
    assert "Kids" in compatible_genders("Girls") and "Kids" not in compatible_genders("Women")
