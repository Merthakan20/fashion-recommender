import pytest

import fashion_rec.__main__ as cli
from fashion_rec.features import build_index
from fashion_rec.text_search import build_text_index


@pytest.fixture
def fake_index(data_dir, cache_path, model, monkeypatch):
    """CLI'nın gerçek veri seti yerine test görsellerini kullanmasını sağlar."""
    index = build_index(data_dir, cache_path, model=model)
    monkeypatch.setattr(cli, "build_index", lambda force=False: index)
    return index


@pytest.fixture
def fake_text_index(data_dir, tmp_path, clip_model, monkeypatch):
    index = build_text_index(data_dir, tmp_path / "clip.npz", model=clip_model)
    monkeypatch.setattr(cli, "build_text_index", lambda model=None, force=False: index)
    monkeypatch.setattr(cli, "load_clip_model", lambda: clip_model)
    return index


def test_recommend_prints_ranked_results(data_dir, fake_index, capsys):
    cli.main(["recommend", str(data_dir / "red.jpg"), "-n", "2"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Girdi: red.jpg"
    assert lines[-2].startswith(" 1.") and "orange.png" in lines[-2]
    assert lines[-1].startswith(" 2.")


def test_index_prints_summary(fake_index, capsys):
    cli.main(["index"])
    assert "5 görsel indekslendi" in capsys.readouterr().out


def test_missing_image_exits_with_error(fake_index):
    with pytest.raises(SystemExit) as exc:
        cli.main(["recommend", "yok.jpg"])
    assert "bulunamadı" in str(exc.value)


@pytest.mark.parametrize("n", ["0", "-3", "abc"])
def test_invalid_n_is_rejected(data_dir, n):
    with pytest.raises(SystemExit):
        cli.main(["recommend", str(data_dir / "red.jpg"), "-n", n])


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


def test_search_prints_ranked_results(fake_text_index, capsys):
    cli.main(["search", "  blue dress  ", "-n", "2"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "Sorgu: blue dress"
    assert lines[-2].startswith(" 1.") and "blue.jpeg" in lines[-2]
    assert lines[-1].startswith(" 2.")


def test_empty_search_exits_with_error(fake_text_index):
    with pytest.raises(SystemExit) as exc:
        cli.main(["search", "   "])
    assert "boş" in str(exc.value)


def test_index_text_builds_both_indexes(fake_index, fake_text_index, capsys):
    cli.main(["index", "--text"])
    out = capsys.readouterr().out
    assert "5 görsel indekslendi" in out
    assert "metin araması için indekslendi" in out


def test_combine_prints_ranked_results(data_dir, fake_text_index, capsys):
    cli.main(["combine", str(data_dir / "red.jpg"), "green", "-n", "2"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == 'Girdi: red.jpg + "green" (metin ağırlığı 0.5)'
    assert lines[-2].startswith(" 1.") and "orange.png" in lines[-2]


def test_combine_rejects_empty_text_and_missing_image(data_dir, fake_text_index):
    with pytest.raises(SystemExit) as exc:
        cli.main(["combine", str(data_dir / "red.jpg"), "  "])
    assert "boş" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        cli.main(["combine", "yok.jpg", "red"])
    assert "bulunamadı" in str(exc.value)


@pytest.mark.parametrize("w", ["-0.1", "1.5", "abc"])
def test_combine_invalid_weight_is_rejected(data_dir, w):
    with pytest.raises(SystemExit):
        cli.main(["combine", str(data_dir / "red.jpg"), "red", "-w", w])
