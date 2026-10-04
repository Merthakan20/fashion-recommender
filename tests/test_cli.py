import pytest

import fashion_rec.__main__ as cli
from fashion_rec.features import build_index


@pytest.fixture
def fake_index(data_dir, cache_path, model, monkeypatch):
    """CLI'nın gerçek veri seti yerine test görsellerini kullanmasını sağlar."""
    index = build_index(data_dir, cache_path, model=model)
    monkeypatch.setattr(cli, "build_index", lambda force=False: index)
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
