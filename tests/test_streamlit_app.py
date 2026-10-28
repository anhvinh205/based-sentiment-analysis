from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_renders_and_analyzes(baseline_dir, monkeypatch):
    monkeypatch.setenv("VIABSA_MODEL_PATH", str(baseline_dir))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 4

    at.selectbox[0].select(at.selectbox[0].options[1]).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("background" in m.value for m in at.markdown)


def test_batch_upload_accepts_csv_with_fewer_than_ten_rows(baseline_dir, monkeypatch):
    monkeypatch.setenv("VIABSA_MODEL_PATH", str(baseline_dir))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.file_uploader[0].upload("small.csv", b"data\nson dep\nmau dep\n", "text/csv").run()

    assert not at.exception
    assert at.slider[0].value == 2
