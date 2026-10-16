"""Opt-in tests against the real Google servers / real models.

  pytest --network                  # fetch the real catalog (~250 KB)
  pytest --models=./models          # translate with installed Hebrew models
"""
from pathlib import Path

import pytest

from offline_translate import Translator
from offline_translate import catalog as cat


@pytest.mark.network
def test_real_manifest_and_profile(tmp_path):
    t = Translator(tmp_path)
    version = t.update_catalog()
    assert tuple(map(int, version.split("."))) >= (8, 5, 0)
    langs = {l.code: l for l in t.languages()}
    assert len(langs) > 90
    assert langs["iw"].supported and 30e6 < langs["iw"].download_size < 120e6
    assert langs["en"].builtin


@pytest.mark.network
def test_real_file_urls_exist(tmp_path):
    t = Translator(tmp_path)
    t.update_catalog()
    pkg = t._catalog()["iw"].packages[0]
    small = min(pkg.files, key=lambda f: f.download_size)
    with cat.open_url(small.url) as r:
        assert r.status == 200


@pytest.fixture(scope="module")
def t(request):
    t = Translator(Path(request.config.getoption("--models")).expanduser())
    if "iw" not in t.installed_languages():
        pytest.skip("Hebrew not installed in --models dir")
    return t


@pytest.mark.models
class TestRealModels:
    def test_short_input_gets_period_treatment(self, t):
        assert t.translate("car", target="he").text == "מכונית"

    def test_en_to_he(self, t):
        assert "שלום" in t.translate("Hello", target="he").text

    def test_he_to_en(self, t):
        r = t.translate("שלום, מה שלומך?", target="en")
        assert r.source == "iw" and "hello" in r.text.lower()

    def test_multi_sentence_and_lines(self, t):
        out = t.translate("Good morning. Thank you.\nGoodbye.", target="he").text
        assert out.count("\n") == 1 and "תודה" in out

    def test_greedy_beam(self, t):
        t.beam = 1
        try:
            assert t.translate("Thank you", target="he").text
        finally:
            t.beam = 4
