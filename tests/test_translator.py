import json
import shutil

import pytest

from fakes import FakeGoogle
from offline_translate import Language, TranslateError, Translation, Translator


def langs_by_code(t):
    return {l.code: l for l in t.languages()}


# ---------- catalog ----------
class TestCatalog:
    def test_fresh_folder_fetches_latest_catalog(self, tmp_path, google):
        t = Translator(tmp_path / "models")
        assert t.catalog_version() is None
        t.languages()
        assert t.catalog_version() == "8.5.0"
        assert google.requests == [
            "https://dl.google.com/translate/offline/manifest/gtr/prod/manifest.pb", google.profile_url]

    def test_catalog_is_cached_on_disk(self, tmp_path, google):
        Translator(tmp_path).languages()
        google.offline = True
        assert "iw" in langs_by_code(Translator(tmp_path))

    def test_update_catalog_picks_up_new_version(self, tmp_path, google):
        t = Translator(tmp_path)
        t.languages()
        google.publish(["iw", "fr", "de"], version=(8, 6, 0))
        assert t.update_catalog() == "8.6.0"
        assert "de" in langs_by_code(t)
        assert Translator(tmp_path).catalog_version() == "8.6.0"

    def test_channel(self, tmp_path, monkeypatch):
        g = FakeGoogle(channel="beta").install(monkeypatch)
        Translator(tmp_path, channel="beta").update_catalog()
        assert "/gtr/beta/" in g.requests[0]

    def test_profile_checksum_is_verified(self, tmp_path, google):
        google.corrupt.add(google.profile_url)
        with pytest.raises(TranslateError, match="checksum"):
            Translator(tmp_path).update_catalog()
        assert Translator(tmp_path).catalog_version() is None

    def test_offline_without_cache_raises_oserror(self, tmp_path, google):
        google.offline = True
        with pytest.raises(OSError):
            Translator(tmp_path).languages()

    def test_models_dir_is_created_and_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        t = Translator("~/a/b")
        assert t.dir == tmp_path / "a" / "b" and t.dir.is_dir()


# ---------- languages ----------
class TestLanguages:
    def test_english_is_builtin_everything_else_not_installed(self, tmp_path, google):
        langs = langs_by_code(Translator(tmp_path))
        assert langs["en"] == Language("en", "English", True, 0, 0, builtin=True)
        assert not langs["iw"].installed and langs["iw"].name == "Hebrew"
        assert langs["iw"].download_size > 0 and langs["iw"].size > 0
        assert not langs["bs"].supported

    def test_sorted_by_name(self, tmp_path, google):
        names = [l.name for l in Translator(tmp_path).languages()]
        assert names == sorted(names)

    def test_installed_only(self, tmp_path, google):
        t = Translator(tmp_path)
        assert [l.code for l in t.languages(installed_only=True)] == ["en"]
        t.download("he")
        assert [l.code for l in t.languages(installed_only=True)] == ["en", "iw"]

    def test_installed_languages_needs_no_catalog(self, tmp_path, google):
        t = Translator(tmp_path)
        assert t.installed_languages() == ["en"]
        t.download("fr")
        google.offline = True
        shutil.rmtree(tmp_path / "_catalog")
        assert Translator(tmp_path).installed_languages() == ["en", "fr"]


# ---------- download / remove ----------
class TestDownload:
    def test_download_installs_both_directions(self, tmp_path, google):
        t = Translator(tmp_path)
        t.download("he")
        for pair in ("iw_en", "en_iw"):
            d = tmp_path / pair
            assert (d / "encoder_0.tflite").read_bytes() == google.file_content(pair, "encoder_0.tflite", "v1")
            meta = json.loads((d / "pack.json").read_text())
            assert set(meta) >= {"encoder_0.tflite", "src_spm_model.model"}
        assert not list(tmp_path.glob("*.partial"))
        assert langs_by_code(t)["iw"].installed

    def test_progress(self, tmp_path, google):
        calls = []
        t = Translator(tmp_path)
        t.download("iw", progress=lambda done, total: calls.append((done, total)))
        total = langs_by_code(t)["iw"].download_size
        assert calls and all(c[1] == total for c in calls)
        assert [c[0] for c in calls] == sorted(c[0] for c in calls)
        assert calls[-1] == (total, total)

    def test_already_installed_is_noop(self, tmp_path, google):
        t = Translator(tmp_path)
        t.download("iw")
        n = len(google.file_requests())
        t.download("iw")
        assert len(google.file_requests()) == n

    def test_force_redownloads(self, tmp_path, google):
        t = Translator(tmp_path)
        t.download("iw")
        n = len(google.file_requests())
        t.download("iw", force=True)
        assert len(google.file_requests()) == 2 * n

    def test_checksum_mismatch_installs_nothing(self, tmp_path, google):
        bad = next(u for u in google.blobs if u.endswith("iw_en/0/encoder_0.tflite.zlib"))
        google.corrupt.add(bad)
        t = Translator(tmp_path)
        with pytest.raises(TranslateError, match="checksum"):
            t.download("iw")
        assert not (tmp_path / "iw_en").exists()
        assert not list((tmp_path / "iw_en.partial").glob("*.part"))
        assert not langs_by_code(t)["iw"].installed

    def test_resume_skips_verified_files_only(self, tmp_path, google):
        t = Translator(tmp_path)
        t.languages()
        partial = tmp_path / "en_iw.partial"
        partial.mkdir()
        good = google.file_content("en_iw", "encoder_0.tflite", "v1")
        (partial / "encoder_0.tflite").write_bytes(good)
        (partial / "decoder_step_0.tflite").write_bytes(b"x" * len(good))  # same size, wrong content
        t.download("iw")
        fetched = google.file_requests()
        assert not any(u.endswith("en_iw/0/encoder_0.tflite.zlib") for u in fetched)
        assert any(u.endswith("en_iw/0/decoder_step_0.tflite.zlib") for u in fetched)
        step = google.file_content("en_iw", "decoder_step_0.tflite", "v1")
        assert (tmp_path / "en_iw" / "decoder_step_0.tflite").read_bytes() == step

    @pytest.mark.parametrize("lang, match", [("en", "built in"), ("xx", "unknown"), ("bs", "older model format")])
    def test_rejects(self, tmp_path, google, lang, match):
        with pytest.raises(TranslateError, match=match):
            Translator(tmp_path).download(lang)
        assert not google.file_requests()

    def test_update_available_and_update(self, tmp_path, google):
        t = Translator(tmp_path)
        t.download("iw")
        t.download("fr")
        assert not langs_by_code(t)["iw"].update_available

        google.publish(["iw", "fr"], version=(8, 6, 0), tag="v2")
        t.update_catalog()
        langs = langs_by_code(t)
        assert langs["iw"].update_available and langs["fr"].update_available

        t.download("iw")
        langs = langs_by_code(t)
        assert not langs["iw"].update_available and langs["fr"].update_available
        assert (tmp_path / "en_iw" / "encoder_0.tflite").read_bytes() == google.file_content(
            "en_iw", "encoder_0.tflite", "v2")

    def test_pack_installed_by_hand_gets_metadata(self, tmp_path, google):
        """Packs copied in without pack.json (e.g. pulled from a phone) are recognised."""
        src = tmp_path / "a"
        Translator(src).download("iw")
        dst = tmp_path / "b"
        dst.mkdir()
        for pair in ("iw_en", "en_iw"):
            shutil.copytree(src / pair, dst / pair, ignore=shutil.ignore_patterns("pack.json"))
        shutil.copytree(src / "_catalog", dst / "_catalog")
        lang = langs_by_code(Translator(dst))["iw"]
        assert lang.installed and not lang.update_available
        assert (dst / "iw_en" / "pack.json").exists()


class TestRemove:
    def test_remove(self, tmp_path, google):
        t = Translator(tmp_path)
        t.download("iw")
        (tmp_path / "fr_en.partial").mkdir()
        t.remove("Hebrew")
        assert not (tmp_path / "iw_en").exists() and not (tmp_path / "en_iw").exists()
        assert t.installed_languages() == ["en"]

    def test_removes_partial_downloads(self, tmp_path, google):
        (tmp_path / "fr_en.partial").mkdir()
        Translator(tmp_path).remove("fr")
        assert not (tmp_path / "fr_en.partial").exists()

    def test_not_installed(self, tmp_path, google):
        with pytest.raises(TranslateError, match="not installed"):
            Translator(tmp_path).remove("iw")

    def test_english(self, tmp_path, google):
        with pytest.raises(TranslateError, match="built in"):
            Translator(tmp_path).remove("en")


# ---------- translation (fake engine) ----------
@pytest.fixture
def installed(tmp_path, google, fake_engine):
    t = Translator(tmp_path)
    t.download("iw")
    t.download("fr")
    return t


class TestTranslate:
    def test_direct(self, installed):
        assert installed.translate("hello", target="he", source="en") == Translation(
            "[en_iw]hello", "en", "iw", ("en", "iw"))

    def test_pivot_through_english(self, installed):
        r = installed.translate("שלום", target="fr")
        assert r.text == "[en_fr][iw_en]שלום"
        assert r.via == ("iw", "en", "fr")

    def test_same_language(self, installed):
        assert installed.translate("hi", target="en", source="en") == Translation("hi", "en", "en", ("en",))

    def test_missing_language(self, installed):
        with pytest.raises(TranslateError, match="download: ar"):
            installed.translate("hello", target="ar", source="en")

    def test_missing_both(self, installed):
        with pytest.raises(TranslateError, match="download: ar, de"):
            installed.translate("x", target="de", source="ar")

    def test_models_are_cached(self, installed, fake_engine):
        installed.translate("a", target="iw", source="en")
        installed.translate("b", target="iw", source="en")
        assert fake_engine.loaded == ["en_iw"]

    def test_redownload_reloads_model(self, installed, fake_engine):
        installed.translate("a", target="iw", source="en")
        installed.download("iw", force=True)
        installed.translate("b", target="iw", source="en")
        assert fake_engine.loaded == ["en_iw", "en_iw"]

    def test_remove_then_translate(self, installed):
        installed.remove("iw")
        with pytest.raises(TranslateError):
            installed.translate("hello", target="iw", source="en")

    def test_route(self, installed):
        assert installed.route("iw", "en") == [("iw", "en")]
        assert installed.route("iw", "fr") == [("iw", "en"), ("en", "fr")]
        assert installed.route("fr", "fr") == []


class TestDetect:
    @pytest.mark.parametrize("text, lang", [
        ("hello world", "en"),
        ("שלום עולם", "iw"),
        ("ça va très bien", "fr"),
        ("123 !?", "en"),
        ("", "en"),
    ])
    def test_detect(self, installed, text, lang):
        assert installed.detect(text) == lang

    def test_only_installed_languages(self, tmp_path, google):
        assert Translator(tmp_path).detect("שלום") == "en"
