import pytest

from offline_translate import NAMES, normalize


@pytest.mark.parametrize("given, expected", [
    ("he", "iw"), ("HE", "iw"), ("iw", "iw"), ("Hebrew", "iw"), ("hebrew", "iw"),
    ("en", "en"), ("English", "en"),
    ("jv", "jw"), ("zh-TW", "zh-Hant"), ("zh-hant", "zh-Hant"), ("zh-Hant", "zh-Hant"),
    ("Chinese (Traditional)", "zh-Hant"), (" fr ", "fr"), ("fil", "tl"),
])
def test_normalize(given, expected):
    assert normalize(given) == expected


def test_unknown_passes_through():
    assert normalize("xx") == "xx"


def test_names_cover_google_codes():
    assert NAMES["iw"] == "Hebrew"
    assert NAMES["en"] == "English"
    assert "he" not in NAMES  # Google's catalog uses the legacy code
