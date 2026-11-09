"""Offline translation with Google Translate's on-device language packs.

    from offline_translate import Translator
    t = Translator("~/models")
    t.download("he")
    t.translate("Hello", target="he").text
"""
from .languages import NAMES, normalize
from .translator import DownloadCancelled, Language, NotInstalledError, TranslateError, Translation, Translator

__version__ = "0.2.0"
__all__ = ["Translator", "Translation", "Language", "TranslateError", "NotInstalledError", "DownloadCancelled", "NAMES", "normalize", "__version__"]
