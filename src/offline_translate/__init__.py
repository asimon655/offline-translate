"""Offline translation with Google Translate's on-device language packs.

    from offline_translate import Translator
    t = Translator("~/models")
    t.download("he")
    t.translate("Hello", target="he").text
"""
from .languages import NAMES, normalize
from .translator import Language, TranslateError, Translation, Translator

__version__ = "0.1.0"
__all__ = ["Translator", "Translation", "Language", "TranslateError", "NAMES", "normalize", "__version__"]
