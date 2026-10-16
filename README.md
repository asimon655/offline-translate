# offline-translate

Offline machine translation in Python using the same on-device neural translation models that the
Google Translate Android app downloads for offline use.

- ~100 languages, each pack translates to/from English (~50 MB per language)
- Fully offline once a pack is downloaded; no API key, no account
- Language pairs without English (e.g. Hebrew → French) pivot through English, like the app does
- Runs on CPU via [LiteRT](https://ai.google.dev/edge/litert) (TensorFlow Lite); ~0.3 s per sentence

```bash
pip install offline-translate
```

Supported platforms are those with `ai-edge-litert` wheels: macOS (Apple Silicon), Linux (x86_64,
aarch64) and Windows (x64), Python 3.10+.

## Usage

```python
from offline_translate import Translator

t = Translator("~/.local/share/offline-translate")   # where models are stored

t.download("he")                          # Hebrew <-> English, ~55 MB
t.translate("Good morning!", target="he").text
# 'בוקר טוב!'

t.translate("בוקר טוב", target="en")      # source is auto-detected among installed languages
# Translation(text='Good morning', source='iw', target='en', via=('iw', 'en'))
```

### Managing languages

```python
for lang in t.languages():                # full catalog (fetched once from Google, then cached)
    print(lang.code, lang.name, lang.installed, lang.download_size, lang.update_available)

t.installed_languages()                   # ['en', 'iw'] — works offline, no catalog needed
t.download("fr", progress=lambda done, total: print(f"{done}/{total}"))
t.remove("fr")

t.update_catalog()                        # check Google for a newer catalog
[l for l in t.languages() if l.update_available]   # then t.download(code) to update
```

Language codes follow Google's catalog (`iw` for Hebrew, `zh-Hant`, `jw`, …). Common aliases
(`he`, `zh-TW`, `jv`) and English names (`"Hebrew"`) are accepted everywhere.

English is built in: there is no separate English pack, since every pack contains both
directions to and from English.

### API

| | |
|---|---|
| `Translator(models_dir, channel="prod", beam=4)` | `models_dir` is created if missing. `beam=1` is faster, slightly worse. |
| `translate(text, target, source=None) -> Translation` | `Translation(text, source, target, via)` |
| `detect(text) -> str` | Guess among installed languages |
| `languages(installed_only=False) -> list[Language]` | `Language(code, name, installed, download_size, size, update_available, builtin, supported)` |
| `installed_languages() -> list[str]` | |
| `download(lang, progress=None, force=False)` | Verifies SHA-1 of every file; resumes interrupted downloads |
| `remove(lang)` | |
| `update_catalog() -> str` | Returns the catalog version |
| `TranslateError` | Raised for unknown/uninstalled languages, checksum failures, … |

Network errors surface as `OSError`.

## Command line

The repository also has a small CLI built on the library (not part of the PyPI package):

```bash
./translate get he                 # download Hebrew
./translate "Good morning"         # English -> Hebrew by default; source auto-detected
./translate -s he -t fr "בוקר טוב"  # any pair, via English
./translate list --installed
```

`tools/pull_from_phone.py` copies already-downloaded packs from a rooted Android phone over adb.

## How it works

The Android app reads a manifest at
`https://dl.google.com/translate/offline/manifest/gtr/prod/manifest.pb`, which lists versions of
a *package profile* — the catalog of language packs and their files. Each translation direction is
a TFLite encoder, an LSTM decoder run step by step with beam search, and SentencePiece tokenizers.

### Limitations

- 15 languages (am, az, bs, hy, kk, ky, ml, mn, ne, pa, ps, si, so, tg, uz) still ship an older
  model format with a WordPiece tokenizer that isn't supported yet; they're listed with
  `supported=False`.
- Quality matches the app's offline mode, which is below online Google Translate.

## Development

```bash
pip install -e ".[test]"
pytest                          # offline unit tests (fake Google server, fake model)
pytest --network                # + checks against the real Google catalog
pytest --models=~/models        # + end-to-end translation with installed Hebrew models
```

## Disclaimer

This project is not affiliated with or endorsed by Google. It contains no Google code or models;
language packs are downloaded from Google's public servers at runtime and are subject to Google's
terms.
