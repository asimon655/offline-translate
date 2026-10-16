"""Public API: manage language packs in a folder and translate between installed languages."""
import hashlib
import json
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

from . import catalog as cat
from .languages import NAMES, normalize

PACK_META = "pack.json"


class TranslateError(Exception):
    pass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    installed: bool
    download_size: int     # bytes to download (0 for English: it needs no pack of its own)
    size: int              # bytes on disk once installed
    update_available: bool = False
    builtin: bool = False  # English: always "installed", every pack is X<->English
    supported: bool = True  # False: pack uses an older model format this engine can't run yet


@dataclass(frozen=True)
class Translation:
    text: str
    source: str
    target: str
    via: tuple  # language hops, e.g. ("iw", "en", "ar") when pivoting through English


class Translator:
    """
    models_dir: folder holding the catalog cache and one sub-folder per model direction (e.g. en_iw/).
    channel:    manifest channel the Android app reads ("prod").
    """

    def __init__(self, models_dir, channel="prod", beam=4):
        self.dir = Path(models_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.channel = channel
        self.beam = beam
        self._packs = None
        self._models = {}

    # ---------- catalog ----------
    @property
    def _catalog_dir(self):
        return self.dir / "_catalog"

    def update_catalog(self):
        """Fetch the latest manifest + package profile from Google. Returns the profile version."""
        refs = cat.parse_manifest(cat.fetch(cat.MANIFEST_URL.format(channel=self.channel)))
        if not refs:
            raise TranslateError("manifest lists no translation profiles")
        ref = refs[-1]
        data = zlib.decompress(cat.fetch(ref.url))
        if hashlib.sha1(data).hexdigest() != ref.sha1:
            raise TranslateError(f"checksum mismatch for {ref.url}")
        self._catalog_dir.mkdir(exist_ok=True)
        (self._catalog_dir / "profile.pb").write_bytes(data)
        (self._catalog_dir / "profile.json").write_text(json.dumps({"version": ref.version_str, "url": ref.url}))
        self._packs = cat.parse_profile(data)
        return ref.version_str

    def catalog_version(self):
        meta = self._catalog_dir / "profile.json"
        return json.loads(meta.read_text())["version"] if meta.exists() else None

    def _catalog(self):
        if self._packs is None:
            path = self._catalog_dir / "profile.pb"
            if path.exists():
                self._packs = cat.parse_profile(path.read_bytes())
            else:
                self.update_catalog()
        return self._packs

    # ---------- languages ----------
    def _pair_dir(self, pair):
        return self.dir / pair

    def _installed_pairs(self):
        return {tuple(d.name.split("_", 1)) for d in self.dir.iterdir()
                if d.is_dir() and (d / "encoder_0.tflite").exists()}

    def _pack_meta(self, pair):
        """sha1 of each installed file; computed once for packs installed by other means."""
        d = self._pair_dir(pair)
        path = d / PACK_META
        if path.exists():
            return json.loads(path.read_text())
        meta = {f.name: hashlib.sha1(f.read_bytes()).hexdigest()
                for f in sorted(d.iterdir()) if f.is_file() and f.name != PACK_META}
        path.write_text(json.dumps(meta, indent=1))
        return meta

    def _package_state(self, pkg):
        """(installed, update_available) for one direction."""
        if not (self._pair_dir(pkg.name) / "encoder_0.tflite").exists():
            return False, False
        meta = self._pack_meta(pkg.name)
        return True, any(meta.get(f.name) != f.sha1 for f in pkg.files)

    def languages(self, installed_only=False):
        """All languages in the catalog (plus English), sorted by name."""
        out = [Language("en", NAMES["en"], True, 0, 0, builtin=True)]
        for code, pack in self._catalog().items():
            states = [self._package_state(p) for p in pack.packages]
            installed = all(s[0] for s in states)
            out.append(Language(code, NAMES.get(code, code), installed, pack.download_size, pack.size,
                                update_available=installed and any(s[1] for s in states),
                                supported=pack.supported))
        out.sort(key=lambda l: l.name)
        return [l for l in out if l.installed] if installed_only else out

    def installed_languages(self):
        """Languages usable for translation right now (works offline, without the catalog)."""
        codes = {c for pair in self._installed_pairs() for c in pair}
        return sorted(codes or {"en"}, key=lambda c: NAMES.get(c, c))

    def _resolve(self, lang):
        code = normalize(lang)
        if code == "en":
            raise TranslateError("English is built in; every language pack includes it")
        if code not in self._catalog():
            raise TranslateError(f"unknown language '{lang}'")
        return code

    def download(self, lang, progress=None, force=False):
        """
        Download (or update) a language pack. Already up-to-date directions are skipped unless force.
        progress(done_bytes, total_bytes) is called while downloading (sizes are compressed bytes).
        """
        pack = self._catalog()[self._resolve(lang)]
        if not pack.supported:
            raise TranslateError(f"{NAMES.get(pack.lang, pack.lang)} uses an older model format that isn't supported yet")
        todo = [p for p in pack.packages if force or self._package_state(p) != (True, False)]
        total = sum(f.download_size for p in todo for f in p.files)
        done = 0
        for pkg in todo:
            tmp = self.dir / f"{pkg.name}.partial"
            tmp.mkdir(exist_ok=True)
            meta = {}
            for f in pkg.files:
                dest = tmp / f.name
                if dest.exists() and hashlib.sha1(dest.read_bytes()).hexdigest() == f.sha1:  # resume
                    meta[f.name] = f.sha1
                else:
                    meta[f.name] = self._download_file(f, dest, lambda n: progress and progress(done + n, total))
                done += f.download_size
                if progress:
                    progress(done, total)
            (tmp / PACK_META).write_text(json.dumps(meta, indent=1))
            final = self._pair_dir(pkg.name)
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            self._models.pop(tuple(pkg.name.split("_", 1)), None)

    @staticmethod
    def _download_file(f, dest, on_bytes):
        sha, dec, got = hashlib.sha1(), zlib.decompressobj(), 0
        part = dest.with_suffix(dest.suffix + ".part")
        with cat.open_url(f.url) as r, open(part, "wb") as out:
            while chunk := r.read(1 << 16):
                got += len(chunk)
                on_bytes(got)
                data = dec.decompress(chunk)
                sha.update(data)
                out.write(data)
            tail = dec.flush()
            sha.update(tail)
            out.write(tail)
        if sha.hexdigest() != f.sha1:
            part.unlink()
            raise TranslateError(f"checksum mismatch for {f.url}")
        part.rename(dest)
        return f.sha1

    def remove(self, lang):
        code = normalize(lang)
        if code == "en":
            raise TranslateError("English is built in and can't be removed")
        removed = False
        for pair in [(code, "en"), ("en", code)]:
            d = self._pair_dir("_".join(pair))
            for p in (d, d.with_name(d.name + ".partial")):
                if p.exists():
                    shutil.rmtree(p)
                    removed = True
            self._models.pop(pair, None)
        if not removed:
            raise TranslateError(f"'{lang}' is not installed")

    # ---------- translation ----------
    def detect(self, text):
        """Guess the source among installed languages by vocabulary coverage of non-ASCII letters."""
        letters = {c for c in text.lower() if c.isalpha() and not c.isascii()}
        best, best_hits = "en", 0
        if letters:
            for src, tgt in sorted(self._installed_pairs()):
                if src == "en":
                    continue
                vocab = (self._pair_dir(f"{src}_{tgt}") / "src_spm_model.vocab").read_text("utf-8")
                hits = len(letters & set(vocab))
                if hits > best_hits:
                    best, best_hits = src, hits
        return best

    def route(self, source, target):
        """Model hops for source->target: direct, or via English like the Android app."""
        pairs = self._installed_pairs()
        if source == target:
            return []
        if (source, target) in pairs:
            return [(source, target)]
        if (source, "en") in pairs and ("en", target) in pairs:
            return [(source, "en"), ("en", target)]
        missing = [l for l in (source, target) if l != "en" and not {(l, "en"), ("en", l)} <= pairs]
        raise TranslateError(f"no model for {source} -> {target}; download: {', '.join(missing)}")

    def translate(self, text, target, source=None):
        source = normalize(source) if source else self.detect(text)
        target = normalize(target)
        hops = self.route(source, target)
        for pair in hops:
            if pair not in self._models:
                from .engine import NmtModel  # deferred: TFLite import is the slow part
                self._models[pair] = NmtModel(self._pair_dir("_".join(pair)))
            text = self._models[pair].translate(text, beam=self.beam)
        via = (source,) + tuple(t for _, t in hops) if hops else (source,)
        return Translation(text, source, target, via)
