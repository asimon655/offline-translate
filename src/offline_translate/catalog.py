"""Google Translate offline package catalog.

How the Android app finds packs (reverse engineered):
  1. Manifest  https://dl.google.com/translate/offline/manifest/gtr/<channel>/manifest.pb
     lists every package-profile version (+ sha1/size). The channel comes from a server flag ("prod").
  2. Profile   https://dl.google.com/translate/offline/package_profile/<maj>/<min>/<patch>/translate_premium.pb.zlib
     lists every language pack, its packages (nmt models, dictionary, translit) and their files.
  3. Files     https://dl.google.com/translate/offline/nmt/<ver>/premium/<src>_<tgt>/0/<file>.zlib
"""
import struct
import urllib.request
import zlib
from dataclasses import dataclass, field

BASE = "https://dl.google.com/translate/offline"
MANIFEST_URL = BASE + "/manifest/gtr/{channel}/manifest.pb"
USER_AGENT = "Mozilla/5.0"

PKG_NMT = 3
# Files our inference engine knows how to run; packages with other layouts are skipped.
NMT_FILES = {
    "encoder_0.tflite", "decoder_init_0.tflite", "decoder_step_0.tflite",
    "src_spm_model.model", "tgt_spm_model.model",
}


# --- minimal protobuf reader (Google's schema isn't public) ---
def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def parse_pb(b):
    """Parse a protobuf message into {field: [values]}; length-delimited values stay bytes."""
    out, i = {}, 0
    while i < len(b):
        key, i = _varint(b, i)
        f, t = key >> 3, key & 7
        if t == 0:
            v, i = _varint(b, i)
        elif t == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif t == 5:
            v, i = struct.unpack("<f", b[i:i + 4])[0], i + 4
        elif t == 1:
            v, i = struct.unpack("<d", b[i:i + 8])[0], i + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {t}")
        out.setdefault(f, []).append(v)
    return out


def _str(m, f):
    return m[f][0].decode()


def _hash(m):
    """{1: {1: algo, 2: hex}, 2: size} -> (hex, size)."""
    return _str(parse_pb(m[1][0]), 2), m[2][0]


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def open_url(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


@dataclass(frozen=True)
class ProfileRef:
    version: tuple
    url: str
    sha1: str  # of the decompressed profile

    @property
    def version_str(self):
        return ".".join(map(str, self.version))


def parse_manifest(data):
    """Returns the premium (NMT) package profiles listed in a manifest, oldest first."""
    refs = []
    groups = parse_pb(data).get(2, [])
    for raw in (r for g in groups for r in parse_pb(g).get(2, [])):
        m = parse_pb(raw)
        url = _str(m, 2)
        if "translate_premium" not in url:
            continue
        v = parse_pb(m[1][0])
        version = tuple(v.get(k, [0])[0] for k in (1, 2, 3))
        refs.append(ProfileRef(version, url, _hash(parse_pb(m[3][0]))[0]))
    return sorted(refs, key=lambda r: r.version)


@dataclass(frozen=True)
class PackFile:
    name: str       # file name on disk, e.g. encoder_0.tflite
    url: str
    sha1: str       # of the decompressed file
    size: int       # decompressed bytes
    download_size: int  # zlib-compressed bytes


@dataclass(frozen=True)
class Package:
    name: str       # e.g. "en_iw" (direction src_tgt)
    files: tuple

    @property
    def source(self):
        return self.name.split("_", 1)[0]

    @property
    def target(self):
        return self.name.split("_", 1)[1]


@dataclass
class Pack:
    """A downloadable language: both directions to/from English."""
    lang: str
    packages: list = field(default_factory=list)
    # False for older packs (nmt/8/1, WordPiece tokenizer) that the engine can't run yet.
    supported: bool = True

    @property
    def download_size(self):
        return sum(f.download_size for p in self.packages for f in p.files)

    @property
    def size(self):
        return sum(f.size for p in self.packages for f in p.files)


def parse_profile(data):
    """Returns {lang_code: Pack} for every language with an NMT model (see Pack.supported)."""
    top = parse_pb(data)
    files = {}
    for raw in top.get(3, []):
        m = parse_pb(raw)
        sha1, size = _hash(parse_pb(m[3][0]))
        _, zsize = _hash(parse_pb(m[4][0]))
        files[_str(m, 1)] = PackFile(_str(m, 5), _str(m, 2), sha1, size, zsize)

    packages = {}
    for raw in top.get(2, []):
        m = parse_pb(raw)
        if m[2][0] != PKG_NMT:
            continue
        pfiles = tuple(files[f.decode()] for f in m.get(4, []))
        packages[_str(m, 1)] = Package(_str(m, 1).removesuffix("_premium_nmt"), pfiles)

    packs = {}
    for raw in top.get(1, []):
        m = parse_pb(raw)
        langs = [_str(parse_pb(x), 2) for x in parse_pb(m[3][0])[1]]
        lang = next((l for l in langs if l != "en"), None)
        pkgs = [packages[p.decode()] for p in m.get(2, []) if p.decode() in packages]
        if lang and pkgs:
            ok = all(NMT_FILES <= {f.name for f in p.files} for p in pkgs)
            packs[lang] = Pack(lang, pkgs, ok)
    return packs
