"""Test doubles: a protobuf writer, a fake dl.google.com, and a fake NMT model."""
import hashlib
import io
import struct
import urllib.error
import zlib

from offline_translate import catalog as cat


# ---------- protobuf writer (mirror of catalog.parse_pb) ----------
def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def field(num, value):
    if isinstance(value, bool) or isinstance(value, int):
        return varint(num << 3) + varint(int(value))
    if isinstance(value, float):
        return varint(num << 3 | 5) + struct.pack("<f", value)
    if isinstance(value, str):
        value = value.encode()
    return varint(num << 3 | 2) + varint(len(value)) + value


def msg(*pairs):
    """msg((1, "a"), (2, 5), (3, msg(...)))"""
    return b"".join(field(f, v) for f, v in pairs)


def sha1(b):
    return hashlib.sha1(b).hexdigest()


def hashmsg(data):
    return msg((1, msg((1, 1), (2, sha1(data)))), (2, len(data)))


# ---------- fake Google ----------
NMT_FILES = ["decoder_init_0.tflite", "decoder_step_0.tflite", "encoder_0.tflite", "processor_spec.pb",
             "src_spm_model.model", "src_spm_model.vocab", "tgt_spm_model.model", "tgt_spm_model.vocab",
             "translation_model.pb"]
WPM_FILES = ["decoder_init_0.tflite", "decoder_step_0.tflite", "encoder_0.tflite", "processor_spec.pb",
             "src_wpm_model.model", "src_wpm_model.vocab", "tgt_wpm_model.model", "tgt_wpm_model.vocab",
             "translation_model.pb"]
# Letters each fake language's source vocabulary covers (used by Translator.detect).
SCRIPTS = {"en": "abcdefghijklmnopqrstuvwxyz", "iw": "אבגדהוזחטיכלמנסעפצקרשתםןץףך", "fr": "éèàçùâêôû",
           "ar": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي", "bs": "čćžšđ"}


class FakeGoogle:
    """Serves a manifest, a package profile and pack files the way dl.google.com does."""

    def __init__(self, langs=("iw", "fr"), unsupported=("bs",), version=(8, 5, 0), tag="v1", channel="prod"):
        self.blobs = {}
        self.requests = []
        self.offline = False
        self.corrupt = set()  # urls served with wrong content
        self.channel = channel
        self.publish(langs, unsupported, version, tag)

    def file_content(self, pair, name, tag):
        if name == "src_spm_model.vocab":
            src = pair.split("_")[0]
            return "\n".join(f"{c}\t0" for c in SCRIPTS.get(src, "")).encode()
        return f"{pair}/{name}/{tag}/".encode() * 50

    def publish(self, langs, unsupported=(), version=(8, 5, 0), tag="v1"):
        """(Re)publish a catalog. Bumping version/tag simulates Google releasing new models."""
        self.version, self.tag = version, tag
        files, packages, packs = [], [], []
        for lang in list(langs) + list(unsupported):
            names = WPM_FILES if lang in unsupported else NMT_FILES
            pkg_names = []
            for pair in (f"{lang}_en", f"en_{lang}"):
                keys = []
                for name in names:
                    key = f"nmt_{pair}_{name}.zlib"
                    url = f"{cat.BASE}/nmt/8/2/premium/{pair}/0/{name}.zlib"
                    data = self.file_content(pair, name, tag)
                    packed = zlib.compress(data)
                    self.blobs[url] = packed
                    files.append(msg((1, key), (2, url), (3, hashmsg(data)), (4, hashmsg(packed)), (5, name)))
                    keys.append(key)
                pkg = f"{pair}_premium_nmt"
                packages.append(msg((1, pkg), (2, cat.PKG_NMT), *[(4, k) for k in keys]))
                pkg_names.append(pkg)
            # a dictionary package that must be ignored
            packages.append(msg((1, f"{lang}_en_dictionary"), (2, 6)))
            pkg_names.append(f"{lang}_en_dictionary")
            langs_msg = msg((1, msg((1, 1), (2, lang))), (1, msg((1, 1), (2, "en"))), (2, 2))
            packs.append(msg((1, f"translate_{lang}_en"), *[(2, p) for p in pkg_names], (3, langs_msg)))
        profile = msg(*[(1, p) for p in packs], *[(2, p) for p in packages], *[(3, f) for f in files],
                      (4, msg((1, version[0]), (2, version[1]))))

        maj, mnr, patch = version
        self.profile_url = f"{cat.BASE}/package_profile/{maj}/{mnr}/{patch}/translate_premium.pb.zlib"
        self.blobs[self.profile_url] = zlib.compress(profile)
        old_url = f"{cat.BASE}/package_profile/8/0/0/translate_premium.pb.zlib"
        std_url = f"{cat.BASE}/package_profile/6/9/0/translate_standard.pb.zlib"
        entry = lambda v, url, data: msg((1, msg((1, v[0]), (2, v[1]), (3, v[2]))), (2, url), (3, hashmsg(data)))
        # Entries deliberately out of order; a "standard" (non-NMT) profile must be skipped.
        manifest = msg((1, 9), (2, msg((1, 1), (2, entry(version, self.profile_url, profile)),
                                       (2, entry((8, 0, 0), old_url, b"old")),
                                       (2, entry((6, 9, 0), std_url, b"std")))))
        self.blobs[cat.MANIFEST_URL.format(channel=self.channel)] = manifest
        self.profile = profile
        return profile

    def get(self, url):
        self.requests.append(url)
        if self.offline:
            raise urllib.error.URLError("network is unreachable")
        if url not in self.blobs:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if url in self.corrupt:
            return zlib.compress(b"corrupted")
        return self.blobs[url]

    def install(self, monkeypatch):
        monkeypatch.setattr(cat, "fetch", lambda url, timeout=60: self.get(url))
        monkeypatch.setattr(cat, "open_url", lambda url, timeout=60: io.BytesIO(self.get(url)))
        return self

    def file_requests(self):
        return [u for u in self.requests if "/nmt/" in u]


class FakeModel:
    """Stands in for engine.NmtModel: tags text with the direction it went through."""
    loaded = []

    def __init__(self, path):
        self.pair = path.name
        FakeModel.loaded.append(self.pair)

    def translate(self, text, beam=4):
        return f"[{self.pair}]{text}"
