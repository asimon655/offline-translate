import pytest

from fakes import FakeGoogle, field, msg, varint
from offline_translate import catalog as cat


class TestParsePb:
    @pytest.mark.parametrize("n", [0, 1, 127, 128, 300, 2**32, 2**63])
    def test_varint_roundtrip(self, n):
        assert cat.parse_pb(field(1, n)) == {1: [n]}

    def test_strings_repeated_and_nested(self):
        data = msg((1, "a"), (1, "b"), (2, msg((3, 7))))
        parsed = cat.parse_pb(data)
        assert parsed[1] == [b"a", b"b"]
        assert cat.parse_pb(parsed[2][0]) == {3: [7]}

    def test_float32(self):
        assert cat.parse_pb(field(4, 3.0)) == {4: [3.0]}

    def test_float64(self):
        data = varint(5 << 3 | 1) + (1.5).hex().encode()[:0] + __import__("struct").pack("<d", 1.5)
        assert cat.parse_pb(data) == {5: [1.5]}

    def test_empty(self):
        assert cat.parse_pb(b"") == {}

    def test_unsupported_wire_type(self):
        with pytest.raises(ValueError):
            cat.parse_pb(varint(1 << 3 | 3))


class TestManifest:
    def test_keeps_only_premium_profiles_sorted(self):
        g = FakeGoogle(version=(8, 6, 0))
        refs = cat.parse_manifest(g.blobs[cat.MANIFEST_URL.format(channel="prod")])
        assert [r.version for r in refs] == [(8, 0, 0), (8, 6, 0)]
        assert refs[-1].url == g.profile_url
        assert refs[-1].version_str == "8.6.0"
        assert refs[-1].sha1 == __import__("hashlib").sha1(g.profile).hexdigest()

    def test_empty_manifest(self):
        assert cat.parse_manifest(msg((1, 9))) == []


class TestProfile:
    @pytest.fixture
    def packs(self):
        return cat.parse_profile(FakeGoogle(langs=("iw", "fr"), unsupported=("bs",)).profile)

    def test_languages(self, packs):
        assert set(packs) == {"iw", "fr", "bs"}

    def test_both_directions_and_no_dictionary(self, packs):
        assert sorted(p.name for p in packs["iw"].packages) == ["en_iw", "iw_en"]
        pkg = next(p for p in packs["iw"].packages if p.name == "iw_en")
        assert (pkg.source, pkg.target) == ("iw", "en")

    def test_files(self, packs):
        pkg = packs["iw"].packages[0]
        names = {f.name for f in pkg.files}
        assert cat.NMT_FILES <= names
        f = pkg.files[0]
        assert f.url.startswith(cat.BASE) and f.url.endswith(".zlib")
        assert len(f.sha1) == 40 and f.size > 0 and f.download_size > 0

    def test_sizes_are_sums(self, packs):
        pack = packs["fr"]
        assert pack.size == sum(f.size for p in pack.packages for f in p.files)
        assert pack.download_size == sum(f.download_size for p in pack.packages for f in p.files)

    def test_old_wordpiece_packs_are_marked_unsupported(self, packs):
        assert packs["iw"].supported and packs["fr"].supported
        assert not packs["bs"].supported
