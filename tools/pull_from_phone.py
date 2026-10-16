#!/usr/bin/env python3
"""Pull Google Translate offline NMT models from a rooted Android phone via adb.

Download the language packs in the Translate app first, then run:
  .venv/bin/python tools/pull_from_phone.py
Models land in models/<src>_<tgt>/ (e.g. models/en_iw, models/iw_en).
"""
import base64
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

PKG_DIR = "/data/data/com.google.android.apps.translate"
PREFS = "shared_prefs"
BLOBS = "files/translate_android/shared/datadownload/public"
OUT = Path(__file__).resolve().parent.parent / "models"


def b64(s):
    return base64.b64decode(s.replace("-", "+").replace("_", "/") + "=" * (-len(s) % 4))


def entries(xml):
    return re.findall(r'<string name="([^"]*)">([^<]*)</string>', xml)


def main():
    cmd = f"cd {PKG_DIR} && tar cf - {PREFS}/gms_icing_mdd_groupstranslate_android.xml " \
          f"{PREFS}/gms_icing_mdd_shared_filestranslate_android.xml {BLOBS}"
    data = subprocess.run(["adb", "shell", "su", "-c", f"'{cmd}'"], capture_output=True, check=True).stdout
    tar = tarfile.open(fileobj=io.BytesIO(data))
    read = lambda name: tar.extractfile(name).read()

    # sha1 -> on-device blob name
    sha2blob = {}
    for k, v in entries(read(f"{PREFS}/gms_icing_mdd_shared_filestranslate_android.xml").decode()):
        m = re.search(rb"datadownloadfile_\d+", b64(v))
        if m:
            sha2blob[k.split("|")[0]] = m.group().decode()

    # file groups list each file's download URL and sha1; URL path tells us pair + filename
    n = 0
    for _, v in entries(read(f"{PREFS}/gms_icing_mdd_groupstranslate_android.xml").decode()):
        for url, sha in re.findall(rb"(https://[\x21-\x7e]+?)\x20.{1,5}\*\x28([0-9a-f]{40})", b64(v), re.S):
            url = url.decode()
            m = re.search(r"/nmt/[\d/]+\w+/([a-z]+_[a-z]+)/\d+/([^/]+)\.zlib$", url)
            if not m:
                continue
            blob = sha2blob.get(sha.decode())
            if not blob:
                print(f"missing blob for {url}", file=sys.stderr)
                continue
            dest = OUT / m.group(1) / m.group(2)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(read(f"{BLOBS}/{blob}"))
            n += 1
            print(f"{dest.relative_to(OUT.parent)}  <-  {blob}")
    print(f"{n} files written")


if __name__ == "__main__":
    main()
