#!/usr/bin/env python3
"""Command-line front end for the offline_translate library.

  translate "Hello world"              translate (source auto-detected; English -> Hebrew by default)
  translate -s he -t ar "שלום"         explicit languages (non-English pairs go via English)
  echo "שלום" | translate              read stdin
  translate -i                         interactive
  translate list [--installed]         languages in Google's catalog
  translate get fr de                  download language packs
  translate remove fr                  delete language packs
  translate update                     refresh catalog and update installed packs
"""
import argparse
import os
import sys
from pathlib import Path

from offline_translate import NAMES, TranslateError, Translator, normalize

DEFAULT_DIR = Path(os.environ.get("OFFLINE_TRANSLATE_MODELS", Path(__file__).resolve().parent / "models"))
COMMANDS = {"list", "get", "remove", "update"}


def mb(n):
    return f"{n / 1e6:.0f} MB" if n else "-"


def progress_bar(label):
    def cb(done, total):
        pct = done * 100 // total if total else 100
        print(f"\r  {label}: {pct:3d}%  {mb(done)} / {mb(total)}", end="", file=sys.stderr, flush=True)
    return cb


def confirm(msg, yes):
    return yes or input(f"{msg} [y/N] ").strip().lower() == "y"


def cmd_list(t, a):
    langs = t.languages(installed_only=a.installed)
    for l in langs:
        mark = "*" if l.installed else " "
        note = (" (built in)" if l.builtin else " (update available)" if l.update_available
                else " (not supported yet)" if not l.supported else "")
        print(f"{mark} {l.code:8} {l.name:24} {mb(l.download_size):>7}{note}")
    if not a.installed:
        print(f"\n{len(langs)} languages, * = installed, catalog {t.catalog_version()}")


def cmd_get(t, a):
    langs = {l.code: l for l in t.languages()}
    todo = []
    for raw in a.langs:
        code = normalize(raw)
        l = langs.get(code)
        if not l:
            sys.exit(f"unknown language '{raw}' (see: translate list)")
        if l.builtin or (l.installed and not l.update_available):
            print(f"{l.name} is already installed")
        elif not l.supported:
            print(f"{l.name} uses an older model format that isn't supported yet")
        else:
            todo.append(l)
    if not todo:
        return
    total = sum(l.download_size for l in todo)
    if not confirm(f"Download {', '.join(l.name for l in todo)} ({mb(total)})?", a.yes):
        return
    for l in todo:
        t.download(l.code, progress=progress_bar(l.name))
        print(file=sys.stderr)


def cmd_remove(t, a):
    for raw in a.langs:
        t.remove(raw)
        print(f"removed {NAMES.get(normalize(raw), raw)}")


def cmd_update(t, a):
    print(f"catalog {t.update_catalog()}")
    stale = [l for l in t.languages(installed_only=True) if l.update_available]
    if not stale:
        print("all installed languages are up to date")
        return
    if confirm(f"Update {', '.join(l.name for l in stale)}?", a.yes):
        for l in stale:
            t.download(l.code, progress=progress_bar(l.name))
            print(file=sys.stderr)


def cmd_translate(t, a):
    def default_target(src):
        if src != "en":
            return "en"
        others = [c for c in t.installed_languages() if c != "en"]
        if not others:
            sys.exit("no language packs installed; see: translate list, translate get <lang>")
        if "iw" in others or len(others) != 1:
            return "iw" if "iw" in others else sys.exit("specify a target language with -t")
        return others[0]

    def run(text):
        src = normalize(a.source) if a.source else t.detect(text)
        return t.translate(text, target=a.target or default_target(src), source=src).text

    if a.interactive:
        try:
            while True:
                line = input("> ").strip()
                if line:
                    try:
                        print(run(line))
                    except TranslateError as e:
                        print(f"error: {e}", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print()
        return
    text = " ".join(a.text) if a.text else sys.stdin.read()
    if text.strip():
        print(run(text.strip()))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-d", "--models-dir", default=DEFAULT_DIR, help=f"models folder (default {DEFAULT_DIR})")

    if argv and argv[0] in COMMANDS:
        p = argparse.ArgumentParser(prog="translate")
        sub = p.add_subparsers(dest="cmd", required=True)
        l = sub.add_parser("list", parents=[common], help="list languages")
        l.add_argument("--installed", action="store_true")
        for name in ("get", "remove"):
            s = sub.add_parser(name, parents=[common])
            s.add_argument("langs", nargs="+")
            s.add_argument("-y", "--yes", action="store_true")
        sub.add_parser("update", parents=[common]).add_argument("-y", "--yes", action="store_true")
    else:
        p = argparse.ArgumentParser(prog="translate", parents=[common], description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("text", nargs="*", help="text to translate (default: stdin)")
        p.add_argument("-s", "--source", help="source language; auto-detected if omitted")
        p.add_argument("-t", "--target", help="target language")
        p.add_argument("-b", "--beam", type=int, default=4, help="beam size (1 = greedy, faster)")
        p.add_argument("-i", "--interactive", action="store_true")
        p.set_defaults(cmd="translate")
    a = p.parse_args(argv)

    t = Translator(a.models_dir, beam=getattr(a, "beam", 4))
    try:
        {"list": cmd_list, "get": cmd_get, "remove": cmd_remove, "update": cmd_update,
         "translate": cmd_translate}[a.cmd](t, a)
    except TranslateError as e:
        sys.exit(f"error: {e}")
    except OSError as e:  # network failures when the catalog is needed
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
