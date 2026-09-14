#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A source-collection script for writing a patch that threads `initial_layout`
through `compile_for_hardware` (2026-09-14, addendum-15 section 4).

## What this does

1. Collects the signature and source of `psf_compile.compile_for_hardware`.
2. Collects the import statements at the top of the `psf_compile` module
   (to see how `transpile` is brought in).
3. Extracts every `transpile(` call site in the module, with line numbers.
4. Writes it out to a dated file, **after automatically redacting personal
   information**.

## Automatic redaction of personal information (important)

Per `publication-policy.md` section 4, the following are redacted before
writing:

- the account name running the script (collected at runtime from
  `getpass.getuser()` and the `~` home directory's actual value. **This
  script never writes the name anywhere in itself**)
- Windows absolute paths of the form `C:\\...`, and `/home/...` / `/Users/...`
- the hostname

As a check after redaction, the text is re-scanned for whether any word that
was supposed to be redacted still remains, and if so, **the file is not
written and the script aborts**.

## Usage

    python dump_psf_compile_source.py

`psf_compile_source_YYYY-MM-DD.txt` is created in the same folder. Please
look over its contents by eye before sending it (don't rely on the automatic
redaction alone).
"""
from __future__ import annotations

import getpass
import inspect
import os
import platform
import re
import socket
import sys
import time


def build_scrubber():
    """Returns a function that redacts words collected at runtime, along
    with the list of words used for the check. **Built this way so that
    the words to be kept secret are never written into this file.**"""
    secrets = []

    def add(v):
        if v and isinstance(v, str) and len(v) >= 3:
            secrets.append(v)

    try:
        add(getpass.getuser())
    except Exception:
        pass
    home = os.path.expanduser("~")
    add(os.path.basename(home.rstrip("\\/")))
    try:
        add(socket.gethostname())
    except Exception:
        pass

    # Replace longest-first (so a shorter word matching first doesn't leave
    # part of a longer one behind).
    secrets = sorted(set(secrets), key=len, reverse=True)

    path_patterns = [
        (re.compile(r"[A-Za-z]:\\[^\s\"'<>|]*"), "<PATH>"),
        (re.compile(r"/(?:home|Users)/[^\s\"'<>|]*"), "<PATH>"),
    ]

    def scrub(text):
        for pat, rep in path_patterns:
            text = pat.sub(rep, text)
        for s in secrets:
            text = text.replace(s, "<USER>")
        return text

    return scrub, secrets


def section(title):
    return "\n" + "=" * 78 + f"\n{title}\n" + "=" * 78 + "\n"


def main():
    scrub, secrets = build_scrubber()

    try:
        import psf_compile
        from psf_compile import compile_for_hardware
    except Exception as e:  # noqa: BLE001
        print(f"Could not import psf_compile: {type(e).__name__}: {e}")
        print("Please run this from the psf_zero_test folder (where psf_compile.py lives).")
        return 1

    out = []
    out.append(section("Environment (safe-to-share fields only)"))
    env = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "CPU": platform.processor(),
        "CPU_count": os.cpu_count(),
    }
    for name in ("qiskit", "rustworkx", "networkx", "numpy"):
        try:
            env[name] = __import__(name).__version__
        except Exception:
            env[name] = "absent"
    out.append(repr(env) + "\n")
    out.append(f"psf_compile's filename (path redacted): "
               f"{os.path.basename(getattr(psf_compile, '__file__', '?'))}\n")

    out.append(section("1. compile_for_hardware's signature"))
    try:
        out.append(str(inspect.signature(compile_for_hardware)) + "\n")
    except Exception as e:  # noqa: BLE001
        out.append(f"(could not obtain: {e})\n")

    out.append(section("2. compile_for_hardware's source"))
    try:
        out.append(inspect.getsource(compile_for_hardware))
    except Exception as e:  # noqa: BLE001
        out.append(f"(could not obtain: {e})\n")

    out.append(section("3. the module's first 60 lines (to see how imports are brought in)"))
    try:
        src = inspect.getsource(psf_compile)
    except Exception as e:  # noqa: BLE001
        src = ""
        out.append(f"(could not obtain the module source: {e})\n")
    if src:
        lines = src.splitlines()
        out.append("\n".join(f"{i+1:>4}: {ln}" for i, ln in enumerate(lines[:60])) + "\n")

        out.append(section("4. transpile( call sites in the module (5 lines of context each)"))
        hits = [i for i, ln in enumerate(lines) if "transpile(" in ln]
        if not hits:
            out.append("(no transpile( calls found)\n")
        for i in hits:
            lo, hi = max(0, i - 5), min(len(lines), i + 6)
            out.append(f"--- around line {i+1} ---\n")
            out.append("\n".join(f"{j+1:>4}: {lines[j]}" for j in range(lo, hi)) + "\n\n")

        out.append(section("5. Uses of generate_preset_pass_manager / PassManager"))
        hits2 = [i for i, ln in enumerate(lines)
                 if "generate_preset_pass_manager" in ln or "PassManager(" in ln]
        if not hits2:
            out.append("(none found)\n")
        for i in hits2:
            lo, hi = max(0, i - 3), min(len(lines), i + 4)
            out.append("\n".join(f"{j+1:>4}: {lines[j]}" for j in range(lo, hi)) + "\n\n")

    text = scrub("".join(out))

    # ------------------------------------------------ check: nothing leaked
    leaked = [s for s in secrets if s and s in text]
    generic = re.findall(r"[A-Za-z]:\\[^\s\"'<>|]*", text)
    if leaked or generic:
        print("**Redaction failed, so the file was not written.**")
        if leaked:
            print(f"  {len(leaked)} word(s) that should have been redacted remain "
                  f"(the words themselves are not shown here).")
        if generic:
            print(f"  {len(generic)} string(s) that look like absolute paths remain.")
        print("  Please let me know about this, and I will prepare a different collection method.")
        return 2

    path = f"psf_compile_source_{time.strftime('%Y-%m-%d')}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print("\n" + "=" * 78)
    print(f"The content above has been written to {path}.")
    print("The account name, absolute paths, and hostname have been redacted "
          "automatically (and the check above passed), but please look it "
          "over once before sending it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
