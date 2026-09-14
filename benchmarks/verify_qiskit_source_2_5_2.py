#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(6/6) Read the installed Qiskit's own source -- is what was read on `main`
actually true of 2.5.2?

## Why this is needed

The 2026-09-14 addendum to `spare-qubit-cliff.md` is based on
`builtin_plugins.py` / `vf2_layout.py` / `transpiler.rs` as read from GitHub's
`main`. **Every measurement in this project was taken on 2.5.2.** If the version
differs, the conclusions could too.

This is not a benchmark. It is a script that **mechanically extracts and records
the relevant fragments from the installed files** -- nothing more. It runs in
seconds, not minutes.

It turns this project's own rule -- "read the implementation before proposing a
mechanism", "always state a number with its version" -- into something that can
be run mechanically.

## What is checked

  1. Does `VF2Layout`'s import include `rustworkx` (if so, this predates #14860)
  2. `VF2Layout.run`'s seed branch (how `seed == -1` is handled)
  3. The `seed` and `call_limit` `DefaultLayoutPassManager` passes to `VF2Layout(...)`
  4. The `seed` passed to `VF2PostLayout(...)`
  5. The contents of `get_vf2_limits`
  6. Whether `qiskit.transpile()` goes through `generate_preset_pass_manager`
  7. Whether `psf_zero_core` has `geometric_decompose_checked`, and the actual
     extension file and its modification time

## Pre-registered predictions

(P1) 2.5.2's `vf2_layout.py` has no `rustworkx` import (2.5.2 postdates #14860's
     merge).
(P2) `DefaultLayoutPassManager` passes `seed=-1` at every level.
(P3) `call_limit` is a 2-tuple, with levels 1/2/3 at
     (50_000, 1_000) / (5_000_000, 10_000) / (30_000_000, 100_000).
(P4) `transpile()` is a thin wrapper around
     `generate_preset_pass_manager(...).run(...)`.

**If this fails**: state explicitly in the relevant addendum section "this differs
in 2.5.2", and keep the `main`-based reading as a statement about `main`.

## Usage

    python verify_qiskit_source_2_5_2.py
"""
from __future__ import annotations

import argparse
import datetime
import glob
import inspect
import os
import re

from vf2_probe_common import banner, default_out, environment, write_csv


def source_of(obj):
    try:
        return inspect.getsource(obj)
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def check(rows, name, ok, detail):
    mark = "OK " if ok is True else ("NG " if ok is False else "?? ")
    print(f"{mark} {name}")
    for line in str(detail).strip().splitlines():
        print("      " + line)
    rows.append(dict(Check=name, Result=str(ok), Detail=str(detail).strip()[:2000]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    banner("(6/6) Checking the installed Qiskit's source", [
        "P1 vf2_layout.py has no rustworkx import",
        "P2 DefaultLayoutPassManager passes seed=-1 at every level",
        "P3 call_limit is a 2-tuple: (50_000,1_000) / (5_000_000,10_000) / (30_000_000,100_000)",
        "P4 transpile() is a thin wrapper around generate_preset_pass_manager(...).run(...)",
    ])

    rows = []

    # 1 & 2 -- VF2Layout
    try:
        from qiskit.transpiler.passes.layout import vf2_layout as m
        src = inspect.getsource(m)
        check(rows, "vf2_layout.py: rustworkx import",
              "rustworkx" not in src,
              "not found" if "rustworkx" not in src else
              "\n".join(l for l in src.splitlines() if "rustworkx" in l))
        seed_lines = [l.strip() for l in src.splitlines()
                      if re.search(r"seed\s*==\s*-1|shuffle_seed|randrange", l)]
        check(rows, "vf2_layout.py: seed handling", None,
              "\n".join(seed_lines) or "<no seed branch found>")
    except Exception as e:  # noqa: BLE001
        check(rows, "vf2_layout.py", False, e)

    # 3 & 4 -- preset construction
    try:
        import qiskit.transpiler.preset_passmanagers.builtin_plugins as bp
        src = inspect.getsource(bp)
        vf2 = re.findall(r"VF2(?:Post)?Layout\((?:[^()]|\([^()]*\))*\)", src)
        joined = "\n".join(re.sub(r"\s+", " ", v) for v in vf2)
        check(rows, "builtin_plugins.py: VF2 constructions",
              bool(vf2) and all("seed=-1" in v for v in vf2),
              joined or "<none found>")
    except Exception as e:  # noqa: BLE001
        check(rows, "builtin_plugins.py", False, e)

    # 5 -- get_vf2_limits
    try:
        from qiskit.transpiler.preset_passmanagers import common
        check(rows, "common.get_vf2_limits", None, source_of(common.get_vf2_limits))
    except Exception as e:  # noqa: BLE001
        check(rows, "common.get_vf2_limits", False, e)

    # 6 -- transpile wrapper
    try:
        from qiskit.compiler import transpiler as ct
        src = inspect.getsource(ct.transpile)
        ok = "generate_preset_pass_manager" in src and "pm.run(" in src
        keep = [l.strip() for l in src.splitlines()
                if "generate_preset_pass_manager" in l or "pm.run(" in l]
        check(rows, "transpile() delegates to the preset pass manager", ok,
              "\n".join(keep) or "<not found>")
    except Exception as e:  # noqa: BLE001
        check(rows, "transpile()", False, e)

    # 7 -- psf_zero_core build
    try:
        import psf_zero_core
        f = getattr(psf_zero_core, "__file__", "")
        d = os.path.dirname(f) if f else ""
        has = hasattr(psf_zero_core, "geometric_decompose_checked")
        files = []
        for pat in ("*.pyd", "*.so", "*.dll", "*.py"):
            for g in sorted(glob.glob(os.path.join(d, pat))):
                ts = datetime.datetime.fromtimestamp(os.path.getmtime(g)).isoformat(
                    timespec="seconds")
                files.append(f"{os.path.basename(g)}  {os.path.getsize(g)} bytes  {ts}")
        check(rows, "psf_zero_core.geometric_decompose_checked present", has,
              "\n".join(files) or "<no files listed>")
    except Exception as e:  # noqa: BLE001
        check(rows, "psf_zero_core", False, e)

    # 8 -- psf_compile verify path
    try:
        import psf_compile
        src = inspect.getsource(psf_compile)
        check(rows, "psf_compile.py: _CORE_CHECKED present",
              "_CORE_CHECKED" in src,
              "\n".join(l.strip() for l in src.splitlines()
                        if "_CORE_CHECKED" in l or "geometric_decompose_checked" in l))
    except Exception as e:  # noqa: BLE001
        check(rows, "psf_compile.py", False, e)

    write_csv(rows, args.out or default_out("qiskit_source_check"), environment())
    print("\n  -> If P1-P4 are all OK, the 2026-09-14 addendum applies to 2.5.2 as written.")
    print("     Any NG means that section needs rewriting scoped to 'on main'.")


if __name__ == "__main__":
    main()
