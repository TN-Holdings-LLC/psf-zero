"""apply_layout_patch.py -- task 1: replace the repository's own
`psf_smart_layout.py` with the verified release version, safely.

Steps, in order:
  1. Refuse to run if `psf_smart_layout_release.py` is missing, or if its
     SHA256 does not match the value verified in the session that produced
     it -- so a corrupted or wrong download cannot be installed by mistake.
  2. Back up the current `psf_smart_layout.py` to a timestamped file. Never
     overwrite without a backup.
  3. Copy the release file into place as `psf_smart_layout.py`.
  4. Import it and run a smoke check: the guard must accept a known-feasible
     chain-shaped input (the class of input the OLD guard rejected) and
     reject a known-impossible one. If either fails, restore the backup
     automatically and report.

Usage:
    python apply_layout_patch.py
"""
from __future__ import annotations

import hashlib
import importlib
import shutil
import sys
import time
from pathlib import Path

EXPECTED_SHA256 = "8ce108f3293ac4fc5d9849b1158d4bdd31fac41655247be2b740451ca4646f48"
RELEASE = Path("psf_smart_layout_release.py")
TARGET = Path("psf_smart_layout.py")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    # 1. Verify the release file is exactly the verified one.
    if not RELEASE.exists():
        print(f"ERROR: {RELEASE} not found in the current directory.")
        return 1
    got = sha256(RELEASE)
    if got != EXPECTED_SHA256:
        print("ERROR: release file hash mismatch -- refusing to install.")
        print(f"  expected {EXPECTED_SHA256}")
        print(f"  got      {got}")
        return 1
    print(f"OK: {RELEASE} hash matches the verified release.")

    # 2. Back up.
    if TARGET.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = Path(f"psf_smart_layout.BACKUP-{stamp}.py")
        shutil.copy2(TARGET, backup)
        print(f"OK: backed up existing {TARGET} -> {backup}")
    else:
        backup = None
        print(f"NOTE: no existing {TARGET}; nothing to back up.")

    # 3. Install.
    shutil.copy2(RELEASE, TARGET)
    print(f"OK: installed {RELEASE} as {TARGET}")

    # 4. Smoke check on the installed module.
    try:
        if "psf_smart_layout" in sys.modules:
            del sys.modules["psf_smart_layout"]
        mod = importlib.import_module("psf_smart_layout")
        from qiskit.transpiler import CouplingMap

        cm8 = CouplingMap.from_grid(8, 8)
        # A chain-shaped input the OLD guard wrongly rejected (17 bare edges +
        # one 30-qubit chain, 46 edges > 32 max matching). Must be accepted.
        chain = [(q, q + 1) for q in range(0, 34, 2)]
        chain += [(34 + i, 35 + i) for i in range(29)]
        ok_feasible = mod._has_feasible_matching(cm8, chain)

        # A genuinely impossible input (32 disjoint pairs on a 4x4 grid whose
        # max matching is 8). Must be rejected.
        cm4 = CouplingMap.from_grid(4, 4)
        impossible = [(2 * i, 2 * i + 1) for i in range(32)]
        ok_reject = not mod._has_feasible_matching(cm4, impossible)

        # The natural ordering must be first.
        import rustworkx as rx
        g = rx.PyGraph()
        for i in range(cm8.size()):
            g.add_node(i)
        for a, b in cm8.get_edges():
            if not g.has_edge(a, b):
                g.add_edge(a, b, None)
        first_name, _ = next(iter(mod._candidate_orderings(g)))
        ok_natural = first_name == "natural"

        print(f"smoke: feasible chain accepted   = {ok_feasible}")
        print(f"smoke: impossible input rejected = {ok_reject}")
        print(f"smoke: natural ordering is first = {ok_natural}")
        if not (ok_feasible and ok_reject and ok_natural):
            raise RuntimeError("smoke check failed")
    except Exception as exc:  # noqa: BLE001 -- any failure must restore
        print(f"ERROR during smoke check: {exc!r}")
        if backup is not None:
            shutil.copy2(backup, TARGET)
            print(f"RESTORED {TARGET} from {backup}")
        return 1

    print("DONE: psf_smart_layout.py replaced and smoke-checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
