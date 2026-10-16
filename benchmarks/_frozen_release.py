"""Tests written for psf_compile 2026-10-14.1 and earlier call `use()` before `import psf_compile`: it installs the
frozen 2026-10-14.1 (patches/psf_compile_release_2026-10-14.1/psf_compile.py) as the module `psf_compile`, since the
release at the repository root (2026-10-16.1) no longer has `compile()` and the other functions those tests use."""
import importlib.util
import os
import sys

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches", "psf_compile_release_2026-10-14.1",
                      "psf_compile.py")
_MOD = []


def use():
    """Make `import psf_compile` give 2026-10-14.1 (loaded once per process)."""
    if not _MOD:
        spec = importlib.util.spec_from_file_location("psf_compile", os.path.abspath(FROZEN))
        m = importlib.util.module_from_spec(spec)
        sys.modules["psf_compile"] = m
        spec.loader.exec_module(m)
        _MOD.append(m)
    sys.modules["psf_compile"] = _MOD[0]
    return _MOD[0]
