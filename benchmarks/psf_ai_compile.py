"""Redirect to the AI front end a23 (2026-10-16), kept where it was built: patches/psf_ai_compile_a23_2026-10-16/.

This file held a full copy of a23 (the adopted front end of 2026-10-16), identical to the patch's file except one
comment. The current front end is a26 (see CURRENT.md); two scripts here (pl_redo.py, pl_gpu_redo.py) import
`psf_ai_compile` and were written against an earlier front end, so this name keeps giving a23 rather than following
the current one. Replaced by this redirect on 2026-11-13 so that no second copy can drift (as psf_compile.py's did,
spare-qubit-cliff Addenda 187-188).
"""
import importlib.util as _ilu
import os as _os
import sys as _sys

_FILE = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, "patches",
                                       "psf_ai_compile_a23_2026-10-16", "psf_ai_compile.py"))
_spec = _ilu.spec_from_file_location(__name__, _FILE)
_module = _ilu.module_from_spec(_spec)
_sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
