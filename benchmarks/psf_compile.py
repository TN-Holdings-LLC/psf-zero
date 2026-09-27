"""Redirect to the one psf_compile.py, at the repository root.

This file holds no code of its own. Scripts under benchmarks/ used to import
a second, full copy of psf_compile.py from here; two copies drifted apart
unnoticed and one experiment silently ran the wrong one (spare-qubit-cliff
Addenda 187-188). Importing `psf_compile` from anywhere now loads the root
file, and `psf_compile.__file__` names it.
"""
import importlib.util as _ilu
import os as _os
import sys as _sys

_ROOT_FILE = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir, "psf_compile.py"))
_spec = _ilu.spec_from_file_location(__name__, _ROOT_FILE)
_module = _ilu.module_from_spec(_spec)
_sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
