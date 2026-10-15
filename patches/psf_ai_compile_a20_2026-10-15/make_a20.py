"""make_a20.py -- the AI front end 2026-10-15.a20 from a19 by exact substitutions: item 35, the ring template (the
ring formula built once by PSF-Zero itself, bound to new angles by closed-form single-qubit synthesis); item 36, the
gate closed (no input handed to Qiskit's level 3: without a Target, or out of PSF-Zero's scope, OutOfScopeError).
    python make_a20.py <a19 psf_ai_compile.py> <a20_template_code.py> <output psf_ai_compile.py>"""
import sys

src, code, dst = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
add = open(code, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub('"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a19\n'
    '(a18 + item 34: formula only for a ring classifier). a18\'s description:\n',
    '"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model, prototype 2026-10-15.a20\n'
    '(a19 + item 35: the ring template; item 36: the gate closed). a18\'s description:\n')
sub('''Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''', '''  35. (a20) The ring template. `ring_template(qc, target)` builds the ring formula once by PSF-Zero itself, with no
      Qiskit pass manager: the fixed CZ / SWAP blocks are decomposed once and checked to 1e-12, and every
      single-qubit run between CZs is kept as a product of the input's own gates and fixed matrices.
      `RingTemplate.bind(qc)` compiles a circuit of the same structure with other angles by re-synthesising only those
      runs in closed form (ZYZ: rz sx rz sx rz, or rz rx rz with rx in the Target). For a training loop, where only the
      angles change. The owner's request of 2026-10-15 (Addendum 494).
  36. (a20) The gate closed: no input is handed to Qiskit's level 3. Without a Target, OutOfScopeError; an input that
      is not a ring classifier goes to psf_compile's own call, which builds it by formula or refuses it
      (GATE_CLOSED = False restores a19's hand-off).
Everything else is a15's: candidates chosen by psf_compile's `_better` (duration, then two-qubit count), no seed or
placement search, no error value or T1/T2 decides; failed elements never used (item 26); a11's direction backstop.
"""''')
sub('AI_COMPILE_VERSION = "2026-10-15.a19"  # prototype: item 34 (formula only for ring classifiers) on a18, which was:',
    'AI_COMPILE_VERSION = "2026-10-15.a20"  # prototype: items 35-36 (ring template; gate closed) + item 34 (formula only) on a18, which was:')
sub('''    seed = seeds[0] if seeds else 0
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if target is not None:''', '''    seed = seeds[0] if seeds else 0
    if GATE_CLOSED and target is None:  # item 36
        raise OutOfScopeError("PSF-Zero compiles only for a device's Target (changelog item 36); without one, use "
                              "Qiskit yourself")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if target is not None:''')
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
