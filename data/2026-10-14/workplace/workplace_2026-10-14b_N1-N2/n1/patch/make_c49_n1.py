"""make_c49_n1.py -- builds the workplace's N1 proposal on prototype 2026-10-14.c49 by exact substitutions: item 86, a
Pauli evolution whose two-qubit terms form one chain is compiled by formula as a brickwork on a path of the device
(no STEINER6, no size limit, no error value read). Every other input is compiled exactly as by c49.
    python make_c49_n1.py <c49 psf_compile.py> <n1_code.py> <output psf_compile.py>"""
import sys

src, code, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
n1 = open(code, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub("VERSION: 2026-10-14.c49 -- prototype (from c48; c43 = release 2026-10-11.4): items 75 revised, 77-80, 82-85",
    "VERSION: 2026-10-14.c49+n1 -- workplace proposal on prototype c49 (c43 = release 2026-10-11.4): items 75 "
    "revised, 77-80, 82-86")
sub('VERSION = "2026-10-14.c49"  # prototype:', 'VERSION = "2026-10-14.c49+n1"  # workplace proposal, item 86 (chain brickwork) on prototype:')
sub('''    is compiled exactly as by c48. `CUT_FAILED_QUBITS = False` restores c48. Counts in `CUT_STATS`.
''', '''    is compiled exactly as by c48. `CUT_FAILED_QUBITS = False` restores c48. Counts in `CUT_STATS`.
86. **DURATION: a chain Hamiltonian built by formula as a brickwork (workplace proposal N1 on c49).** W2 (2026-10-14)
    found that Qiskit level 3 given a hand-written brickwork circuit equals PSF-Zero on TFIM chains up to n = 50, and
    beats it at n = 100 (item 70's size limit sent 10 Trotter steps to the usual path: 82 times longer; on
    FakeMarrakesh STEINER6's placement around the failed couplers was not a clean path: 2.9-4.1 times longer). With the
    device given (`steiner` "auto" true) and its two-qubit gate cz, an input whose terms (read by items 72-73, merged
    where exact) act on one or two qubits, the two-qubit ones forming one path through all the circuit's qubits, is
    compiled without STEINER6 and without item 70's limit: W5's cut (failed qubits and couplers removed, largest
    connected component kept); the chain on the longest surviving piece of Heron's snake, centred, or else on the
    first path rustworkx's VF2 finds with nodes in index order (no score) -- both tried first on the couplers of the
    device's most common two-qubit gate length (gate lengths are fixed per device), then on all couplers; every term
    at level 1 + the largest level of an earlier term that shares a qubit and does not commute with it, emitted by
    level, even edges of the chain, then odd edges, then single-qubit terms (exact: every non-commuting pair keeps its
    order); Qiskit's level-3 preset translates on that fixed layout with no routing, given the coupling map and basis
    only. On W2's 48 cells: CZ at or below the hand-written brickwork's in 48, duration in 46 (0.80-1.02 of it); the
    two exceptions are FakeMarrakesh n = 100 (8 and 10 steps, +0.2% and +1.6%), where every 100-qubit path uses
    80 ns couplers. The brickwork is then compared with c49's own output for the same call by item 84's rule and
    returned unless c49's is shorter (a tie keeps the brickwork): on a device whose gate lengths differ from qubit to
    qubit (a generic test backend) c49's choices can give the shorter schedule. Any other input, or a chain with no
    path of n qubits, goes on as by c49. `CHAIN_BRICKWORK = False` restores c49. Counts in `CHAIN_STATS` ("kept",
    "kept_c49").
''')
sub("\ndef compile_for_hardware(\n", "\n" + n1 + "\n\ndef compile_for_hardware(\n")
s = s.rstrip("\n") + "\n\n\nimport inspect as _inspect  # noqa: E402  (item 86)\n_CFH_PARAMS = set(_inspect.signature(compile_for_hardware).parameters)\n"
sub('''    if (CUT_FAILED_QUBITS and not _inner and target is not None and coupling_map is None and initial_layout is None
            and _refine_target is None and layout_edge_errors is None and layout_qubit_errors is None):  # item 85''',
    '''    if (CHAIN_BRICKWORK and not _CHAIN_BUSY[0] and steiner and not _inner and target is not None
            and coupling_map is None and initial_layout is None
            and not (final_resynthesis or compare_level3 or compare_floor)):  # item 86
        out86 = _chain_brickwork(qc, target, prune_max_error, seed_transpiler)
        if out86 is not None:  # kept unless c49's own output is shorter (item 84's rule; a tie keeps item 86's)
            a86 = {k: v for k, v in locals().items() if k in _CFH_PARAMS}
            _CHAIN_BUSY[0] += 1
            try:
                own = compile_for_hardware(**a86)
            except FailedElementsError:
                own = None
            finally:
                _CHAIN_BUSY[0] -= 1
            if own is not None and _better(own, out86, target):
                CHAIN_STATS["kept_c49"] += 1
                return own
            CHAIN_STATS["kept"] += 1
            return out86
        del out86  # item 50 copies this call's locals as its arguments
    if (CUT_FAILED_QUBITS and not _inner and target is not None and coupling_map is None and initial_layout is None
            and _refine_target is None and layout_edge_errors is None and layout_qubit_errors is None):  # item 85''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
