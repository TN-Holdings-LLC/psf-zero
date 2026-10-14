"""make_c51_87_88.py -- builds the workplace's proposal of items 87-88 on prototype 2026-10-14.c51 by exact
substitutions: item 87, a Pauli evolution whose two-qubit terms form one chain is compiled by formula as a brickwork
on a path of the device (N1, renumbered); item 88, with rzz and rx in the Target, its terms are emitted as rzz / rx
with the signs carried in a Pauli X frame that is undone inside the circuit. Every other input is compiled exactly as
by c51. The code is appended at the end of the file; c51's wrapper `compile_for_hardware` is renamed
`_compile_for_hardware_c51` and wrapped once more.
    python make_c51_87_88.py <c51 psf_compile.py> <c51_87_88_code.py> <output psf_compile.py>"""
import sys

src, code, dst = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src, encoding="utf-8").read()
add = open(code, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


# 1. version (home: adjust these two to c52's strings when merging)
sub("VERSION: 2026-10-14.c51 -- prototype (from c49 as c50, item 86 revised; c43 = release 2026-10-11.4): items 75 "
    "revised, 77-80, 82-86",
    "VERSION: 2026-10-14.c51+87-88 -- workplace proposal on prototype c51 (c43 = release 2026-10-11.4): items 75 "
    "revised, 77-80, 82-88")
sub('VERSION = "2026-10-14.c51"  # prototype:',
    'VERSION = "2026-10-14.c51+87-88"  # workplace proposal, items 87-88 (chain brickwork, rzz in a Pauli frame) on prototype:')
# 2. changelog, after item 86
sub('''    the folded circuit. `FRACTIONAL = False` turns off the re-synthesis only; the folding stays, since an unfolded
    rzz does not run.
"""''', '''    the folded circuit. `FRACTIONAL = False` turns off the re-synthesis only; the folding stays, since an unfolded
    rzz does not run.
87. **DURATION: a chain Hamiltonian built by formula as a brickwork (workplace proposal N1, on c51).** W2
    (2026-10-14) found that Qiskit level 3 given a hand-written brickwork circuit equals PSF-Zero on TFIM chains up to
    n = 50, and beats it at n = 100 (item 70's size limit sent 10 Trotter steps to the usual path: 82 times longer; on
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
    only. On W2's 48 cells (c49 + N1): CZ at or below the hand-written brickwork's in 48, duration in 46 (0.80-1.02 of
    it); the two exceptions are FakeMarrakesh n = 100 (8 and 10 steps, +0.2% and +1.6%), where every 100-qubit path
    uses 80 ns couplers (accepted by home: the path is not chosen by duration). The brickwork is compared with c51's
    own output for the same call (c49, then item 86) by item 84's rule and returned unless c51's is shorter (a tie
    keeps the brickwork). The comparison is made in the outermost wrapper (`compile_for_hardware`, which wraps c51's
    `_compile_for_hardware_c51`); nested calls go straight to c51. Any other input, or a chain with no path of n
    qubits, goes on as by c51. `CHAIN_BRICKWORK = False` restores c51. Counts in `CHAIN_STATS`.
88. **DURATION: item 87's terms as the device's rzz / rx, the signs in a Pauli frame (workplace proposal, on
    c51).** With rzz and rx in the Target (`CHAIN_FRAME`), item 87 emits each two-qubit term as one rzz between
    single-qubit basis changes and each X term as rx, drops diagonal gates directly before measurements, and makes
    every rzz angle executable (0 < theta <= pi/2) without a second rzz: angles reduced to (-pi, pi], those of size
    above pi/2 moved by pi with Z x Z (two virtual rz(pi)), and a negative angle made positive by one x before the rzz
    (merged into the single-qubit run before it) while the x after it is kept in a Pauli X frame (single-qubit gates
    conjugated, later rzz signs flipped). The frame is undone inside the circuit, by one x before the measurement or
    at the end of an unmeasured qubit, merged into the last single-qubit gate where there is one: the circuit outputs
    the input's distribution, and acts as its unitary when not measured; no measured bit is flipped afterwards. Every
    single-qubit run is merged into rz rx rz (one pulse). An angle left outside the range (none seen) goes through
    item 86's `_fold`. Without rzz or rx in the Target, item 87's output is the CZ brickwork (N1). `CHAIN_FRAME =
    False` gives the CZ brickwork on every Target.
"""''')
# 3. c51's wrapper renamed; the new wrapper is in the appended code
sub('''def compile_for_hardware(qc, *args, **kwargs):
    """Item 86 (prototype c50):''', '''def _compile_for_hardware_c51(qc, *args, **kwargs):
    """Item 86 (prototype c50):''')
sub('compile_for_hardware.__doc__ += _compile_for_hardware_c49.__doc__ or ""\n',
    '_compile_for_hardware_c51.__doc__ += _compile_for_hardware_c49.__doc__ or ""\n')
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
