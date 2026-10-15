"""make_c57_97.py -- workplace proposal, 2026-10-15: prototype c57 + item 97, by exact substitutions.

Item 97 (orientation rule). Before c55's path compiles an in-scope circuit (item 96 says it embeds), every top-level
two-qubit instruction that is symmetric under exchanging its qubits is rewritten with the smaller qubit index first:
cz, cp, rzz, rxx, ryy, swap, iswap, and a two-qubit PauliEvolutionGate whose operator equals its own qubit-exchanged
copy (ZZ, XX + YY + ZZ, ...). The operation and its parameters are unchanged, so the circuit is the same unitary;
instruction order is unchanged. Nothing is compared or chosen: the rule decides. An input handed to Qiskit's level 3
(out of scope) is not touched, so the hand-off stays identical to Qiskit's output for the input as written.
ORIENT97 = False restores c57. Counts in ORIENT97_STATS.

    python make_c57_97.py <c57 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


sub('VERSION = "2026-10-14.c57"  # prototype: ',
    'VERSION = "2026-10-14.c57+97"  # workplace proposal: item 97 (symmetric two-qubit instructions written smaller '
    'index first, in scope) + prototype c57: ')
sub('''SCOPE_GATE = True  # item 96
''', '''SCOPE_GATE = True  # item 96
ORIENT97 = True  # item 97
ORIENT97_STATS = {"calls": 0, "rewritten_calls": 0, "flipped": 0}
_SYM97 = frozenset(("cz", "cp", "rzz", "rxx", "ryy", "swap", "iswap"))


def _sym97(op):
    """Item 97: True if the two-qubit `op` is the same operation with its qubits exchanged."""
    if op.name in _SYM97:
        return True
    if op.name == "PauliEvolution":
        try:
            from qiskit.quantum_info import SparsePauliOp
            ham = op.operator
            ops = ham if isinstance(ham, list) else [ham]
            for h in ops:
                h = SparsePauliOp(h).simplify()
                swapped = SparsePauliOp.from_list([(lab[::-1], c) for lab, c in h.to_list()]).simplify()
                if not h.equiv(swapped):
                    return False
            return True
        except Exception:  # noqa: BLE001
            return False
    return False


def _orient97(qc):
    """Item 97: `qc` with every top-level symmetric two-qubit instruction written smaller index first (same
    operation, same parameters, same order); `qc` itself when nothing changes."""
    ORIENT97_STATS["calls"] += 1
    flips = []
    for k, ins in enumerate(qc.data):
        if len(ins.qubits) == 2 and not ins.clbits and getattr(ins.operation, "condition", None) is None:
            a, b = (qc.find_bit(q).index for q in ins.qubits)
            if a > b and _sym97(ins.operation):
                flips.append(k)
    if not flips:
        return qc
    out = qc.copy_empty_like()
    fl = set(flips)
    for k, ins in enumerate(qc.data):
        out.append(ins.replace(qubits=ins.qubits[::-1]) if k in fl else ins)
    ORIENT97_STATS["rewritten_calls"] += 1
    ORIENT97_STATS["flipped"] += len(flips)
    return out
''')
sub('''    if in_scope:
        SCOPE96_STATS["in_scope"] += 1
        _S96_DEPTH[0] += 1
        try:
            return _compile_for_hardware_c55(qc, **kwargs)''', '''    if in_scope:
        SCOPE96_STATS["in_scope"] += 1
        _S96_DEPTH[0] += 1
        try:
            return _compile_for_hardware_c55(_orient97(qc) if ORIENT97 else qc, **kwargs)  # item 97''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
