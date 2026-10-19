# psfz -- PSF-Zero in Rust, prototype r4 (2026-10-18)

**Used by `psf_compile.py` since release 2026-10-18.1 (item 105, Addendum 525):** the default call runs this program
on every circuit first and returns its output when it compiles the circuit. Build it once with
`cargo build --release` here; `psf_compile.py` finds `target/release/psfz` (or `$PSFZ_BIN`). Validated in PSFZ-VAL2
(Addendum 522) and, through `psf_compile.py`, in C68-VAL (Addendum 524).

A new compiler with a different structure from `psf_compile.py`: one Rust program, no outside crate, no Qiskit. It reads
a gate-level OpenQASM 3 circuit and a device file (JSON), and writes OpenQASM 3 on the device's qubits with the
device's gates only (rz, sx, x and cz on IBM Heron; ecr on IBM Eagle). The owner's decisions (2026-10-17, Addendum 518):
chains and rings only; OpenQASM in and out; placement by the lattice's coordinates only (no VF2); judged by
exactness and by duration against release 2026-10-17.1.

**No search, no estimate** (r1-r3; r4 adds a computed longest path and a per-gate choice of the ECR form, below). Placement: the heavy-hex lattice is read as rows, column offsets and bridges; a chain goes
on the longest working piece of the snake (centred), a ring on the boundary of a band of hexagons; candidates come in
a fixed order and the first whose qubits and couplers all work (first among couplers of the most common duration) is
used. Nothing is scored, matched or simulated. Every rewrite (diagonal gates as ZZ and phases, ZZ on CZ, the 3-CZ
canonical form, CZ on ECR, single-qubit runs as rz/sx/x) is a closed form, checked once by the unit tests.

    cargo build --release            # rust/psfz; no network needed (no dependencies)
    cargo test                       # the closed forms, the readers
    target/release/psfz compile DEVICE.json IN.qasm OUT.qasm [REPORT.json]

Exit 0: compiled; 2: refused (out of scope; the reason on stderr and in the report); 1: an input error.

**r2 (Addendum 519):** rings of more shapes (straight, bulging and narrowing sides), and a rule for which qubit of a
pair takes a form's basis changes; on the exploratory set, longer than release 2026-10-17.1 in 2 of 350 inputs (by
0.06% and 0.23%).

**r3:** consecutive two-qubit rotations on one pair, with the single-qubit operations between them, form one
block, multiplied out and written by a Weyl (KAK) decomposition in closed form (`src/kak.rs`, no outside crate),
3 CZ at most (2 for two rotation kinds); validated in PSFZ-VAL (`data/2026-10-17/psfz_val/`).

**r4 (Addendum 521):** two changes, both found on PSFZ-VAL's inputs (so exploratory until a fresh validation).
(1) A chain whose snake has no long enough piece on couplers of the most common duration goes on the longest
*row-and-bridge path* on them (`place::row_bridge_path`): along a stretch of a row, across a bridge, along a stretch of
the next row in either direction, and so on; the snake is the member that always turns at the row's end. The longest
member is computed exactly, row by row (dynamic programming over columns and bridges), not searched for; inputs whose
snake suffices are placed as before. (2) On ECR devices each CZ becomes an ECR in one of the 64 forms
CZ = (A x B) ECR (C x D) with C and D single-qubit Cliffords (`compile::ecr_forms`), the one that lets the ECR start
earliest given what is pending on its two qubits (`compile::ecr_adaptive`; ties: fewer pulses after it), one gate at a
time in program order. This is a choice by a computed schedule, as r2's rule for a form's basis changes already was;
nothing is revisited. On PSFZ-VAL's 2,860 inputs: never longer than release 2026-10-17.1, 0.96 x on the geometric mean.

**r5 prototype, templates (2026-10-19, exploratory; `psfz template`, `psfz bind`):** every single-qubit gate of
the input with parameters becomes a slot, carried through the compiler as a factor; every decision that read a value
reads the structure instead, and every run with a slot is written in the general form rz sx rz sx rz. So the gate list
depends on the input's structure and the device only, and `bind` recomputes each run's three angles in closed form
(ZYZ of the product of its factors): 0.28 ms for a 12-qubit, 16-layer ring classifier. `compile` is r4's, unchanged
(96 of 96 outputs identical). Checked in `data/2026-10-19/psfz_r5_template_check/` (exact to 2.4e-16; TFIM templates
16-19 % more sx than r4). Two-qubit gates with parameters are not slots; ECR devices are refused.

**MPS twin prototype (2026-10-19, exploratory; `psfz twin`):** the chain output simulated as a matrix product
state, noiseless, every bond dimension bounded by a formula before the run (CZ crossing each cut); exact SVD splits.
80-qubit TFIM chains, 6 steps: 42 s on one core, dropped weight 1e-23; <Z> within 6.5e-15 of a state vector up to 20
qubits (`data/2026-10-19/psfz_mps_twin_check/`). The dependency-free SVD is the bottleneck; rings and noise not yet.

**Scope (r1-r4):** every qubit in one chain or one even ring of two-qubit gates; gates id x y z h s sdg t tdg sx sxdg rx
ry rz p u1 u2 u3 u, cx cz cp crz rzz rxx ryy; measurements only at the end. Not yet: other graphs, fractional
gates (rzz/rx natives), templates, mid-circuit measurement.

**Checking (`harness/`, Python, not part of the compiler):** `compare.py` runs psfz and release 2026-10-17.1 on
C63-VAL's 14 chain and ring tests on five fake devices (exactness against Qiskit's state vector); `summarize.py`.
