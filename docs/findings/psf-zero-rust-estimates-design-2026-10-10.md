# Item 57b: the estimate and check loops in the Rust core -- design and prototype (2026-10-10)

**Status: design, with a prototype of the core loop tested against a NumPy copy of the Python code** (written in the
morning of 2026-10-10; Addendum 425). The candidate built from it, c30, is pre-registered in Addendum 426.

## 1. Why

- **The slow tests that item 56 does not help.** Item 56 (candidate c29) removes the estimates that cannot change
  the result (type B, hwb10). The other slow development tests are type A, where the checks can be made: JW-14,
  enc_gray_dvalues_8-8-8, JW-10 and parity10 ([weakness report](psf-zero-weakness-report-2026-10-09.md)). There the
  recommended call spends much of its time in:
  - `excitation_cost` and `hybrid_cost`, which simulate the circuit's state gate by gate;
  - item 39's checks, which simulate both circuits (`_apply_ops`).
- **The time is in the loop, not in one step.** Addendum 415's model is about 20 µs per gate application plus 46-57
  ns per amplitude at the workplace. Making one step cheaper in NumPy (item 57a, Addendum 421) saved 2-11%. The time
  goes to three things:
  - Python's work per gate: the Target lookup, `find_bit`, the matrix, `tensordot` and `moveaxis` dispatch;
  - NumPy's work per amplitude: a copy at each `moveaxis` and `tensordot`;
  - the passes that form the reduced states.

## 2. What moves, and what stays

| part | where | why |
|---|---|---|
| reading the circuit: instructions, qubits, matrices (`_gate_matrix`), the Target's error and duration, T1 and T2 | Python | it reads Qiskit objects; the per-op work is a few lookups, without NumPy dispatch |
| the cases that return None (over 16 touched qubits, an instruction without a matrix) | Python | decided while reading, before the core is called |
| item 49's walk: single-qubit gates waiting in `pend`, each qubit's 2x2 `rho`, the state updated at each wider gate | Rust | the loop itself |
| the cost terms of `excitation_cost` and of `hybrid_cost`'s gates | Rust | read inside the loop |
| `readout_cost` (added by `hybrid_cost`) | Python | it reads measurements, not gates |
| `_apply_ops` in item 39's checks | Rust | the same state update, without the waiting |
| preparing the product states, the reduction and the overlap in `_implements` / `_same_action` | Python | once per check |

**The call.** Python packs one byte buffer per call, and the core returns a float (the estimate) or a state.

- The buffer carries:
  - a header;
  - T1 and T2 per touched qubit;
  - each instruction's width, positions, has-properties flag, error and duration;
  - the matrices, interleaved real and imaginary;
  - for `_apply_ops`, the initial state.
- No NumPy binding is needed in the core: pyo3 0.19's `&PyBytes` is enough, and the core's other dependencies do not
  change.
- The format is specified in `parse` (`src/statevec.rs`).

**Conventions kept**, as in psf_compile.py:

- axis j of the state is the j-th touched qubit, which is bit (k - 1 - j) of the flat index;
- Qiskit's little-endian matrices: bit j of the index belongs to q[j];
- `_embed_1q`'s Kronecker order;
- the order of every cost term.

**When the core is older** (it has no item 57b functions), the Python code runs as now. The path taken is counted.

## 3. The prototype

It is a standard-library-only Rust module, `statevec.rs`, with a driver `main.rs` that reads buffers from files. A
cross-check, `ref57.py`, copies the bodies of c29's `excitation_cost`, `hybrid_cost` and `_apply_ops` with the
circuit and Target replaced by plain lists. A timing script, `bench57.py`, measures both.

- **Two-qubit gates in one pass.** A two-qubit gate is applied in one pass over the state. The same pass forms both
  qubits' 2x2 reduced states from the new amplitudes. This is item 57a's idea, fused with the gate: the Python code
  needs three passes.
- **Other gates.** Wider gates use a general loop followed by one pass per qubit. Single-qubit gates wait, as in
  Python.

**Correctness.** On 300 random problems:

- 1-10 qubits;
- up to 60 one-, two- and three-qubit unitaries;
- missing and zero T1, T2, errors and durations mixed in.

The largest relative differences from the NumPy code are 4.6e-16 (`excitation_cost`) and 4.3e-16 (`hybrid_cost`).
The largest amplitude difference after `_apply_ops` is 1.3e-15. The differences come only from the order of sums,
which NumPy chooses differently. Unit tests check:

- Qiskit's qubit order;
- the waiting single-qubit gates;
- that the fused pass equals the general one;
- that malformed buffers are rejected.

**Speed** (this container: a shared 2.8 GHz Xeon, 2 vCPUs, one thread). The setup is 2,000 gates, a third of them
two-qubit, with properties on every gate.

| touched qubits | NumPy copy of the Python code | Rust (whole process) | per amplitude per two-qubit gate |
|---|---|---|---|
| 10 | 0.10 s | 0.012 s | about 11 ns |
| 14 | 0.28 s | 0.10-0.13 s | about 9 ns (6 ns without forming the reduced states) |

**What the figures mean.**

- Up to about 12 qubits the gain is large: Python's per-gate work dominates there, and the core removes it.
- At 14-16 qubits the core is 2-3 times as fast: the arithmetic per amplitude dominates there, and the core is only
  modestly cheaper than NumPy's.
- What remains in Python is the reading per instruction, which this container cannot measure: Qiskit is not
  installed here. The real gain on the type-A tests must be measured on whole compiles.

## 4. Identity

- **The values differ from Python's in their last bits.** A relative difference of about 1e-16 can change a decision
  only where two estimates lie at the edge of item 46's tie band (`ESTIMATE_TIE_TOL` = 1e-12 relative). This is not
  identity by construction. As for item 56, it is tested before it is proposed:
  1. **Values.** On random circuits transpiled for FakeTorino, and on every estimate the development tests make,
     each value from the core is within 1e-12 (relative) of Python's, and every check returns the same boolean.
  2. **Outputs.** The development tests, by value, where the release repeats itself, as in C29-ID.
  3. **Speed.** Whole-compile times on the type-A tests, reported.
- **The core is used only where it is present.** Without it, the Python code runs.

## 5. Next steps

1. Write the pyo3 binding (two functions taking the buffer) and the Python side (the buffer, the fallback, a counter
   of the path taken), on candidate c29.
2. Build the core with `maturin develop --release` on the home PC, and run the unit tests and the value test there.
3. Pre-register the identity test, as above.
4. Later, if the 14-16-qubit cases still dominate: threads over the state for k >= 14. That changes the order of
   sums again, which needs the same tests.

## 6. What this does not address

- Type B slow tests: item 56 handles them.
- The fixed cost of the recommended call on small circuits (TOQB run 1c, weakness report): another weakness.
- `pauli_cost` and `kraus_cost`: the recommended call does not use them.
