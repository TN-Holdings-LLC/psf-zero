# PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-purple.svg)](https://github.com/qiskit/ecosystem)
[![Rust Core](https://img.shields.io/badge/Core-Rust_Native-E34F26.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3 Binding](https://img.shields.io/badge/FFI-PyO3-blue.svg)](https://pyo3.rs/)
[![DOI (Paper 1)](https://zenodo.org/badge/DOI/10.5281/zenodo.22977930.svg)](https://doi.org/10.5281/zenodo.22977930)
[![DOI (Paper 2)](https://zenodo.org/badge/DOI/10.5281/zenodo.22978090.svg)](https://doi.org/10.5281/zenodo.22978090)

PSF-Zero is a Qiskit-compatible compiler that builds circuits for a device by formula, with no search.

- **`compile_for_hardware()` and `compile_template()`: compilation for a device.** Since 2026-10-16.1, given the
  device's `Target` (a CZ device: IBM Heron), it builds by formula every circuit whose two-qubit interaction graph
  embeds in the device: chains, lattices and embeddable graphs, in a time-aware order, with IBM's native rzz where
  offered. **Every other circuit, and every call without the device, is refused with `OutOfScopeError`:** compile
  those with Qiskit yourself (until 2026-10-14.1 they were handed to Qiskit's level 3; the owner closed that path,
  Addenda 494 and 510). `compile_template(qc, target)` builds once and binds new angles in about a millisecond, for
  training loops.
- **`compile()`: fast, exact two-qubit synthesis**, with a closed-form Cartan (KAK) decomposition in a small Rust
  core. It is not in 2026-10-16.1, whose default calls never used it; it remains in the previous release,
  [`patches/psf_compile_release_2026-10-14.1/psf_compile.py`](patches/psf_compile_release_2026-10-14.1/psf_compile.py)
  (see [Quick start](#quick-start)).
- **Earlier (2026-10-14.1):** the formula path above, with every other circuit handed to Qiskit's level 3
  unchanged. Every output it can check is checked exact. On the circuits it builds, its outputs are
  0.20-0.23 times Qiskit level 3's duration on average (geometric mean, offline), and the time-aware order raised the fidelity on real devices
  8 of 8 times. See [Scope](#scope-where-psf-zero-competes-declared-2026-10-14). The earlier error-aware placement
  and noise-estimated choice, described in the release notes up to 2026-10-11.4, are not what decides any more.
- **General circuits: about level with Qiskit level 2, and slower.** On all 880 Benchpress transpilation tests that
  no earlier test of this project had used (BP-FINAL, pre-registered, Addenda 407-408), the default call of
  2026-10-07.1 used 1.046 times Qiskit level 2's two-qubit gates (geometric mean, 95% interval 1.037-1.055; fewer on
  117 tests, as many on 314, more on 446), at a median 3.5 times its compile time. See
  [Where it is weaker](#where-it-is-weaker).

Everything is pre-registered and self-audited: predictions are locked in git before a scored run, and results,
including the failures, are recorded in [`docs/findings/`](docs/findings/).

## Scope: where PSF-Zero competes (declared 2026-10-14)

**On real IBM devices, PSF-Zero's gains are concentrated where it builds the circuit by formula.** We pooled every
compiled pair run on a real device between 2026-10-12 and 2026-10-14: 54 pairs, four runs, three devices
(ibm_kingston, ibm_marrakesh, ibm_fez). The pooling was done after the fact
([Addendum 488](docs/findings/spare-qubit-cliff-combined-450.md)).

| | pairs | PSF-Zero's side won | F ratio (geometric mean) |
| :--- | :---: | :---: | :---: |
| Ising chains, built by formula | 23 | 19 | **1.21** (worst 0.94) |
| QAOA and Barabasi-Albert, compiler choices only | 25 | 9 | **0.78** |
| QAOA and Barabasi-Albert, with the device's native rzz | 6 | 5 | 1.56 |

- **Where the formula applies, PSF-Zero wins.** Two pre-registered tests on two devices showed the same gates in a
  time-aware order raising the fidelity by 1.17-1.51 times, 8 of 8 (HW2 part S, TIME1).
- **Elsewhere it does not.** Placement there decides the result, and placement needs search or calibration, which
  PSF-Zero does not use by design. In TIME1 its choice by duration lost to Qiskit level 3 on deep circuits.

**So PSF-Zero competes only where it can build by formula:**

- chains, rings and lattices that embed in the device;
- the device's native fractional gates (rzz, rx).

**Every other circuit is handed to Qiskit's level 3 unchanged**, so that PSF-Zero is no worse than Qiskit there.
Since release 2026-10-14.1 this is the default (item 96; C57-VAL, Addendum 491).

The other families are open to anyone who wants to work on them. Our losing data are published for that purpose:
the family map (Addendum 484), TIME1 (Addendum 487) and the pooled pairs (Addendum 488).

What these numbers are not:

- Many pairs repeat the same tests on different devices, so they are not independent trials.
- Rings are built by formula but have not run on a device yet.
- The release 2026-10-14.1 (prototype c57) has not run on a device as a whole. Its order (item 83) and rzz have.

## Current version

**`psf_compile.py` 2026-10-16.1** and the AI front end **a23** (Part 11, Addenda 503-510). The changelog of
`psf_compile` is [`docs/psf_compile_CHANGELOG.md`](docs/psf_compile_CHANGELOG.md). Every release and dated notice:
[`docs/RELEASES.md`](docs/RELEASES.md).

> **New in 2026-10-16.1: no search and no check is left on the default path, and the file is a third of its size.**
> The release is prototype c65 (Addenda 504-508): c63a (validated in C63-VAL and C63A-VAL), with the code its default
> calls never reach removed by the workplace (c64: 9,240 lines to 3,045, the same output on 49,590 inputs), and
> item 96's embedding test decided by formula before VF2 (item 102). The AI front end a23 replaces a22's
> compare-and-choose by a rule on the ring size (the same output, bit for bit).
>
> - **What changes for a user:** circuits that do not embed in the device, devices without CZ, and calls without
>   the device are refused with `OutOfScopeError` instead of being handed to Qiskit. `compile()` and the other
>   functions the default calls never used are not in this file (they remain in 2026-10-14.1, kept in `patches/`).
>   Only `target=`/`backend=`, `prune_max_error=` and `seed_transpiler=` are read; other options are ignored.
> - **Time:** against 2026-10-14.1 + a18, the compile of the same circuits takes 0.023 x (C63-VAL's tests) and
>   0.063 x (TOQB's ring circuits) per call (median, offline; Addendum 505).
> - **Known cost (option A, Addendum 508):** in `compile_template` on fractional (rzz) targets, inputs whose signs
>   match the build's are up to 1.21-1.35 x c62's duration; on mixed signs the median is 1.000.
> - Not run on a device as a whole.

**Earlier: `psf_compile.py` 2026-10-14.1** and the AI front end **a18**, with `psf_smart_layout` 2026-10-01.1 and the
Rust core `CORE_VERSION` 2026-09-29.1, and optionally `psf_zero_core57` 2026-10-10.c30 (Parts 10-11, Addenda 383-492).

> **New in 2026-10-14.1: PSF-Zero builds what it can build by formula, and hands everything else to Qiskit.** The
> release is prototype c57 (C57-VAL, every prediction confirmed; Addenda 490-491) and the AI front end a18
> (Addendum 489). See [Scope](#scope-where-psf-zero-competes-declared-2026-10-14) for why.
>
> - **The scope gate (item 96).** Given the device (`backend=` or `target=`), a circuit whose two-qubit interaction
>   graph embeds in the device's working couplers is compiled by PSF-Zero. Every other circuit is handed to Qiskit's
>   level 3 unchanged, with seed 0 by default; on a Target with IBM's fractional gates its rzz angles are folded into
>   the executable range, as IBM's backend would do. Without the device, nothing changes.
> - **Built by formula, inside the scope:**
>   - the time-aware order of the same gates (item 83), which raised the fidelity on real devices 8 of 8 times;
>   - chains, lattices and embeddable graphs as brickwork or by edge colouring (items 87, 91);
>   - IBM's native rzz / rx with signs in a Pauli frame (item 88; item 86's fractional re-synthesis stays off by
>     default);
>   - a diagonal Clifford before the final measurements dropped (item 89);
>   - failed qubits cut out first (item 85).
> - **Every output checked exact** where it can be (items 93, 95: outcome distribution within 1e-6 up to 20 active
>   qubits). An inexact one is never returned silently.
> - **The AI front end a18:** ring classifiers built by formula (a16-a17), with no search (item 32); everything else
>   handed on (item 33). On 64 classifier cells it is 0.54 x Qiskit level 3's duration, shorter on 64 of 64.
> - **C57-VAL** (28 tests, 5 devices, offline):
>   - in scope, 0.20-0.23 x Qiskit level 3's duration;
>   - out of scope, Qiskit's own output (folded where IBM requires it);
>   - every output native, exact and on no failed element.
> - **What it costs:** out of scope, the outputs are 37-49% longer than the previous prototypes' own choices offline.
>   That is on purpose: those choices lost on real devices (Addenda 487-488).
> - **Not established:** a real-device test of this release as a whole. The order (item 83) and rzz have run on
>   devices; the ring classifiers have not. 2026-10-11.4 is kept in
>   [`patches/psf_compile_release_2026-10-11.4/`](patches/psf_compile_release_2026-10-11.4/), and a12 as
>   `benchmarks/psf_ai_compile_a12.py`.

> **In 2026-10-11.4: the same input gives the same output, and circuits the earlier releases refused
> compile.** Items 74-76 (prototype c43), accepted in C43-VAL on circuits not used before (Part 11, Addenda 452-453):
>
> - **What changes.** Qiskit's VF2PostLayout, which returned a different layout from call to call with a fixed seed,
>   is replaced by a deterministic re-placement scored on the device's own errors (item 75). With the device, that
>   also ends a fault of the earlier releases: Qiskit's pass moved measured-only qubits onto failed qubits, and the
>   release then refused the circuit (Bernstein-Vazirani and random circuits on four of six fake devices). Without
>   the device, nothing is re-placed. Item 73 also reads circuits that end with a qubit permutation, such as QFT with
>   its swaps (item 76). An output that uses a near-failed element (error 0.1-0.5) is compared with one that avoids
>   the elements it used (item 74).
> - **C43-VAL** (210 jobs on six devices; QFT, QAOA, quantum volume, random, Bernstein-Vazirani):
>   - the same output on every second call (195 of 195; 2026-10-11.3: 191); no output on a failed element;
>   - 15 test-device pairs that 2026-10-11.3 refused now compile, at Qiskit level 3's ESP or close to it;
>   - ESP 1.10 times 2026-10-11.3's and 1.18 times Qiskit level 3's on the runnable tests; QFT 1.15 times;
>   - 1.5 times 2026-10-11.3's compile time.
> - **Where it loses.** The largest random circuit tested (48 qubits) on three devices: 0.48-0.90 of 2026-10-11.3's
>   ESP. Item 74 gains little (+4%). Outputs differ from 2026-10-11.3's by value.

> **In 2026-10-11.3: every circuit that reads as Pauli rotations is also tried by construction, and the
> better on the device is kept.** Items 69-73 (prototype c41), accepted in three tests on circuits they were not
> developed on (C39-VAL, C40-VAL, C41-VAL; Parts 10-11, Addenda 447-451):
>
> - **What changes.** With `backend=` (or `target=`), an input that can be read as a sequence of Pauli rotations is
>   compiled twice, by STEINER6's construction and by the usual path, and the output with the higher ESP on the
>   device's error table is returned (item 69). Read are: Pauli evolutions whose terms commute or not, in any product
>   formula (items 71-72); QAOA circuits and rotation gates (item 72); and circuits written as gates, such as QASM
>   files with cx-rz-cx and basis changes, by carrying their Clifford gates to the end (item 73; the output equals
>   the input up to a global phase). Every other input is compiled exactly as by 2026-10-11.2.
> - **The tests** (six fake devices; generated circuits and 89 Benchpress tests):
>   - no output on a failed element, never below 2026-10-11.2 on any test, unread inputs unchanged;
>   - ESP compared with 2026-10-11.1: 1.45 times on commuting Hamiltonians (the ts225 loss of 2026-10-11.2 is gone),
>     1.19 on QAOA, 1.08 on non-commuting Hamiltonians, 1.20 on Benchpress QASM files read as gates (1.29 on QAOA
>     QASM; gate-level UCCSD with less than half the two-qubit gates);
>   - about 4 times 2026-10-11.1's compile time on the inputs read (median under 1 s; up to 48 s).
> - **Where it gains little.** One Trotter step of a non-commuting Hamiltonian (+3%), small MaxCut instances, Toffoli
>   circuits (+7%). Pass `steiner=False` to keep 2026-10-11.2's usual path.

> **In 2026-10-11.2: a Hamiltonian whose terms all commute is built on the device by construction.** Item
> 68 (candidate c38), accepted in a test on circuits and devices it was not developed on (C38-VAL; Part 10, Addenda
> 444-446):
>
> - **What changes.** With `backend=` (or `target=`), one PauliEvolutionGate whose terms all commute is no longer
>   routed: STEINER6 builds it term by term on the device (Addendum 444). It chooses the order of the terms, moves
>   qubits with SWAPs planned from the terms still to come (a SWAP joining a ZZ term costs one CNOT), and picks
>   placements and SWAPs by the device's errors. The output carries its layout. Every other input is compiled
>   exactly as by 2026-10-11.1.
> - **C38-VAL** (232 jobs on six devices, two of them never used by STEINER6; generated MaxCut-type and 3-SAT-type
>   Hamiltonians and new HamLib instances):
>   - no output on a failed element; non-commuting inputs unchanged (107 of 107);
>   - on the runnable commuting tests, ESP 1.39 times 2026-10-11.1's and 1.36 times Qiskit level 3's (1.22-1.56 by
>     device), with 10% fewer two-qubit gates;
>   - 4-10 times 2026-10-11.1's compile time on those inputs.
> - **Where it loses.** Dense two-local diagonal Hamiltonians, such as tsp in unary encoding, are compiled worse
>   than before: 0.87 of 2026-10-11.1's ESP on HamLib's ts225, with 13% more two-qubit gates. Small MaxCut instances
>   (16 qubits or fewer) are even. Pass `steiner=False` to keep 2026-10-11.1's path.

> **In 2026-10-11.1: given the device, the call also routes at Qiskit's level 3.** Item 64 (candidate c36),
> accepted in a pre-registered test on two devices it was not found on (C36-VAL; Part 10, Addenda 439-442):
>
> - **With `backend=` (or `target=`)** the call routes with Qiskit's level-3 settings instead of level 1. Without the
>   device nothing changes.
> - **C36-VAL** (106 Benchpress tests on FakeMarrakesh and FakeFez):
>   - no output on a failed element (Qiskit level 2: 60; level 3: 57);
>   - ESP 1.06 and 1.08 times 2026-10-10.3's;
>   - 0.976 and 0.963 of Qiskit level 3's, on the tests where level 3's output is clean;
>   - two-qubit gates level with Qiskit level 2's;
>   - 1.26-1.34 times 2026-10-10.3's compile time, about 0.2 s per test as a median.
> - **Still behind Qiskit level 3** by 2-4% of ESP where its output is clean. At level 3 PSF-Zero's two-qubit count is
>   about level 3's: what it adds is the avoidance of failed elements and the placement by the errors.

> **In 2026-10-10.3: give the call the device, and it never returns a circuit on a failed element.** Items 58-63
> (candidates c31-c35), accepted together in a pre-registered test (C35-VAL; Part 10, Addenda 429-438):
>
> - **Why:** real devices' Targets keep failed couplers and qubits at error 1. The live ibm_kingston Target did on
>   2026-09-28, and every PSF-Zero output of Addendum 246 used them (Addendum 434). A call that is not given the
>   device cannot avoid them.
> - **`backend=` (item 58):** `compile_for_hardware(qc, backend=backend)` takes the Target, coupling map and basis
>   from the device. Given the device, the call places by the errors (item 60) and avoids failed elements. It raises
>   `FailedElementsError` rather than return a circuit that cannot avoid them (`on_failed_elements="keep"` returns it
>   with a warning). Without the device it warns once per process.
> - **C35-VAL** (106 Benchpress tests on FakeTorino and FakeKingston, Addenda 436-437):
>   - with `backend=`, 0 of 212 outputs on a failed element (Qiskit level 2: 44; level 3: 34);
>   - estimated success probability (ESP) 1.002 and 0.964 of 2026-10-10.2's recommended call, at 0.41-0.43 of its
>     time;
>   - ESP 1.16 and 1.15 times Qiskit level 2's;
>   - without the device, 2026-10-10.2's two-qubit counts on 212 of 212.
> - **Still behind Qiskit level 3:** ESP 0.957 and 0.899 of its on the tests where a result survives, at 1.4-1.6 times
>   its compile time.
> - **By PSF-Zero's own `hybrid_cost`** (which counts decoherence while qubits wait), the full-option call was worse
>   than an older call's on 1 of 20 small test cells (2.4%, from item 63's single-qubit gates) and better on 2
>   (Addendum 438).
> - **Speed (items 61-63):** with the device, one compile on the pruned map. The SWAP absorption is synthesised by
>   Qiskit's Rust decomposer, and rustworkx does the layout search's matching checks.

> **In 2026-10-10.2: the same circuits, and the recommended call's estimates and checks run in Rust.** One
> change (item 57b), tested for identity (Part 10, Addenda 425-428):
>
> - **Item 57b:** the per-gate loops of the recommended call's estimates and exactness checks run in an optional
>   Rust module, `psf_zero_core57` (see [Install](#install)); without it the Python code runs, with the same values
>   as before. On 152 Benchpress development tests (C30-ID, Addenda 426-427) it returned 2026-10-10.1's circuit, by
>   value, on all 184 test-calls where 2026-10-10.1 repeats itself. Where the recommended call makes estimates, it
>   took 0.656 of the time (geometric mean of 14 test-calls); on `enc_gray_dvalues_4-4-4-4-4-4-4`, 39 s instead of
>   1,546 s.
> - **Not changed:** the default call, the AI front end (a12), the layout search, the core `psf_zero_core`.

> **In 2026-10-10.1: the same circuits, and the recommended call no longer estimates candidates it can never
> use.** Two changes (items 53 and 56), each tested for identity (Part 10, Addenda 413-424):
>
> - **Item 56:** the recommended call replaces its circuit with a candidate (re-synthesis, a floor-aware compile,
>   Qiskit level 3) only if an exactness check confirms it, and that check cannot be made on circuits of more than
>   200,000 instructions or 16 qubits. It now decides this from counts before estimating the candidate, and skips
>   estimates that could never change the result. On 152 Benchpress development tests (C29-ID, Addenda 422-423) it
>   returned 2026-10-07.1's circuit, by value, on all 183 test-calls where 2026-10-07.1 repeats itself; on `hwb10`
>   it finished in 82 s where 2026-10-07.1 did not finish in an hour.
> - **Item 53:** the estimates and checks keep standard gates' matrices and embed one-qubit gates without `np.kron`:
>   the same values (C26-ID, Addenda 413-414), about 2% less time for the recommended call.
> - **Not changed:** the default call, the AI front end (a12), the layout search and the core.

> **In 2026-10-07.1: on general circuits the default call is about level with Qiskit level 2 (1.03 times its
> two-qubit gates, where 2026-10-06.4 needed 1.33), and the recommended call is about four to eight times faster at
> 16 qubits.** Five changes (items 46-50), each accepted after its own pre-registered test and released together
> (Part 10, Addenda 383-397):
>
> - **Default call (items 48 and 50):** instructions on three or more qubits (`ccx`, a `PauliEvolutionGate`, a QFT
>   block) are expanded before PSF-Zero's own pipeline; and where Qiskit's commutative cancellation removes two-qubit
>   gates from the input, the call compiles both and keeps the circuit with fewer. On 139 Benchpress tests the
>   two-qubit count went from 1.33 to 1.03 times Qiskit level 2's (geometric means; BP-MOCK, BP-MOCK2, CANCEL and
>   the exploratory Addendum 396). Benchpress's BV-like test goes from 392 two-qubit gates to none, a 160-qubit QFT
>   on heavy-hex from 21,361 to 15,793.
> - **Recommended call (items 46, 47 and 49):** its state-vector checks apply fewer, larger matrices, are made only
>   where they can change the output, and its estimates follow single-qubit gates on 2x2 reduced states. At 16
>   qubits 0.36-0.49 and then 0.37-0.51 of the time (FUSE and TRACK, 360 circuits each on 6 devices), about
>   0.13-0.24 of 2026-10-06.4's together (the product of the two tests' medians per device). The same circuits: in
>   FUSE 359 of 360 as 2026-10-06.4's, the other a near-tie of 1.35e-16 between two estimates, now kept as a tie;
>   in TRACK 360 of 360 as FUSE's candidate.
> - **Unchanged:** on inputs with no instruction on three or more qubits and no two-qubit gate that cancels, both
>   calls return 2026-10-06.4's circuit, apart from near-ties like the one above and inputs on which Qiskit's own
>   level 1 does not reproduce itself (see [Where it is weaker](#where-it-is-weaker)).
> - **Cost:** where cancellation removes something the default call compiles twice (1.5 times the time as a median,
>   up to 4 times).

> **In 2026-10-06.1-.4 and a11/a12: readout is counted, ecr devices are fixed, the AI front end is faster,
> the target-aware calls no longer fail on full-device circuits or on wide instructions, and they are about four
> times faster above 16 qubits.**
>
> - **2026-10-06.4 (item 45):** above 16 logical qubits the recommended call no longer builds Qiskit level 3 and
>   the floor candidate, which its equivalence check (item 39) cannot check and always refused. Same circuit,
>   about a quarter of the time; at full occupancy of a 27-qubit device 0.06-0.16 s instead of 8.6-9.5 s.
>   Pre-registered test SKIP (Addenda 379-380): identical on 294 of 294 circuits on six devices.
> - **2026-10-06.3 (item 44):** the recommended call and the AI front end could abort the Python process (a Rust
>   allocation failure) or run for minutes on a circuit holding one wide instruction, such as Benchpress's HamLib
>   inputs (one `PauliEvolutionGate` on all qubits): item 39's equivalence check turned it into a matrix. It is now
>   expanded through its definition. Nothing else changes. Found by a probe on 12 Benchpress tests (Addendum 377),
>   in which the default call also used more two-qubit gates than Qiskit level 2 on 10 of 12 (geometric mean
>   1.53x; level only on Quantum Volume); closing that gap is open work.
> - **2026-10-06.2 (item 43):** on a circuit that needs the whole device, the recommended call and the AI front end
>   raised an exception when the device reports failed elements; now they return the circuit and warn that it uses
>   them. Nothing else changes. Found and tested in a PennyLane loop on FakeKingston (PL-REDO, Addenda 372-373): the
>   default call took 0.048 s per compile on the full 156 qubits, swap-free, about 320 times faster than Qiskit
>   level 3; it does not look at the device's failed elements, which the target-aware calls avoid when there is room
>   (0.4-0.6 s with 16 spare qubits).
> - **Readout:** compile a circuit that will be sampled **with** its final measurements. The choice among candidates
>   and the AI front end's estimate now include each measured qubit's readout error; without measurements nothing
>   changes.
> - **ecr devices:** a9 could return ECR gates in a direction the device does not provide; a11 never does.
> - **Faster AI front end:** a12 returns a11's circuits in 0.54-0.89 times the time (0.54-0.60 on cz devices); a
>   pre-registered test found them identical on 768 of 768 circuits (SPEED, Addenda 364-365).
> - **Opt-in `candidate_score="kraus"`:** an estimate exact to first order for the simulator's noise model. It has no
>   readout term, so the recommended call stays `"hybrid"`.
> - **Pre-registered tests** (fake devices; Addenda 358-361):
>   - RECR (104 sampled circuits on each of 6 devices): compiling with the measurements cut the measured qubits'
>     readout error to 0.40 and 0.18 times on the cz devices (2026-10-05.1 could already do this); the release's
>     readout term adds a small gain on top (0.96-1.00 times); a11's classical infidelity was 0.46 and 0.16 times
>     a9's there; a9 returned 622 and 860 gates in an unsupported direction on two ecr devices, a11 none; 10 of 10
>     predictions confirmed.
>   - KRAUS (1,506 circuits on each of 9 devices): `kraus` at least as good as `hybrid` on every device, 6.3% better on
>     the one known weak case; 3-5% more compile time; 7 of 7 confirmed. It reproduces the simulator's own noise
>     model, so this test favours it by construction.

> **Correctness fix: if you use 2026-10-03.1, .2, .3 or 2026-10-04.1, update.** On cx devices those releases could
> return a circuit that is NOT equivalent to the input, when called with `final_resynthesis` or `compare_level3`. The
> same held for the AI front end (a7, a8) given a target.
>
> - **When:** the input contains two-qubit unitaries near the boundary of Qiskit issue #17057, for example explicit
>   `unitary` gates from numerical optimisation or written by a language model, or Trotter steps with very small
>   angles.
> - **Fix:** since 2026-10-05.1 and a9, every circuit that Qiskit makes as a whole is checked before it is returned.
>   A circuit that fails the check, or cannot be checked, is replaced by PSF-Zero's own, guarded circuit.
> - **Pre-registered test** (Addenda 342-343; 6 devices, 768 near-boundary and control circuits, plus 1,506 ordinary
>   circuits on each of 9 devices):
>   - 2026-10-04.1 was wrong on 101 circuits on the cx devices (infidelity up to 0.34), a8 on 108, Qiskit level 3
>     alone on 112;
>   - 2026-10-05.1 and a9 were wrong on none;
>   - on the ordinary circuits they returned exactly the circuits of 2026-10-04.1 and a8;
>   - every refusal checked was of a wrong circuit (an exploratory check after the run, on FakeAuckland);
>   - cost: median compile time 1.27-1.42 times 2026-10-04.1 on the nine devices (35-55 ms more).
> - cz devices were not affected.
> - Whether Qiskit's failure appears depends on floating-point rounding: it appeared in Linux environments
>   (WSL2 and a Linux sandbox), and not in a Windows environment tested on 2026-10-06 (Addendum 357). The fix
>   protects either way.

## Quick start

**Two-qubit synthesis** (fast, exact, deterministic):

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
import importlib.util  # compile() is in the previous release, kept in patches/
spec = importlib.util.spec_from_file_location("psf_compile_2026_10_14_1",
                                              "patches/psf_compile_release_2026-10-14.1/psf_compile.py")
rel = importlib.util.module_from_spec(spec); spec.loader.exec_module(rel)
psf_compile = rel.compile

qc = QuantumCircuit(2)
qc.append(UnitaryGate(random_unitary(4)), [0, 1])
optimized = psf_compile(qc)          # add verify=False for the fastest path
```

**For a device** (the recommended call since 2026-10-10.3, the same on cx and cz devices): give it the device.
Since 2026-10-11.1 it then also routes at Qiskit's level 3 (item 64; C36-VAL, Addenda 440-441), and since
2026-10-11.2 a Pauli evolution whose terms all commute is built by STEINER6 (item 68; C38-VAL, Addendum 445).
Since 2026-10-11.3 every input that reads as Pauli rotations, circuits written as gates included, is also built
by STEINER6 and the better output on the device is kept (items 69-73; Addenda 447-451). Since 2026-10-11.4 the
same input gives the same output, and circuits the earlier releases refused compile (items 74-76; Addenda
452-453).

```python
from qiskit_ibm_runtime.fake_provider import FakeTorino
from psf_compile import compile_for_hardware

backend = FakeTorino()
out = compile_for_hardware(qc, backend=backend)   # 2026-10-16.1: OutOfScopeError if qc does not embed in the device
```

Since 2026-10-16.1 the call reads only `backend=`/`target=`, `prune_max_error=` and `seed_transpiler=`; the options
below were 2026-10-14.1's and are ignored.

The earlier recommended options (`placement_refine=True, final_resynthesis="select", compare_level3=True,
compare_floor=True, candidate_score="hybrid"`) still work with `backend=`. In C35-VAL they gave about 1-2% more ESP at
about twice the time (Addendum 437).

If the circuit will be sampled, compile it **with** its final measurements: since 2026-10-06.1 the placement and the
choice among candidates then count readout error (Addenda 358, 360).

**Always give the call the device** (`backend=`, or `target=`).

- Real devices' Targets keep failed couplers and qubits at error 1, and an output that uses one has an estimated
  success probability of 0. The live ibm_kingston Target did on 2026-09-28 (Addendum 434).
- A call given only a coupling map and a basis does not read the errors. It cannot avoid failed elements, and warns
  once. In C35-VAL 63 of 106 FakeTorino outputs of such a call used one; with `backend=`, none (Addendum 437).
- In CALSPLIT a classifier compiled without the device reached 0.77 accuracy on FakeTorino even at 1,023 shots,
  against 0.94 with any call given the Target (Addendum 403).
- Qiskit level 2 and level 3 can use failed elements too (Addendum 393).

## Install

```bash
git clone https://github.com/TN-Holdings-LLC/psf-zero.git
cd psf-zero
pip install -e .            # the Python package (psf_compile.py), numpy, scipy, networkx, qiskit==2.5.2
maturin develop --release   # the Rust core (src/lib.rs, psf_zero_core) -- must run LAST
(cd patches/psf_zero_core57_2026-10-10 && maturin develop --release)   # optional: item 57b's Rust loops
python benchmarks/check_core_build.py   # prints RESULT: OK if the core has every function the Python code calls
```

The order matters. With both `Cargo.toml` and `pyproject.toml` in one folder, running `maturin develop` first builds
the wrong package, and running `pip install -e .` after it undoes the core. The second step needs a Rust toolchain
([rustup.rs](https://rustup.rs)) and `maturin` (`pip install maturin`).

Source: [`psf_compile.py`](psf_compile.py) is the compiler, and the one place it lives (its `VERSION:` line names the
revision) · [`lib.rs`](lib.rs) is the Rust core · [`benchmarks/psf_smart_layout.py`](benchmarks/psf_smart_layout.py)
is the layout search · [`benchmarks/psf_ai_compile.py`](benchmarks/psf_ai_compile.py) is the AI front end for
model-written circuits.

## What it does well, and what it costs

| | `compile()` | `compile_for_hardware()`, recommended call | Qiskit `optimization_level=3` |
| :--- | :--- | :--- | :--- |
| Compile time | 2.5-5x faster than Qiskit L3 on two-qubit-block circuits (15-1000 qubits) | median about 0.2 s on 4-8-qubit circuits, about 13x Qiskit L3 with the Target | reference |
| Output | same two-qubit count as Qiskit L3 at 156 qubits; deterministic | simulated infidelity 0.959-0.991 of Qiskit L3 with the Target, on 9 fake devices | reference |
| Against TKET | 150-270x faster; depth 9 against TKET's 7 | not compared | |

- **The depth gap to TKET is the price of not searching.** We know no way to close it without giving up determinism.
- **`compile()` only helps on circuits with deep same-pair two-qubit chains** (Trotter steps, layered entanglers,
  SU(4) blocks). On a generic `random_circuit()` it reports `0/0 blocks` and passes the circuit through.
- **The device numbers are noisy simulations** with Qiskit Aer and the fake devices' published calibration. The
  noise estimates PSF-Zero chooses by share the simulator's physics, so part of that advantage is built in. Hardware
  is the real test, and it has not been run for the device layer.

### Where it is weaker

**General circuits (Benchpress).** BP-FINAL (Addenda 407-408) compiled every published Benchpress transpilation
test that no earlier test of this project had used: 880 tests (QASMBench small, medium and large and HamLib on
all-to-all, square, heavy-hex and linear maps; HamLib and Feynman on FakeTorino). Each was compiled with the default
call of `compile_for_hardware()` and with Qiskit level 2 as Benchpress calls it. The predictions were locked before
any of the 880 was compiled, and all six were confirmed. Every PSF-Zero output passed Benchpress's validator, and
every one that could be checked (491) implements its input.

| two-qubit gates / Qiskit level 2 (geometric mean) | |
| :--- | :--- |
| **2026-10-07.1 on 877 unseen tests (BP-FINAL)** | **1.046** (95% 1.037-1.055): fewer on 117, as many on 314, more on 446 (by over 10% on 131) |
| per family | QASMBench 1.020 (404 tests), HamLib on the four maps 1.069 (367), HamLib on FakeTorino 1.090 (67), Feynman 1.022 (39) |
| 2026-10-07.1 on the 139 development tests (BP-MOCK, BP-MOCK2, CANCEL; Addendum 396) | 1.03 |
| 2026-10-06.4 on the same 139 | 1.33 |

- **It is about level, not better**, and a little worse on unseen tests than on the tests its changes were written
  from (1.046 against 1.03). Hamiltonian simulation is where it is weakest: 1.06-1.09 per HamLib stratum.
- **It is slower:** a median 3.5 times Qiskit level 2's compile time (geometric mean 3.0; BP-FINAL, 12 jobs at a
  time on one machine).
- **Not reproducible where Qiskit is not.** The default call lays out and routes with Qiskit level 1. On some inputs
  (`bv_n30` on square, `bv_n140` on linear) Qiskit's own `transpile(optimization_level=1, seed_transpiler=0)` returns
  a different circuit in each process, with the same two-qubit count, and so does PSF-Zero's default call
  (Addendum 393). `seed_transpiler` does not fix it. In BP-FINAL a second run of the default call returned a
  different circuit on 83 of 877 tests, never with a different two-qubit count.
- **Failed couplers:** on FakeTorino the default call, which does not read the target, placed two-qubit gates on
  couplers or qubits the device reports as failed (error 0.5 or more) in 62 of 105 tests (40,801 gates on `hwb11`);
  Qiskit level 2, given the device, in 25; the recommended call in none, at 1.035 times Qiskit level 2's two-qubit
  count (BP-FINAL). Give the call the device (see [Quick start](#quick-start)): with `backend=`, 2026-10-10.3 used
  no failed element in C35-VAL (Addendum 437).
- Not covered: Benchpress's other test groups, Benchpress's own gym (BP-FINAL used this project's harness with
  Benchpress's builders, backends and validator), depth as a target, hardware. Three tests timed out (1,500 s) for
  the default call or Qiskit and are left out of the ratios.

**With a calibration that is not the device's (CALSPLIT and MARGIN, Addenda 402-406).** DEPTH-R's classifiers (4
and 6 qubits, 1-16 layers; Addenda 399-400) were compiled with stale Targets (gate errors off by about 30%, T1 and T2
by about 20%, three draws per device) and scored with the device's true noise, on FakeAuckland and FakeTorino;
MARGIN repeated this on new data.

- **Reading a calibration that is 30% wrong still beats not reading one.** Every target-aware call kept more margin
  and flipped fewer answers than the calls that ignore the target. At 15 shots the recommended call was 1.7 points
  (FakeAuckland) and 4.9 points (FakeTorino) of accuracy ahead of the better of them; at 1,023 shots 0.1 and 0.4.
- **The recommended call's estimate-driven choices pay off with a fresh calibration; with a stale one it depends on
  the data.** Over the guarded call (`target` and `placement_refine` only) they gained +0.0056 and +0.0058 of
  classification margin on FakeAuckland with the true calibration (Addenda 400, 406). With stale calibrations they
  lost 0.0041 in CALSPLIT (0.0142 in one draw; 0.3 points of accuracy at 15 shots) and gained 0.0011 on new data in
  MARGIN. On FakeTorino the lead held in both (+0.0194, +0.0246): it comes from routing, not from the estimate. Given
  the same stale Target it stayed level with Qiskit level 3 (+0.0049 and -0.0098).
- **A variant that switches only for an estimated gain above 5% (candidate c24, item 51) is not recommended.** In
  MARGIN it never beat the release: it gave up 0.0034 of the gain on FakeAuckland with a fresh calibration and gained
  nothing with stale ones (Addendum 406).
- **The default call on a device with failed couplers is a real hazard.** On FakeTorino it placed 21,204 two-qubit
  gates on failed couplers and lost 17 points of accuracy (see [Quick start](#quick-start)).
- Not covered: real calibration drift (this was a perturbation model), hardware, other tasks, stale readout errors.

**The recommended call at 16 qubits.** Up to 16 logical qubits it builds Qiskit level 3 and a second candidate and
checks them by state-vector simulation. 2026-10-07.1 does this in about 0.13-0.24 of 2026-10-06.4's time (see
Current version); with 2026-10-06.4 it took a median of 31 s for Hamiltonians of 48 Pauli terms and 20 s for QFT at
16 qubits (SKIP, Addendum 380), against about 0.3 s at 17 qubits.

## Results in brief

**Compile time against Qiskit `optimization_level=3`** (`compile()`, dense two-qubit-block circuits, `verify=False`,
10 seeds per point):

| | 15q | 50q | 100q | 156q | 300q | 500q | 1000q |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qiskit ÷ PSF-Zero** | **5.15x** | **3.98x** | **3.45x** | **2.86x** | **3.02x** | **2.78x** | **2.54x** |

**Against TKET** (`FullPeepholeOptimise`, same family):

| | 10q | 20q | 40q | 80q | 160q |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Speed-up** | **152x** | **237x** | **262x** | **258x** | **272x** |
| Depth (TKET / PSF-Zero) | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 | 7 / 9 |

**For a device** (pre-registered tests on 1,506 new circuits per device, 9 fake devices: four cx, five cz):

- Release 2026-10-04.1 against Qiskit level 3 with the Target: 0.959-0.991 (Addendum 337). Release 2026-10-05.1
  returns the same circuits on those circuits, on all nine devices (Addendum 343), and so does 2026-10-06.1 (it differs
  only on circuits compiled with measurements).
- Sampled circuits compiled with their measurements (6 devices): release 2026-10-06.1 at 0.66-1.01 times Qiskit level 3's
  classical infidelity, a11 at 0.63-0.98 (Addendum 360).
- With a stale calibration (errors off by 30%, T1/T2 by 20%) the previous release stayed ahead of level 3
  (Addendum 335). On a classification task 2026-10-07.1's recommended call stayed level with level 3 given the same
  stale Target; against the guarded call on FakeAuckland it lost 0.004 in CALSPLIT and gained 0.001 on new data in
  MARGIN (Addenda 403, 406; see [Where it is weaker](#where-it-is-weaker)).
- At 8-10 logical qubits it stayed ahead too (0.951-0.998).
- No failed coupler or qubit was used in any of these tests. (On an input that is one instruction over many
  qubits, 2026-10-06.4's recommended call could still use them, with a warning; 2026-10-07.1 avoids them where
  the device has room: Addendum 398.)

**On real IBM hardware** (15 qubits, 10 jobs on `ibm_marrakesh` and `ibm_fez`, `compile()` path): fidelity was
indistinguishable from Qiskit L3 (0.0919 ± 0.0016 against 0.0925 ± 0.0016), and compile time 14-16x shorter.
The device layer has since run in four pre-registered tests on three devices (2026-10-12 to 2026-10-14). See
[Scope](#scope-where-psf-zero-competes-declared-2026-10-14) and Addenda 454-488.

**Numerical accuracy of the core:** worst infidelity 1.1e-15 over 500 Haar-random SU(4), 2.6e-14 near CNOT, no
fallbacks. Repeated recompilation drifts linearly (6.1e-11 after 1,000 laps). See
[`core-verification.md`](docs/findings/core-verification.md) and [`docs/RELEASES.md`](docs/RELEASES.md).

**Correctness checks.** Qiskit's CX-basis synthesis returns wrong circuits for two-qubit unitaries near a boundary
(reported as [Qiskit issue #17057](https://github.com/Qiskit/qiskit/issues/17057)). PSF-Zero checks every block it
takes from that decomposer (since 2026-09-26.4), and since 2026-10-05.1 every circuit Qiskit makes as a whole. The
fix above is for releases 2026-10-03.1 to 2026-10-04.1, which used such circuits unchecked.

The full tables, figures and caveats are in [`docs/findings/`](docs/findings/) and the README snapshot
[`docs/README_2026-10-05_before_restructure.md`](docs/README_2026-10-05_before_restructure.md).

## A finding that is not about PSF-Zero: the coupling-map cliff

**Qiskit's `optimization_level` 2 and 3 slow down by 40x-420x when a circuit nearly fills the coupling map.** On one
42-qubit grid, a 42-qubit circuit takes 6.8 s at `opt=3`, while a 38-qubit circuit takes 28 ms.

- **What we found:**
  - Reproduced in three environments.
  - Present in every Qiskit release from 1.4.6 through 2.5.2.
  - Also reachable through PennyLane.
  - Also present on IBM's square-lattice model (FakeNighthawk, 12.9 s at spare 0) and, with circuits that use 3-qubit
    paths, on heavy-hex devices.
- **The cost** sits in two VF2-family searches, `VF2Layout` and `VF2PostLayout`, which report "no solution" on
  instances where a zero-SWAP layout provably exists.
- **PSF-Zero's layout search** finds those layouts in milliseconds.
- **How often:** the cliff needs a circuit that is both fully embeddable and fully saturated. Generic dense circuits
  do not land on it (Addenda 135-136).

Full account: [`spare-qubit-cliff.md`](docs/findings/spare-qubit-cliff.md) (summary) and Paper 1.

## Papers

- [**Paper 1 — Ordering Sensitivity in Subgraph-Isomorphism Layout Search**](docs/papers/vf2_cliff_paper.pdf)
  ([DOI: 10.5281/zenodo.22977930](https://doi.org/10.5281/zenodo.22977930)). It characterizes the failure region of
  Qiskit's `VF2Layout`.
- [**Paper 2 — PSF-Zero: An Analytic Two-Qubit Gate Synthesizer Combined with a Verified, Ordering-Aware Layout
  Search**](docs/papers/psf_zero_paper.pdf) ([DOI: 10.5281/zenodo.22978090](https://doi.org/10.5281/zenodo.22978090)).
- [**Technical overview slides**](docs/papers/psf_zero_technical_overview.pptx) (8 slides).

**Versions.** Both papers are at version 2 (2026-09-26). Version 1 remains citable:
[10.5281/zenodo.22869976](https://doi.org/10.5281/zenodo.22869976) and
[10.5281/zenodo.22870141](https://doi.org/10.5281/zenodo.22870141). The device layer of `compile_for_hardware()`
(October 2026) came after both papers.

## Known limits and open questions

- **Hardware:** the device layer ran on real devices in four pre-registered tests (HW1, HW2, TIME1, FRAC1;
  2026-10-12 to 2026-10-14). The noise estimate (ESP) that the release chooses by misranked the pairs on HW1. It
  ranked them reasonably on HW2, but no predictor held on both devices (Addenda 455, 461, 476). Choosing by duration
  was not confirmed either (TIME1). Building by formula and the time-aware order were (HW2 part S, TIME1
  part T1). See [Scope](#scope-where-psf-zero-competes-declared-2026-10-14).
- **ecr devices:** the release has not been tested on them in a pre-registered test. A workplace exploration found it
  working there. It also found the AI front end defect noted above.
- **Above 16 touched qubits** the noise estimates and the equivalence checks are not made, and the release keeps its
  own circuit. Since 2026-10-06.4 the alternatives that would be refused there are not built at all (item 45).
- **Tolerance of the equivalence check:** a Qiskit-made circuit is accepted up to a state infidelity of 1e-6. On
  near-boundary Trotter circuits the accepted ones were off by up to 5.8e-8, where PSF-Zero's own path is exact to
  1e-14 (Addendum 343).
- **Benchpress:** BP-FINAL covered every published transpilation test not used during development (880; see
  [Where it is weaker](#where-it-is-weaker)). No unseen Benchpress transpilation test is left for testing later
  changes, and there is no Benchpress gym for PSF-Zero yet. Qiskit's own level 1, which the default call
  routes with, is not reproducible on some inputs (Addendum 393); whether to report that upstream is open.
- **Two upstream findings:**
  - Qiskit #17057 (CX-basis synthesis) is open.
  - A qiskit-aer `save_expectation_value` defect with qubit truncation was found on 2026-10-05 and is not yet reported.
- The longer list of open questions as of 2026-10-05 is in the README snapshot
  [`docs/README_2026-10-05_before_restructure.md`](docs/README_2026-10-05_before_restructure.md#open-questions).

## How these numbers were produced

- **Timing:** warm-up outside the timer for every engine; repeated timed calls; several seeds; `spawn`;
  `seed_transpiler` pinned.
- **Every timing comes with an equivalence check.** Medians, not means; ranges, not peaks.
- **Device results** come from pre-registered tests. Predictions are locked by a git commit before the scored run,
  re-checked by an independent script, and adopted only by the owner's decision.
- **Three earlier headline claims were retracted** after re-measurement. The record, including every retraction, is
  kept verbatim in [`docs/log/`](docs/log/) and [`docs/findings/`](docs/findings/).
- The conventions are in [`record-keeping.md`](record-keeping.md).

## Where everything is

| | |
| :--- | :--- |
| [`docs/RELEASES.md`](docs/RELEASES.md) | Every release, update and correctness notice, newest first |
| [`docs/findings/spare-qubit-cliff-combined-450.md`](docs/findings/spare-qubit-cliff-combined-450.md) | Part 11 of the full record (Addenda 450 on: releases 2026-10-11.4 and 2026-10-14.1, the TOQB loop runs, the search removal); earlier parts are linked from it |
| [`docs/findings/spare-qubit-cliff-combined-383.md`](docs/findings/spare-qubit-cliff-combined-383.md) | Part 10 (Addenda 383-449: the Benchpress tests and release 2026-10-07.1); earlier parts are linked from it |
| [`docs/findings/spare-qubit-cliff-combined-248.md`](docs/findings/spare-qubit-cliff-combined-248.md) | Part 9 (Addenda 248-382) |
| [`docs/findings/compile-time.md`](docs/findings/compile-time.md) | The compile-time arc: three retractions, the `verify` split, and what survives |
| [`docs/findings/spare-qubit-cliff.md`](docs/findings/spare-qubit-cliff.md) | The Qiskit coupling-map cliff |
| [`docs/findings/core-verification.md`](docs/findings/core-verification.md) | The core's infidelity harness |
| [`docs/findings/entangling-basis.md`](docs/findings/entangling-basis.md) | Why `entangling_basis="cx"` exists |
| [`docs/findings/real-hardware.md`](docs/findings/real-hardware.md) | IBM hardware runs and job IDs |
| [`docs/log/`](docs/log/README.md) | The unedited early log, with a chronology of every claim this project got wrong |
| [`data/`](data/), [`benchmarks/`](benchmarks/), [`patches/`](patches/) | Raw data, harnesses, and the candidates each test locked |

## Working with us

This is AGPL-licensed research code, not a supported product. If you're evaluating PSF-Zero against your own circuits
and want a second opinion before investing further, send a representative circuit (or a sanitized equivalent) to
`love.os.architect@proton.me` (under NDA first, if the circuit itself is sensitive), and we'll run it and report
results directly. No commitment is implied on either side; the goal at that stage is reproducing a result, not a sales
conversation.

## Citation

```bibtex
@software{psf_zero_2026,
  author = {The Architect},
  title  = {PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis},
  year   = {2026},
  url    = {https://github.com/TN-Holdings-LLC/psf-zero},
  license = {AGPL-3.0}
}

@misc{vf2_cliff_2026,
  author = {{TN Holdings}},
  title  = {Ordering Sensitivity in Subgraph-Isomorphism Layout Search:
            A Characterization of Catastrophic Failure Regions in
            Quantum Circuit Transpilation},
  year   = {2026},
  doi    = {10.5281/zenodo.22977930},
  url    = {https://doi.org/10.5281/zenodo.22977930},
  note   = {Preprint, revised 26 September 2026; version 1: 10.5281/zenodo.22869976}
}

@misc{psf_zero_paper_2026,
  author = {{TN Holdings}},
  title  = {PSF-Zero: An Analytic Two-Qubit Gate Synthesizer Combined
            with a Verified, Ordering-Aware Layout Search for Quantum
            Circuit Transpilation},
  year   = {2026},
  doi    = {10.5281/zenodo.22978090},
  url    = {https://doi.org/10.5281/zenodo.22978090},
  note   = {Preprint, revised 26 September 2026; version 1: 10.5281/zenodo.22870141}
}
```

AGPL v3. See `LICENSE`. **Evaluating this for potential commercial use?** AGPL's copyleft terms may not fit a
closed-source integration; see [Working with us](#working-with-us) before assuming the license as published is the
final word.

[Previous repository.](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/Previous_repository.md)
