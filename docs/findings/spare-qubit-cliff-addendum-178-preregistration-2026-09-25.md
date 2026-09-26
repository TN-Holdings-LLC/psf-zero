# Addendum 178 -- Pre-registration: does the layout cliff appear on IBM's square-lattice generation (FakeNighthawk), and if so, which compiler meets a 1-second deadline? Quality equivalence pre-registered this time (2026-09-25)

**Status: pre-registration only. No measurement has been run.**

## 1. Why this experiment exists

Two lines of evidence meet here.

- **Quality**: wherever the layout cliff does not occur, PSF-Zero and
  Qiskit produce circuits of the same quality (Addenda 157, 159, 172, and
  the depth sweep of Addendum 177 in progress). PSF-Zero's established
  advantage is speed at the cliff, not output quality.
- **Where the cliff can occur**: it was found on square grids; on IBM's
  heavy-hex devices it does not occur, because heavy-hex graphs admit no
  perfect matching (workplace Addenda 39-40) and random circuits rarely
  land on the narrow feasible-and-saturated condition (Addenda 135-136).
  FakeNighthawk, a snapshot of IBM's newer square-lattice generation,
  appeared in the Stage 1 backend list (Addendum 167). Whether the cliff
  appears there has never been tested.

A comparison of output quality alone cannot show a speed advantage, just as
an untimed exam cannot show who answers faster. This experiment therefore
scores compilers the way a timed exam does: did a correct answer arrive
within the deadline?

## 2. Design

**Stage 0 (prerequisite, run first; the experiment stops if it fails).**
From `FakeNighthawk().coupling_map`, report the qubit count, the degree
distribution, whether the graph is bipartite with equal parts, and the size
of a maximum matching. The cliff's condition (spare = 0: every physical
qubit used by disjoint interacting pairs) requires a perfect matching.

**Circuits.** The generator every cliff script in this project uses
(`build_dense_pair_blocks_circuit`, copied verbatim from
`bench_cliff_1v1.py`): logical pairs (0,1), (2,3), ..., each carrying 20
Haar-random 2-qubit unitaries. Logical qubit count = N - spare for spare
in {0, 2, 4, 8} (N = FakeNighthawk's qubit count). 5 seeds per spare.

**Compilers.** Each call runs in a fresh child process, timed inside the
child around the compile call only, killed at a hard cap of 180 s
(recorded as "did not finish", DNF). A process is used because the
layout search runs in compiled code that a Python timer signal cannot
interrupt.
- **Q3 -- Qiskit default**: `transpile(qc, FakeNighthawk(),
  optimization_level=3, seed_transpiler=0)`.
- **P -- PSF-Zero**: `compile_for_hardware(qc, coupling_map=backend
  coupling map, basis_gates=backend native gates, entangling_basis="cx",
  layout_search=True, on_unsupported="raise", seed_transpiler=0)`, other
  arguments at their defaults (including the layout search's own 2 s time
  budget).

**Deadline.** Primary: **1 second**, chosen before running from the
intended use -- recompiling on every iteration of a training loop with
hundreds of iterations, where more than about a second per compile
dominates the loop. Only the 1 s deadline scores N2 and N3. Success rates at
**0.01 s, 0.1 s, 1 s and 10 s** are all reported side by side, not scored,
so the result can be read across deadlines without the verdict depending
on which one was picked (scoring every deadline would raise the chance of
a difference appearing somewhere by accident). The four levels follow the
response-time limits long used in usability work -- about 0.1 s for a
response to feel instantaneous, about 1 s for a user's flow of thought to
stay uninterrupted, about 10 s for attention to stay on the task -- plus
0.01 s, below human perception, where only machine-driven repetition (a
training loop) feels the difference. The 1 s primary deadline is the "flow
stays uninterrupted" limit. Because every compile's time is recorded, any
other deadline can be computed later; if one is, it is labelled as chosen
after seeing the data. Expected in advance: at
0.01 s neither compiler meets the deadline, because P's time includes
Qiskit's own routing at optimization level 1 on a ~120-qubit device, not
only PSF-Zero's layout search and synthesis.

**Quality.** For each finished compile: routed two-qubit gate count; and,
when the routed circuit contains no two-qubit gate outside the physical
positions of a single logical pair (i.e. no SWAP connects different
pairs), an **exact per-pair check**: the routed operations on each pair's
two physical qubits, as a 4x4 operator, against that pair's logical
operator, up to global phase. The circuit is a product of independent
pairs, so this is exact at 120 qubits, where a whole-circuit operator is
not computable.

## 3. Pre-registered predictions

**N0 (prerequisite).** FakeNighthawk's coupling graph has a perfect
matching (spare = 0 is feasible). If not, the experiment stops and reports
that the cliff's condition cannot arise on this device.

**N1 (the cliff exists on Nighthawk).** Q3's median compile time at
spare = 0 is at least 10 times its median at spare = 8, or Q3 does not
finish within 180 s in at least 3 of 5 seeds at spare = 0.
**If N1 fails, the cliff does not appear on this snapshot, and N2 is not
applicable: that is the finding.**

**N2 (the deadline -- the main prediction, applicable only if N1 holds).**
At spare = 0, P finishes within 1 s in at least 4 of 5 seeds, and Q3 in at
most 1 of 5.

**N3 (no advantage away from the cliff).** At spare = 8, both P and Q3
finish within 1 s in at least 4 of 5 seeds.

**N4 (quality equivalence, pre-registered).** For every seed and spare
where both finish: equal routed two-qubit gate counts, and every pair
passes the exact per-pair check (infidelity < 1e-9) for both compilers.
If routing inserted SWAPs so that the per-pair check does not apply, that
is reported, with two-qubit counts compared instead.

**The assistant's expectation, stated before running**: N0 holds (a square
lattice with an even qubit count has a perfect matching); N1 is genuinely
uncertain, because the cliff was measured with a bare coupling map and
Qiskit's preset passes with a full device target differ.

## 4. What this cannot establish

- Real hardware; FakeNighthawk is a snapshot.
- Whether 1 s is the right deadline for any particular user (0.1 s and
  10 s reported for that reason).
- Noisy execution quality (no simulation; structural and exact per-pair
  checks only).

## 5. Script lock

`nighthawk_deadline_cliff.py`, normalized SHA-256 (trailing whitespace
stripped per line, surrounding blank lines removed):
`ce11be15185a46621f1b99947763b8540d661c5eb9f8c30a9bd2cb1627f9a4ff` (the Section 6 version; it supersedes `abd5f02d...`).
Re-check on the machine that runs the experiment BEFORE running, and record
the check in the results. Output: `nighthawk_deadline_cliff_2026-09-25.csv`
(40 rows: 4 spares x 5 seeds x 2 arms). The deadline list was widened to 0.01 / 0.1 / 1 / 10 s before any run; the hash above is of that version. Worst-case running time is bounded
by the 180 s cap per compile (at most 2 hours if every compile hit the cap).

## 6. Amendment before a valid run: a failed first run, its causes, and the fix

**Run 1 (invalid, recorded as a failure).** Run on WSL2 (home). The
machine's file was an OLD version (7242 bytes, hash `78666802...`, before
the deadline list was widened) -- it did not match the locked hash, and the
run should not have been started. Stage 0 completed; then all 39 compiles
before the run was interrupted (spare 0, 2, 4 and 8, both arms) hit the
180 s cap (DNF), including spare = 8, where no cliff is expected. No CSV
was written (it is written only at the end). Log: `nighthawk_result.txt`.

**Stage 0 from Run 1 is a valid observation** (it does not depend on the
harness): FakeNighthawk has 120 qubits and 218 couplings, degree 2 to 4
(median 4), is bipartite with parts 60 and 60, and has a perfect matching
(60 pairs). **N0 holds**: unlike heavy-hex, the cliff's condition
(spare = 0) can arise on this device.

**Diagnosis (in a single process, no child processes), spare = 8, seed 0:**
building the circuit took 0.52 s (12,320 instructions); Q3 compiled in
0.18 s; P first failed with `ImportError` because `psf_smart_layout.py`
(in `benchmarks/`) was not importable, then compiled in 0.29 s once
`benchmarks/` was on the import path. So the compilers themselves were
fast; the harness was broken in two ways:
1. forking child processes after Qiskit's internal thread pools had
   started left every child hung until the cap;
2. P could not import its layout search at all.

**Fix (script changes only; design, predictions and deadlines unchanged):**
child processes use the "spawn" start method; the script puts its own
directory and `benchmarks/` on the import path. New hash above.

**Disclosed prior knowledge:** the diagnosis showed, before any valid run,
that at spare = 8, seed 0 both arms finish well within 1 s. That is one of
the five seeds N3 scores; N3 is kept as registered and this is recorded so
it is not mistaken for a blind prediction for that seed.
