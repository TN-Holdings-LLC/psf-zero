> **Archived record — part 6 of 6.**
> This file is one slice of the original PSF-Zero README, kept **verbatim**. It is a
> lab notebook: it contains claims that were later retracted, hypotheses that were
> later refuted, and corrections written directly underneath the text they correct.
> Nothing has been deleted. For what the project currently claims, read the
> [README](../../README.md); for a settled account of one topic, read
> [`docs/findings/`](../findings/).
>
> What had not been verified, the design notes, the full Roadmap with every DONE / RESOLVED / REFUTED / CLOSED annotation, citation and license.
>
> Source: README.md lines 2608-2973, as of 2026-09-11.

---

## What we haven't verified yet

In the interest of not overstating anything:

- **Why PSF-Zero v6 loses fidelity on `deep2q`/`multi_deep2q`: root cause
  strongly corroborated and fixed on two independent fronts, still not
  literally confirmed against the actual lost benchmark script.**
  Section 8's noisy-simulator comparison shows a real, repeatable fidelity
  deficit relative to Qiskit, TKET, and the Hybrid pipeline on two of the
  three circuit families tested. Investigating it found and fixed two
  separate, real issues: (1) `compile_for_hardware()` silently left
  `RXX`/`RYY`/`RZZ` undecomposed because `basis_gates` was never threaded
  through to its `transpile(...)` call, and (2) even once a target basis is
  supplied, `RXX`/`RYY`/`RZZ` costs native hardware gates that `CX` doesn't
  at low transpile optimization levels — confirmed directly against the
  real production `psf_compile.py` and real `test_real_hardware_fidelity.py`
  script (not just a stand-in), and fixed with an opt-in
  `entangling_basis="cx"` parameter that closes the gap on both affected
  families while leaving `wide` (where PSF-Zero makes no changes anyway)
  unaffected, as expected. We looked for the *original* scripts that
  produced sections 7/8's very first numbers across the working repository
  and did not find them — they appear to be lost, not merely unexamined —
  so we can't say with certainty that this exact mechanism, rather than
  some combination of it and something else, produced those specific
  original numbers. What we can say is that the mechanism is real,
  reproduces at matching qualitative and quantitative scale on three
  independent fronts (a from-scratch stand-in compiler, the real
  `compile_for_hardware()` patched and run end-to-end, and the real,
  unmodified `test_real_hardware_fidelity.py` with only its PSF call site
  changed), and a validated fix exists and is confirmed against the real
  production code.
- **RESOLVED. Why PSF-Zero's own Rust-core synthesis cost more per block
  than a warmed-up Qiskit transpile, beyond the smallest scale tested
  (section 4).** Breaking `synthesize()` into its four sub-phases
  (`benchmarks/profile_synthesize_breakdown.py`) found the actual
  decomposition call was only ~3.3% of the per-block time; ~87% went to
  `synthesize()`'s own unconditional `Operator()` fidelity self-check (the
  "no silent fallback" policy re-verifying every synthesized block against
  the target unitary on every call) — not to Qiskit doing anything
  Qiskit-specific. (That also reframes the original question: Qiskit's own
  per-block 2-qubit synthesis is itself an analytic Cartan/KAK decomposition,
  not a search, so "PSF-Zero should win because it skips search" was never
  quite the right mechanism at this level.) Making that check optional
  (`verify=False`, `benchmarks/compile_optional_verify.patch`) and
  re-measuring the full 15–1000 qubit sweep with 10 seeds per point on real
  hardware confirmed it: PSF-Zero is faster than Qiskit at every scale
  tested (2.4x–5.2x) once the redundant self-check is skipped, with
  correctness unaffected. See section 4's final table, plus its
  cross-machine cumulative-loop addendum confirming the same ratio holds
  under sustained repeated use on two further, independent machines.
  **Update 2026-09-09 — the product decision this exposed has been
  addressed, in a better way than the binary it was originally framed as.**
  Rather than flipping the default to `verify=False`, `verify` became
  `Union[bool, str]`: `True` (still the default) now runs a cheap
  Rust-core check and `"strict"` runs the old `Operator()` reconstruction.
  Measured over a 50,000-iteration loop, the default path is 4.79x faster
  than Qiskit L3 with the check still on, and `verify=False` is 9.12x. The
  safety net now costs ~1.9x instead of ~4–5x, so the advantage is no
  longer opt-in and no correctness guarantee was given up to get it. See
  section 4's first 2026-09-09 update.
- **RESOLVED. Section 5's compile-time comparison (previously gate-count/depth
  only) needed its own confound-hunting before it could be trusted.** The
  first attempt showed `0/N` blocks processed (wrong circuit generator, same
  root cause section 4 hit); after fixing that, PSF-Zero's
  `compile_for_hardware()` still measured 1.7x–2.9x *slower* than Qiskit's
  `optimization_level=3`. That turned out not to be about warm-up depth or
  `multiprocessing.Process` suppressing Qiskit's internal parallel search
  (both hypotheses were tested directly and ruled out) but about
  `transpile(optimization_level=3)` not pinning `seed_transpiler` — its
  randomized layout/routing search returns a different result, and takes a
  different amount of time, on every call, even for an identical circuit
  (the same 500-qubit circuit measured at both 0.06s and 0.23s across
  separate runs). Pinning `seed_transpiler` and expanding to 10 seeds
  (run twice, 20 measurements per scale) resolved it: PSF-Zero is faster
  than Qiskit's own routed compilation at every scale tested here too, by
  1.0x–1.4x — smaller than section 4's synthesis-only 2.4x–5.2x, which makes
  sense given `compile_for_hardware()` pays for a full separate Qiskit
  routing pass on top of PSF-Zero's own synthesis. See section 5's new
  "Compile time under the same constraint" subsection.
- **RESOLVED (2026-09-08). Which `routing_optimization_level` was actually
  in effect for section 5's own 1.0x–1.4x number, and whether `level=2`
  (this section's own earlier default) was silently discarding PSF-Zero's
  contribution.** Yes: at `routing_optimization_level=2`,
  `compile_for_hardware()`'s output is bit-identical to a plain
  `transpile(optimization_level=2)` call on the *uncompressed* circuit —
  Qiskit's own `init`-stage `ConsolidateBlocks`/`UnitarySynthesis` re-derive
  the decomposition from scratch regardless of what PSF-Zero already did.
  `routing_optimization_level=1` avoids this: same 2-qubit gate count as
  Qiskit's `optimization_level` 2/3 for 1/20th–1/59th of the time, at a real
  but modest depth cost (~30–40%). A custom `PassManager` that strips the
  redundant re-synthesis stages instead of stepping down a level was
  prototyped and rejected (no measurable benefit over plain `rl=1`).
  `compile_for_hardware()`'s default is now 1, not 2. See section 5's
  2026-09-08 update and `phase3-hardware-routing-regression.md`.
- **RESOLVED (2026-09-10), with the claim corrected in the process. Whether
  Qiskit `optimization_level` 2/3's catastrophic slowdown on this workload
  is actually caused by the coupling map having no spare qubits, or merely
  correlated with it.** It is caused by it. The evidence up to 2026-09-09
  was a correlation across four points that three independent environments
  reproduced — but since `get_grid_cmap()` saturates the grid at exactly
  n=100 and n=156, "no spare qubits" and "one of those two sizes" were
  perfectly confounded, and no number of re-runs of those same four points
  could separate them. A controlled experiment
  (`benchmarks/phase3_v5_spare_qubits.py`) that holds the coupling map
  fixed and varies only how much of it the circuit occupies breaks the
  confound: on one unchanged 42-qubit grid, a 42-qubit circuit takes 621ms
  and a 38-qubit circuit takes 34ms; a *smaller* circuit on a saturated
  grid runs 39x slower (`opt=2`) to 190x slower (`opt=3`) than a *larger*
  one with spare qubits. **The correction:** the threshold is not zero
  spare qubits, as previously stated — on a 100-qubit grid, 2 spare is
  still fully slow and 4 spare is fully fast (a 244x cliff at `opt=3`),
  while on a 72-qubit grid 2 spare is already fast. So it is not a fixed
  count and the boundary moves with the map. Still unknown: the mechanism
  (no pass-level instrumentation was done), whether it generalizes beyond
  Qiskit 2.5.2 and grid topologies, and whether the dense adjacent-pair
  circuit structure is also required (the `passthrough` control was not
  re-run). See section 5's 2026-09-10 update.
- **Whether section 7's "14.4x–16.2x faster" (now also confirmed at
  13.3x faster over 11 runs) real-hardware compile-time result holds up
  under the same warm-up correction applied to section 4.** That script
  calls `transpile()`/`compile_for_hardware()` exactly once per process (one
  process per real-hardware job submission), the same structural pattern
  that produced section 4's now-retracted numbers, but we have not re-run
  it with a warm-up patch — doing so means spending real IBM QPU time, and
  we wanted to flag the open question rather than either assume it's fine
  or spend hardware time before deciding it's worth checking. See section
  7's caveat and Roadmap.
- **Whether the compile-time advantage (section 4) actually reduces
  real-hardware calibration-drift exposure in an iterative compile/execute
  workflow (VQE, QAOA parameter search).** Plausible mechanism, not yet
  tested — see section 4's cross-machine addendum and Roadmap.
- **GPU / massively parallel execution.** Because PSF-Zero decomposes each
  2-qubit block independently, the per-block synthesis is embarrassingly
  parallel in principle. We have not implemented or benchmarked a parallel
  execution path — this is a plausible direction, not a measured result.
- **The 1000-qubit "615x–867x" and 156–1000-qubit "Empirical Benchmark
  Dataset" figures from an earlier draft of this README have been removed.**
  Both were produced using circuit generators (`random_circuit()` /
  `generate_scalable_dense_circuit()`) that structurally never produced blocks
  large enough for PSF-Zero's `block_gate_floor` to activate — PSF-Zero was
  returning the input circuit essentially unchanged, and the reported speedup
  reflected doing no work rather than doing the work faster. We caught this by
  directly measuring block sizes in the generators and by observing that
  PSF-Zero's own reported output depth was, in the worst case, no better than
  the unoptimized input. We'd rather retract these than leave them up.

## Design notes

- **Deterministic by construction.** The decomposition is exact and
  closed-form, so the same input unitary always produces the same canonical
  circuit (up to global phase). There is no random seed to control for.
- **Weyl-chamber canonicalization.** Every synthesized 2-qubit unitary is
  projected into the canonical region ($0 \le c_3 \le c_2 \le c_1 \le \pi/2$),
  so results are directly comparable across runs.
- **No silent fallbacks.** Degeneracies and edge cases in the decomposition are
  surfaced as explicit Rust `Result` errors rather than approximated away.
- **Scope.** PSF-Zero targets the 2-qubit unitary synthesis step specifically.
  It is not a full replacement for a transpiler's routing, layout, or
  multi-qubit gate decomposition — it composes with those (as shown in the
  coupling-map benchmark above), it doesn't replace them.

## Roadmap

- **DONE.** Section 4's 300/500/1000-qubit points now have a proper 10-seed
  loop (added to `phase2.py` alongside the `verify=False` change), matching
  the 15/50/100/156-qubit points' statistical footing. The whole 15–1000
  qubit curve is now on equal footing.
- **DONE — this was the highest-priority item, and it's now confirmed, not
  projected.** `verify=False` (`benchmarks/compile_optional_verify.patch`,
  applied to the real `psf_compile.py` and integrated into
  `phase1_verify_false.patch` / `phase2_verify_false.patch`) is confirmed
  on the real `psf_zero_core`, real hardware, across the full 15–1000 qubit
  range, 10 seeds per point: correctness unaffected, and PSF-Zero faster
  than Qiskit at every scale tested (2.4x–5.2x) — see section 4's final
  table, now further confirmed under sustained repeated use across two more
  independent machines (section 4's cross-machine addendum).
- **DONE (2026-09-09), and the original framing of the question was wrong.**
  The open item used to be "should `verify=False` become the new default,
  rather than staying opt-in?" — a choice between speed and the safety net.
  It was resolved by not taking that trade: `verify` became
  `Union[bool, str]`, where `True` (unchanged as the default) now runs a
  cheap Rust-core check and `"strict"` preserves the old `Operator()`
  reconstruction for anyone who wants it. The default path measures 4.79x
  faster than Qiskit L3 over a 50,000-iteration loop with verification
  still on; `verify=False` measures 9.12x. Nothing was traded away. See
  section 4's first 2026-09-09 update.
- **NEW, and it should be closed before any speed number in this README is
  quoted externally: a single `test1_v3.py` run showed ratios ~3x lower
  than the Linux sandbox** (`opt=3 ÷ psf canonical` at 156 qubits: 4.15x
  vs. 1.44x), **and the mechanism this README proposed for it — a spinning
  0.2ms RSS sampler contending for the GIL during measurement — is
  retracted (2026-09-10) after actually reading `test1_v3.py`'s source.**
  The sampler thread only runs in a separate, untimed pass *after* the
  timed repetitions finish; it cannot affect a measurement it doesn't
  overlap with. Two independent re-runs since (a different script,
  `phase1.py`/`phase2.py`, with coarse-or-no memory sampling, on one
  machine; and `test1_v3.py` itself, unmodified, sampler still present and
  still showing the same odd sample-count signature, on a different
  machine) both gave healthy ratios (2.78x–9.51x and 2.8x–7.65x
  respectively) — consistent with the sampler being irrelevant, as the
  source now shows, but neither one identifies what actually caused the
  original single low run. **The open item now is simply: the original
  4.35x→1.44x table has not reproduced on either of two later runs, on two
  different machines, and no candidate mechanism explains it.** It may be
  a one-off environmental fluke (this project has documented exactly this
  shape of thing before — see the retracted 10,000/50,000-iteration decay
  above, also traced to transient contention) or it may recur; without a
  repeat occurrence there is nothing further to investigate right now.
  Treat the original table as an unexplained outlier, not as this script's
  typical behavior.
  **Update (2026-09-10): substantially explained, one flag away from
  confirmed.** A fourth run (second on the Intel machine, agreeing with the
  first to within 4–11%) made it possible to compare the original run's
  absolute times arm by arm instead of only its ratios. The original was
  paying two *scale-independent* penalties — ~1.35x on the Qiskit arm,
  ~2.65x on the PSF arm — so its "decline with scale" is the same shape
  every run of this script has, uniformly scaled down; the thing to explain
  is the arm asymmetry, not the slope. The ~1.35x matches this project's
  measured machine-to-machine gap and the extra ~1.95x on the PSF arm
  matches the size of the 2026-09-09 `verify` change, making "the slower
  machine running a pre-2026-09-09 `psf_compile.py`" a quantitatively
  consistent account. It is not confirmed — nobody recorded which
  `psf_compile.py` that run used. **Remaining action: re-run `test1_v3.py`
  with the PSF arms at `verify="strict"`; the ratios should collapse to
  roughly the original's 4.35x/2.21x/1.66x/1.44x if the account holds.**
  See section 4's 2026-09-10 fourth-run update, and the Correction and
  Updates that precede it.
  **DONE and REFUTED (2026-09-10). This whole item is now closed.** The
  experiment above was run as specified (paired, same run, 10 seeds × 5
  reps, `benchmarks/test1_v3_verify_strict.py`). The ratios did not
  collapse to the original's 4.35/2.21/1.66/1.44 — they went to
  **1.42/0.67/0.48/0.47**, overshooting by about 3.1x at every scale,
  because `verify="strict"` slows the PSF arm by 5.3x–7.0x where the
  account needed 2.5x–2.8x. The pre-2026-09-09-`psf_compile.py` account is
  therefore rejected by its own pre-registered criterion, no candidate
  mechanism remains, and — since the original run's output file and
  environment record are both gone (see the CLOSED note below) — none can
  now be tested against it. **The 4.35x→1.44x table is a single anomalous
  run, permanently unexplained, and should not be treated as evidence about
  anything.** Two useful by-products: `verify="strict"` costs 5.1x–6.6x the
  current default and makes PSF-Zero *slower* than Qiskit `opt=3` at every
  scale above 15 qubits, and that cost is larger than the pre-2026-09-09
  default appears to have been, so `verify="strict"` should not be
  described as simply "the old default, still available" without checking.
  See section 4's refutation update.
  **Separately, the machine-attribution half of this item is now CLOSED as
  unanswerable (2026-09-10).** A Windows account name was found to be shared
  across more than one physical machine, which is why recent updates
  identify machines by the CPU signature the run itself printed. For the
  *original* declining run there is no such record to consult: all thirteen
  of this project's accumulated raw CSVs were reviewed and ten of them
  carry no environment metadata at all, while `test1_v3.py` writes to a
  fixed filename and has overwritten its own earlier output. The files are
  archived at `psf-zero/data/archive/` with a provenance map. Nothing
  further can be recovered; the fix is forward-looking and already in place
  (every current harness records `platform.processor()`).
- **DONE.** Section 5's compile-time comparison now has its own confirmed,
  seed-pinned, 20-measurement-per-scale result (1.0x–1.4x faster than
  Qiskit) — see section 5 and the RESOLVED item above.
- **DONE (2026-09-10).** The saturated-coupling-map instability in Qiskit's
  `optimization_level` 2/3 — flagged on 2026-09-08 as "a correlation with
  no confirmed mechanism" and sent here for a controlled experiment
  varying the spare-qubit count — has had that experiment run
  (`benchmarks/phase3_v5_spare_qubits.py`). Holding the coupling map fixed
  and varying only the circuit's occupancy confirms spare qubits are the
  causal variable, and corrects the threshold: not zero spare, but a
  cliff whose position moves with the map (2 spare is slow on a 100-qubit
  grid, fast on a 72-qubit one). The mechanism itself is still unidentified
  and stays open, along with whether it survives outside Qiskit 2.5.2 and
  grid topologies. A useful by-product for anyone hitting this: padding the
  coupling map with a few spare qubits removes the blow-up entirely. See
  section 5's 2026-09-10 update.
- **DONE (2026-09-08).** `compile_for_hardware()`'s `routing_optimization_level`
  default corrected again, 2 → 1: at level 2, its output is bit-identical to
  a plain `transpile(optimization_level=2)` call on the uncompressed
  circuit — PSF-Zero's own synthesis work is computed and then entirely
  discarded. Level 1 keeps PSF-Zero's synthesis intact: the same 2-qubit
  gate count as Qiskit's `optimization_level` 2/3 for a fraction of the
  compile time, at a real but modest depth cost. A custom `PassManager`
  alternative was prototyped and rejected (no measurable benefit over plain
  `rl=1`). See section 5's 2026-09-08 update and
  `phase3-hardware-routing-regression.md`.
- **DONE.** Whether the `RXX`/`RYY`/`RZZ` native-gate-cost hypothesis for
  section 8's fidelity gap actually holds against the real production code:
  confirmed directly, and fixed with an opt-in `entangling_basis="cx"`
  parameter — see section 8's second root-cause thread.
- Whether the compile-time advantage (section 4) reduces real-hardware
  calibration-drift exposure in an iterative compile/execute workflow (VQE,
  QAOA parameter search) — plausible, not yet tested, and not planned
  without a specific reason to spend real QPU time on it.
- `compile_for_hardware()` doesn't yet expose a `seed_transpiler` parameter
  of its own, so its internal routing `transpile()` call is still unpinned
  even after section 5's fix on the Qiskit-comparison side. We saw no sign
  of instability from this on the PSF-Zero side while producing section 5's
  table, but the comparison isn't perfectly symmetric until this is added.
  A general lesson worth carrying forward: any future benchmark that calls
  `transpile()` with a `coupling_map` at `optimization_level >= 2` should
  pin `seed_transpiler` from the start, the same way this project now pins
  circuit generation seeds — we found this the hard way, twice.
- Section 5's compile-time comparison (and, apparently, its gate-count/
  depth benchmark too) only stresses adjacent logical pairs, which never
  require a SWAP under a row-major grid coupling map — so neither engine's
  router has done any real work in either table yet. A version built on
  non-adjacent logical pairs, which actually forces SWAP insertion, is
  needed before this project can claim to have measured routing cost under
  real connectivity pressure rather than just block-synthesis cost with a
  free routing pass tacked on.
- Re-running section 7's real-hardware compile-time comparison with the same
  symmetric warm-up treatment applied to section 4 — if the real-hardware
  numbers hold up under that correction, that's worth confirming explicitly
  rather than leaving as an open caveat; if they don't, section 7 needs the
  same kind of correction section 4 just got.
- Applying the validated `compile_for_hardware()` fix (`basis_gates`
  parameter, `routing_optimization_level` now defaulting to 1 — see section
  8 and section 5's 2026-09-08 update) and the `entangling_basis="cx"` fix
  to the real repository: minimal, backward-compatible patches for both are
  ready to apply —
  [`benchmarks/compile_for_hardware.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_for_hardware.patch)
  — and existing call sites need to start passing `basis_gates` explicitly
  (e.g. `backend.operation_names`) for it to take effect. Real,
  independently-confirmed bugs worth fixing on their own merits, regardless
  of whether they turn out to be the full cause of the original fidelity
  gap.
- `real_device_15q_fidelity_v2.py` and the *original*
  `test_real_hardware_fidelity.py` — the scripts that actually produced
  sections 7 and 8's very first fidelity numbers — were searched for across
  the working repository and not found. A working copy of the *current*
  `test_real_hardware_fidelity.py` has since been recovered and used
  directly (see section 8's second root-cause thread), but it postdates the
  original numbers, so this item isn't fully closed. If the original
  scripts resurface (backup, another machine, version control history),
  re-running them against the fixed `compile_for_hardware()` and
  `entangling_basis="cx"` would settle the remaining provenance question
  directly.
- Repeating the real-hardware fidelity comparison (section 7) on more
  backends and larger qubit counts.
- Running PSF-Zero through [Benchpress](https://github.com/Qiskit/benchpress) (IBM's open-source SDK benchmark suite)
  for an apples-to-apples comparison against Qiskit, TKET, and the other SDKs
  it already covers, on its own broad, realistic circuit collection rather
  than our own narrower constructions. (In progress: opened an upstream
  discussion on Benchpress's own integration process — see
  [Benchpress issue #114](https://github.com/Qiskit/benchpress/issues/114) —
  and started prototyping a `psf_gym` folder modeled on the existing
  `tket_gym`.)
- Exploring parallel (multi-core / GPU) execution of independent block
  synthesis — currently unimplemented.
- PennyLane integration (`qml.transforms`) — planned, not yet built.

## Citation

```bibtex
@software{psf_zero_2026,
  author = {The Architect},
  title = {PSF-Zero: Analytic KAK Decomposition for Two-Qubit Circuit Synthesis},
  year = {2026},
  url = {https://github.com/TN-Holdings-LLC/psf-zero},
  license = {AGPL-3.0}
}
```

## License

AGPL v3. See `LICENSE`.

[Previous repository.](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/Previous%20repository.md)
