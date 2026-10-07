# PSF-Zero releases and dated notices

Every release block, update and correctness notice that the README carried until 2026-10-05, newest first and
unchanged in wording, followed by those since (from 2026-10-05.1 on, each release is added here and the README
keeps only the current one). They were moved here from the README on 2026-10-05; only links were adjusted so that they work
from this folder. The full record behind each entry is in Parts 9 and 10 of the findings
([`findings/spare-qubit-cliff-combined-248.md`](findings/spare-qubit-cliff-combined-248.md), Addenda 248-382;
[`findings/spare-qubit-cliff-combined-383.md`](findings/spare-qubit-cliff-combined-383.md), from Addendum 383) and
the earlier parts they link to. The README itself, as it was before the move, is kept as
[`README_2026-10-05_before_restructure.md`](README_2026-10-05_before_restructure.md).

> **Current version (2026-10-06, fourth release): `psf_compile.py` 2026-10-06.4 and the AI front end a12, with
> `psf_smart_layout` 2026-10-01.1 and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)**
> ([Part 9](findings/spare-qubit-cliff-combined-248.md), Addenda 379-381). The recommended call is unchanged.
>
> - **Item 45:** item 39 refuses a Qiskit-made candidate whose check cannot be made, and it cannot be made for a
>   logical circuit of more than 16 qubits (`_implements`) or a circuit touching more than 16 (`_same_action`).
>   2026-10-06.3 still built the floor candidate and Qiskit level 3 there, and item 35's re-synthesis, only to
>   refuse them. 2026-10-06.4 does not build them and counts the skips in `SKIP_STATS`. The output is
>   2026-10-06.3's; only counters and a `callback`'s calls during the floor's compile differ.
> - **Pre-registered test SKIP (Addenda 379-380;** 294 circuits on FakeTorino, FakeKingston, FakeAuckland,
>   FakeHanoiV2, FakeBrussels, FakeOsaka): the same circuit on 294 of 294; above 16 qubits the median time ratio
>   per device 0.21-0.28 (130 s to 18 s in total), at full occupancy of FakeAuckland and FakeHanoiV2 0.013; up to
>   16 qubits 1.00-1.03; 4 of 4 predictions confirmed.
> - **Not changed:** up to 16 qubits the recommended call still builds and checks every candidate; at 16 qubits
>   that takes 20-60 s for Hamiltonian and QFT circuits (Addendum 380).

> **Previous release (2026-10-06, third release): `psf_compile.py` 2026-10-06.3 and the AI front end a12, with
> `psf_smart_layout` 2026-10-01.1 and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)**
> ([Part 9](findings/spare-qubit-cliff-combined-248.md), Addenda 377-378). The recommended call is unchanged.
>
> - **Item 44:** item 39's checks (`compare_floor`, `compare_level3`, `final_resynthesis="select"`) built the
>   logical circuit's action from `to_matrix()` of every instruction, before their 16-qubit limit. On Benchpress's
>   HamLib inputs (one `PauliEvolutionGate` on all qubits, FakeTorino) 2026-10-06.2's recommended call aborted the
>   process at 48 qubits (Rust allocation failure, no exception) and ran past 600 s at 14 qubits; so would the AI
>   front end. 2026-10-06.3 applies the limits first and expands an instruction on more than 6 qubits through its
>   definition. For a `PauliEvolutionGate` that is the product formula every compiler builds, not the exact
>   exponential. On circuits whose instructions act on at most 6 qubits the output is 2026-10-06.2's.
> - **BP-PROBE (Addendum 377, exploratory;** 12 Benchpress tests against Qiskit level 2, Benchpress's call): every
>   output valid; the default call used more two-qubit gates on 10 of 12 (geometric mean 1.53x) and more time on
>   11; it was level on Quantum Volume. With 2026-10-06.3 the recommended call finished the 14-qubit HamLib test
>   with 3,616 two-qubit gates (Qiskit level 2: 3,689) in 38.8 s; above 16 qubits it returns the default call's
>   circuit, because item 39 cannot check the alternatives.

> **Previous release (2026-10-06, second release): `psf_compile.py` 2026-10-06.2 and the AI front end a12, with
> `psf_smart_layout` 2026-10-01.1 and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)**
> ([Part 9](findings/spare-qubit-cliff-combined-248.md), Addenda 372-374). The recommended call is unchanged.
>
> - **Item 43:** with a `target`, an output that uses a failed element is compiled again on the pruned coupling
>   map (item 31). When the circuit needs the whole device, no placement on that map exists and 2026-10-06.1 raised
>   `TranspilerError` ("A connected component of the DAGCircuit is too large ...") -- in the recommended call and
>   in the AI front end. 2026-10-06.2 keeps the first output, warns, and counts it in
>   `PRUNE_STATS["unavoidable"]`. Everywhere else the output is 2026-10-06.1's.
> - **Pre-registered test PL-REDO (Addenda 372-373;** the PennyLane loop of Addenda 254-255 on FakeKingston, 156
>   qubits, 20 laps, on the owner's Linux machine): 2026-10-06.1's recommended call and a12 raised on 20 of 20 laps
>   at full occupancy, the candidate on none, and it returned 2026-10-06.1's circuit on all 40 laps where
>   2026-10-06.1 returned one; the default call took 0.048 s per compile, swap-free, meaning kept to 1.3e-13; at
>   full occupancy the target-aware calls take Qiskit level 3's time (16 s); 11 of 11 predictions confirmed.
> - **Not changed:** the default call (no target) does not look at failed elements. With 16 spare qubits it used
>   one on every lap, while the target-aware calls avoided them in 0.4-0.6 s.

> **Update (2026-10-06) -- AI front end a12** ([Part 9](findings/spare-qubit-cliff-combined-248.md), Addenda 364-367).
> `benchmarks/psf_ai_compile.py` is now a12 (a11 is kept as `benchmarks/psf_ai_compile_a11.py`). The release is
> unchanged.
>
> - **Item 17:** the state-aware re-placement keeps each gate's scoring terms within a call, and re-places identical
>   candidates only once. The scores are the same numbers added in the same order, so the result is a11's.
> - **Pre-registered test SPEED (Addenda 364-365;** 128 model-style circuits on each of 6 fake devices): identical to
>   a11 on 768 of 768; median time 0.60 and 0.54 times a11's on the cz devices, 0.74-0.75 on the ecr devices,
>   0.86-0.89 on the cx devices; 3 of 3 predictions confirmed.

> **Previous release (2026-10-06): `psf_compile.py` 2026-10-06.1 and the AI front end a11, with `psf_smart_layout`
> 2026-10-01.1 and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)**
> ([Part 9](findings/spare-qubit-cliff-combined-248.md), Addenda 357-362). The recommended call is unchanged; compile a
> circuit that will be sampled with its final measurements.
>
> - **Item 40 (release) and item 15 (a11): readout is counted.** The choice among candidates and the front end's
>   estimate now include the readout error of each measured qubit. A circuit without measurements gets the same
>   result as before.
> - **Item 16 (a11): gate direction kept on ecr devices.** a9 could return ECR gates in a direction the device does
>   not provide; a11 never does, and checks any fix-up it makes.
> - **Item 41 (release):** the failed-element check is direction-aware, like the pruning. It changed no output in the
>   test.
> - **Item 42 (release), opt-in:** `candidate_score="kraus"` chooses by an estimate exact to first order for the
>   simulator's noise model. It has no readout term, so the recommended call stays `candidate_score="hybrid"`.
> - **Pre-registered test RECR (Addenda 358, 360;** 104 sampled circuits on each of 6 fake devices, cx, cz and ecr):
>   - compiled with measurements, the summed readout error of the measured qubits was 0.40 and 0.18 times that of
>     compiling without them, on the two cz devices (2026-10-05.1 could already do this; it was not documented);
>   - the release's classical infidelity with measurements was 0.96-1.00 times 2026-10-05.1's, with lower readout
>     error on every device; without measurements its output was 2026-10-05.1's on every circuit;
>   - a11's classical infidelity was 0.46 and 0.16 times a9's on the two cz devices, and 0.63-0.98 times Qiskit level
>     3's on the six devices;
>   - a9 returned 622 and 860 gates in an unsupported direction on FakeBrussels and FakeOsaka; a11 none;
>   - 10 of 10 predictions confirmed; compile time 0.91-1.10 times.
> - **Pre-registered test KRAUS (Addenda 359, 361;** 1,506 circuits on each of 9 fake devices): `kraus` chose at
>   least as well as `hybrid` on every device (0.998-1.000 times its infidelity), 0.937 times on HOLD6's H4 case;
>   ahead of Qiskit level 3 (0.959-0.990); 3-5% more compile time; 7 of 7 predictions confirmed.
> - **Known limits:**
>   - Fake devices only. Readout on hardware drifts and is correlated across qubits; Aer's model is neither.
>   - `kraus_cost` reproduces Aer's own noise model, so a simulation test favours it by construction.
>   - Both scored runs were made on Windows, where Qiskit issue #17057 does not appear (Addendum 357); the locks were
>     committed before the runs and pushed afterwards (the workplace PC has no GitHub login).

> **Previous release (2026-10-05): `psf_compile.py` 2026-10-05.1, with `psf_smart_layout` 2026-10-01.1 and the Rust
> core `CORE_VERSION` 2026-09-29.1 (both unchanged)** ([Part 9](findings/spare-qubit-cliff-combined-248.md),
> Addenda 340-344). A correctness fix for 2026-10-03.1 to 2026-10-04.1 (the known defect below); the recommended call
> is unchanged:
>
> - **Item 39:** every circuit that Qiskit makes as a whole is checked for equivalence before it can be returned:
>   item 35's re-synthesis against the release's own circuit, and items 36-37's floor-placed and level-3 candidates
>   against the input (two seeded random product states, state infidelity <= 1e-6, global phase ignored). A circuit
>   that fails, or cannot be checked, is refused and the release's own, guarded circuit is kept.
> - **Pre-registered test (EXACT, Addenda 342-343):**
>   - near-boundary and control circuits (128 per device on 4 cx and 2 cz devices): 2026-10-04.1 wrong on 23-27 of 40
>     explicit near-boundary unitary circuits on each cx device (infidelity up to 0.34), 2026-10-05.1 on none;
>   - HOLD6's 1,506 F circuits on each of 9 devices: the same circuits as 2026-10-04.1, all exact;
>   - 8 of 9 predictions confirmed; the time prediction (median <= 1.2 times) ambiguous at 1.26 (1.27-1.42 by device,
>     35-55 ms more per compile);
>   - an exploratory check after the run: all 128 refusals of item 35's re-synthesis on FakeAuckland were of wrong
>     circuits (process infidelity 1.2e-5 to 0.34), none of exact ones. On near-boundary Trotter circuits 81% of the
>     re-syntheses were wrong; 2026-10-04.1 happened not to select them there.
> - **Known limits:**
>   - Accepted Qiskit-made circuits can be off by up to 1e-6 in state infidelity (5.8e-8 seen, where the guarded path
>     is exact to 1e-14).
>   - Circuits with resets or conditionals, and above 16 touched qubits, cannot be checked: the release's own circuit
>     is kept.
>   - Not tested on hardware or on ecr devices.

> **Update (2026-10-05) -- AI front end a9** ([Part 9](findings/spare-qubit-cliff-combined-248.md), Addenda 342-344).
> `benchmarks/psf_ai_compile.py` is now a9 (a8 is kept as `benchmarks/psf_ai_compile_a8.py`).
>
> - **The defect it fixes:** with a target, a7 and a8 offered Qiskit level 3's output as a candidate without checking
>   it; a8 was wrong on 27 of the 128 circuits of the EXACT test on each cx device.
> - **a9** uses level 3's output only if the release's item-39 check confirms it. In the EXACT test: wrong on none;
>   it refused level 3's output 28 times per cx device, exactly level 3's 28 wrong outputs; identical to a8 on every
>   sampled ordinary circuit.
> - **Not fixed:** on ecr devices it can return ECR gates in a direction the device does not provide (a workplace
>   exploration, 2026-10-05). Do not give it a target on ecr devices. *(Fixed in a11, 2026-10-06, Addenda 358-362.)*

> **Known defect, found 2026-10-05 (fix under test: Addenda 340-342).** *(Fixed in 2026-10-05.1 and a9, Addenda
> 343-344; this is the notice as it stood before the fix.)* On cx devices, the recommended call of
> 2026-10-04.1 can return a circuit that is NOT equivalent to the input. The same holds for every call with
> `final_resynthesis` or `compare_level3` since 2026-10-03.1.
>
> - **When:** the input contains two-qubit unitaries near the boundary of Qiskit issue #17057, for example explicit
>   `unitary` gates from numerical optimisation or written by a language model. The cause is that circuits made by
>   Qiskit were used without an equivalence check.
> - **Until the fix is released:** for such circuits on cx devices, use `target=..., placement_refine=True` without
>   `final_resynthesis` and `compare_level3`, or check the result's equivalence yourself.
> - **The AI front end** (`benchmarks/psf_ai_compile.py`) is affected the same way when given a target. A workplace
>   exploration also found that on ecr devices it can return ECR gates in a direction the device does not provide.
>   Until fixed, do not give it a target on cx or ecr devices.
> - cz devices are not affected by the #17057 path.

> **Previous release (2026-10-04): `psf_compile.py` 2026-10-04.1, with `psf_smart_layout` 2026-10-01.1 and the Rust
> core `CORE_VERSION` 2026-09-29.1 (both unchanged)** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 334-338). One opt-in addition to 2026-10-03.3, and one recommended call on every device:
>
> - **Recommended call with a device target (cx and cz devices alike):**
>
>   ```python
>   compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
>                        target=backend.target, placement_refine=True, final_resynthesis="select",
>                        compare_level3=True, compare_floor=True, candidate_score="hybrid")
>   ```
>
> - **`candidate_score="hybrid"`** chooses among the candidates (the release's circuit, the floor-placed one, level
>   3's) by `hybrid_cost`. It counts amplitude damping as `excitation_cost` does, pure dephasing as the Z part of
>   `pauli_cost`, and the reported error above the T1/T2 floor.
>   - Why: `pauli_cost` lost on XXZ chains (amplitude damping) and `excitation_cost` on GHZ chains (dephasing); an
>     exploratory diagnosis found the combination choosing better than either on all nine devices (Addendum 335).
>   - Pre-registered test (Addenda 336-337; 1,506 new circuits on each of 9 devices, plus 72 at 9-10 qubits):
>     - better than 2026-10-03.3's recommended calls on all nine devices (0.02-0.7%);
>     - the XXZ-chain loss of `pauli_cost` repaired on the cx devices (0.8-2.3%);
>     - on cz devices the floor-placed candidate now helps too (FakeMarrakesh GHZ chains 5%);
>     - ahead of Qiskit level 3 on all nine devices (0.959-0.991);
>     - failed elements never used; about 0.16 s per compile (median; 1.8 times 2026-10-03.3's recommended calls).
> - **Known limits:**
>   - On FakeAlgiers, 4-qubit GHZ chains are 6.7% worse than with `candidate_score="pauli"` (all 48 such circuits;
>     6- and 8-qubit chains equal). A fixed ranking error of `hybrid_cost` on one placement; under investigation.
>   - Not tested on hardware, on ecr devices, or above 16 touched qubits.
>   - With a stale calibration (errors off by 30%, T1/T2 by 20%; Addendum 335) 2026-10-03.3's calls stayed ahead of
>     level 3; this release was not tested that way.
> - With the defaults, nothing changes.

> **Update (2026-10-04) -- AI front end a8** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 335-338). `benchmarks/psf_ai_compile.py` is now a8 (a7 is kept as `benchmarks/psf_ai_compile_a7.py`).
>
> - **The defect it fixes:** above 8 qubits a7 compiled WITHOUT the device target (the `target` argument was never
>   forwarded). At 9-10 qubits that gave 1.1-2.8 times the release's infidelity, and failed couplers or qubits were
>   used (WIDE, Addendum 335).
> - **a8** hands such circuits to the release's recommended call with the target. Pre-registered test (Addenda
>   336-337): 13-63% better than a7 there, ahead of Qiskit level 3 on all nine devices, no failed element.
> - At or below 8 qubits, and without a target, a8 is a7.

> **Previous release (2026-10-03, third release): `psf_compile.py` 2026-10-03.3, with `psf_smart_layout` 2026-10-01.1
> and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 330-333). One opt-in addition to 2026-10-03.2, recommended on cx devices only:
>
> - **Recommended call on cx devices** (e.g. FakeAuckland, FakeGeneva, FakeAlgiers, FakeHanoiV2):
>
>   ```python
>   compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
>                        target=backend.target, placement_refine=True, final_resynthesis="select",
>                        compare_level3=True, compare_floor=True, candidate_score="pauli")
>   ```
>
>   On cz devices keep 2026-10-03.2's recommended call (without the last two options).
> - **`compare_floor=True`** adds a third candidate: the release's pipeline re-placed on a Target whose errors are
>   max(reported error, T1/T2 floor). **`candidate_score="pauli"`** chooses among the candidates by `pauli_cost`, a
>   state-aware Pauli estimate that includes dephasing.
>   - Why: on GHZ-type circuits on cx devices the remaining gap to the AI front end a7 was placement, which the
>     release's `excitation_cost` cannot see (Addendum 330).
>   - Pre-registered test (Addenda 331-332; 1,506 new circuits on each of 9 devices):
>     - on the cx devices 0.2-2.0% better than 2026-10-03.2 overall, and 7-28% better on GHZ chains on three of four;
>     - the AI front end a7 within 3.7% on every device;
>     - still ahead of Qiskit level 3 everywhere (0.963-0.993);
>     - about 0.15 s per compile (twice 2026-10-03.2).
> - **Known limits:**
>   - On cz devices no gain (0.1-0.2% loss on three of five), hence the cx-only recommendation.
>   - On XXZ-type chains (F3) up to 1.8% worse: `pauli_cost` averages relaxation into symmetric Pauli errors and
>     misses the decay of |1> that `excitation_cost` sees. A combined estimate is the next step (2026-10-04.1).
>   - Not tested on hardware, on ecr devices, or above 16 touched qubits.
> - With the defaults, nothing changes.

> **Previous release (2026-10-03, second release): `psf_compile.py` 2026-10-03.2, with `psf_smart_layout` 2026-10-01.1
> and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 326-329). One opt-in addition to 2026-10-03.1:
>
> - **Recommended call with a device target:**
>
>   ```python
>   compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True,
>                        target=backend.target, placement_refine=True, final_resynthesis="select",
>                        compare_level3=True)
>   ```
>
> - **`compare_level3=True` also compiles the input with Qiskit's level 3 on the target and keeps whichever circuit
>   has the lower `excitation_cost`** (the estimate of 2026-10-03.1). It never keeps a circuit that touches a failed
>   qubit or a failed gate direction, or one that is off the target.
>   - Why: the release's remaining gaps to level 3 had three different causes (Addendum 326): synthesis of routed
>     periodic chains on cx devices, placement of rings on cz devices, and routing of QFT.
>   - Pre-registered test (Addenda 327-328; fake devices, noisy simulation, 1,506 new circuits on each of 9 devices):
>     - at or ahead of Qiskit level 3 with the Target on all nine devices (mean infidelity 0.967-0.993 times);
>     - in all 63 family-device cells, at most 1.003 times level 3;
>     - better than 2026-10-03.1 on every device (0.920-0.987 times);
>     - the choice picks the better circuit in 88% of circuits, within 0.5% of the best possible on eight devices;
>     - failed elements never used;
>     - about 28 ms extra per compile.
> - **Known limits:**
>   - It selects; it does not repair. The causes above remain inside PSF-Zero.
>   - The AI front end (`benchmarks/psf_ai_compile.py`, a7) is still 0.3-4.8% ahead, most on FakeAuckland and
>     FakeGeneva.
>   - Above 16 touched qubits the estimate is not made and the release's own circuit is kept.
>   - Not tested on hardware.
> - Without `compare_level3`, nothing changes.

> **Previous release (2026-10-03): `psf_compile.py` 2026-10-03.1, with `psf_smart_layout` 2026-10-01.1 and the Rust
> core `CORE_VERSION` 2026-09-29.1 (both unchanged)** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 318-325). One opt-in addition to 2026-10-02.2:
>
> - **`compile_for_hardware(..., target=backend.target, placement_refine=True, final_resynthesis="select")`
>   re-synthesises the finished circuit's two-qubit blocks with Qiskit, and keeps whichever circuit an
>   excitation-aware estimate prefers.**
>   - Why: on cx devices PSF-Zero's own two-qubit synthesis chooses local frames that leave qubits excited during the
>     long cx gates, where amplitude damping acts (Addendum 322).
>   - The estimate is the summed reported gate error, plus, for every gate, duration / T1 times the excited population
>     of its qubits just before it.
>   - Pre-registered test (Addenda 323-324; fake devices, noisy simulation, 1,506 new circuits on each of 9 devices):
>     - better than 2026-10-02.2 on all nine devices (mean infidelity 0.969-0.998 times);
>     - on open chains on the cx devices, the gap to Qiskit level 3 with the Target falls from 1.09-1.21 to
>       1.01-1.05;
>     - the estimate picks the better circuit in 88% of circuits;
>     - failed elements never used;
>     - about 16 ms extra per compile.
>   - `final_resynthesis=True` always re-synthesises. It is not recommended: in the same test it cost 1-2% on the cz
>     devices.
> - **Known gaps** (closed by `compare_level3=True` in 2026-10-03.2, Addendum 328):
>   - Periodic chains on cx devices remain 12-40% behind level 3 (routing).
>   - F1-type rings on cz devices remain 6-8% behind.
>   - The AI front end (`benchmarks/psf_ai_compile.py`, a7) is still 4-9% ahead.
>   - Above 16 touched qubits "select" keeps the release's circuit.
> - Without `final_resynthesis`, nothing changes.

> **Update (2026-10-02) -- AI front end a7** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 312-317). `benchmarks/psf_ai_compile.py` is now a7 (a5 is kept as `benchmarks/psf_ai_compile_a5.py`).
> Given a device Target, a7 also offers Qiskit level 3's own output as a candidate and keeps whichever its state-aware
> estimate prefers; it costs no extra compile. Pre-registered test (fake devices, noisy simulation; 693 GAP circuits
> and 153 model-written circuits per device):
>
> - **The one place a5 trailed Qiskit L3T is fixed:** FakeAuckland Heisenberg chains, from 1.11-1.23 times L3T's
>   infidelity to 0.90-0.93.
> - **Against L3T:**
>   - FakeAuckland GAP circuits 0.95 overall;
>   - model-written circuits 0.73 (FakeAuckland) and 0.92 (FakeTorino, FakeKingston);
>   - Heron GAP circuits 0.98.
> - **Against a5:** never worse on average (0.93-1.00). About 3% of circuits get slightly worse, typically by about
>   1%. Median compile time 0.65 s, against 0.61 s.
> - **Integrating release 2026-10-02.2 inside the front end (a6) changed nothing** (Addendum 313). The front end is
>   clearly better than the release alone on model-written circuits (0.73-0.92).

> **Previous release (2026-10-02, second release): `psf_compile.py` 2026-10-02.2, with `psf_smart_layout`
> 2026-10-01.1 and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)**
> ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md), Addenda 306-311). One opt-in addition to 2026-10-02.1:
>
> - **`compile_for_hardware(..., target=backend.target, placement_refine=True)` places the routed circuit by the
>   device's gate errors.** After PSF-Zero's own layout and routing, the circuit is re-placed with the step Qiskit
>   level 3 ends with (`VF2PostLayout` scored on the exact error of each gate as placed). Only physical qubits are
>   relabelled: gate counts and depth are unchanged. Pre-registered test (Addenda 309-310, fake devices, noisy
>   simulation, 693 circuits per device, 2,079 in all):
>   - better than 2026-10-02.1 in all 18 family-device cells (mean infidelity 0.57-0.98 times);
>   - on chains, equal to Qiskit level 3 with the Target on FakeTorino and FakeKingston (0.998, 1.000);
>   - failed couplers never used, and no recompile around them needed;
>   - about 1 ms extra per compile.
> - **Known gaps** (the first largely closed by 2026-10-03.1: chains 1.01 on FakeAuckland, Addendum 324):
>   - On FakeAuckland, whose simulated errors include a T1/T2 floor above the reported errors, a gap to level 3
>     remains (chains 1.10; Addendum 308).
>   - Where PSF-Zero's routing uses more two-qubit gates than level 3, that difference remains.
>   - Readout is not part of the score: the circuits tested have no measurements.
> - **Why not Qiskit's own layout stage:** handing placement to Qiskit's level-1 layout stage was tested and not
>   adopted (Addendum 307). That stage ranks qubits by an average that mixes in readout error (Addendum 308).
> - Without `placement_refine`, nothing changes.

> **Previous release (2026-10-02.1): `psf_compile.py` 2026-10-02.1, with `psf_smart_layout` 2026-10-01.1
> and the Rust core `CORE_VERSION` 2026-09-29.1 (both unchanged)** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 302-305). One opt-in addition to 2026-10-01.1:
>
> - **`compile_for_hardware(..., target=backend.target)` avoids failed couplers and qubits.** A device can list a
>   coupler whose reported error is 1.0; the layout search and routing see only the coupling map, so the previous
>   release could route through it (7-14 times per circuit for a 6-qubit ring on FakeTorino, Addendum 302). With
>   `target`, a compiled circuit that uses such an element is recompiled on the coupling map without failed
>   couplers and qubits; every other result is returned unchanged. Pre-registered test (Addenda 303-304, fake
>   devices, noisy simulation): failed elements never used; 1,926 of 1,926 unaffected circuits bit-for-bit
>   unchanged; all 153 affected circuits improved (mean infidelity 0.956 to 0.375). Without `target`, nothing
>   changes.
> - **Known gap, addressed by 2026-10-02.2:** the layout search ignores gate errors. In the same tests the
>   release's noisy infidelity is 1.03-1.70 times that of Qiskit level 3 with the device Target in every circuit
>   family and device tested (Addenda 301 and 304).

> **Update (2026-10-01) -- noise-model estimates and an AI front end (prototype)** (Addenda 275-289).
> Noisy simulation with Qiskit Aer and the fake devices' published calibration; not real hardware.
>
> - **Fewer two-qubit gates did mean higher fidelity in this model:** in about 92% of same-circuit
>   pairs the compile with fewer two-qubit gates had the higher fidelity, and infidelity fell by
>   38-46% from the previous release to the front end below (Addendum 281).
> - **A front end for model-written circuits** (then `benchmarks/psf_ai_compile.py`, now kept as
>   `benchmarks/psf_ai_compile_a5.py`; prototype 2026-10-01.a5, separate from `psf_compile.py`) tries several placements and decompositions and,
>   given a device Target, keeps the candidate with the lowest estimated error, estimating each
>   gate's error from the circuit's ideal state at that point. Mean infidelity on 40 held-out random
>   circuits per device: 0.0601 against Qiskit L3 with error-aware layout 0.0707 (FakeAuckland),
>   0.0422 against 0.0451 (FakeTorino), 0.0192 against 0.0205 (FakeKingston). 30,000 repeated
>   compiles with the calibration changed every 1,000 showed no drift in output, time or memory
>   (Addenda 288-289).
> - **Caveat:** the estimate uses the same physics as the simulator that scores it, so these wins
>   are partly built in. On FakeAuckland some qubits' published two-qubit errors are below the bound
>   their own T1/T2 imply, which misled every error-aware method there. Whether real devices show
>   this, and whether the gains survive on hardware, is not yet tested.

> **Previous release (2026-10-01): `psf_compile.py` 2026-10-01.1, `psf_smart_layout` 2026-10-01.1
> and the Rust core `CORE_VERSION` 2026-09-29.1 -- rebuild the core (`maturin develop --release`)
> when you update** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md), Addenda 272-274; core:
> Addenda 250-251). Pre-registered on held-out inputs in the workplace sandbox (2 CPUs, fake backends;
> times are sandbox times) and adopted by the owner with 13 of 14 predictions confirmed and one
> ambiguous (a different, equally swap-free layout in one tiling).
>
> - **Layout:** the 2026-09-29 candidate (corrected feasibility check, short-path shortcut) plus an
>   exact packing search for disjoint 2- and 3-qubit paths. All 10 feasible held-out heavy-hex
>   tilings were placed swap-free within 1 s, which closes the layout gap reported in the 2026-09-29
>   update below. Four large mixed tilings on FakeTorino and FakeKingston remain unplaced by both
>   versions; whether they are feasible is unknown.
> - **Compile (`entangling_basis="cx"` only):** short blocks are consolidated when that saves CX
>   gates, SWAPs are removed by relabelling, and routed SWAPs are re-synthesised with their
>   neighbours. On 90 held-out dense random circuits per device, two-qubit gates fell from 1,905 to
>   1,147 (FakeAuckland) and from 2,013 to 1,168 (FakeKingston), none worse; the ratio to Qiskit L3
>   went from 1.71-1.79 to 1.03-1.04. The canonical basis is unchanged bit for bit. Larger circuits
>   got no more gates; compile time rose by at most 1.29x on PSF-Zero's favourable families and
>   1.51x on random circuits.
> - **Core 2026-09-29.1:** an eigen-route fallback, with output identical to 2026-09-28.1 wherever
>   that core succeeded and no fallbacks in the 100,000-compile run. Every evaluation since
>   2026-09-29 used it.

> **Update (2026-09-30) -- 100,000 compounded laps, and a first recorded test of a
> language-model front end** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md),
> Addenda 258-270). All pre-registered except three pilots marked exploratory;
> fake backends, no QPU; the language-model tests ran on rented RunPod GPUs.
>
> - **100,000 compounded PennyLane laps on a fully occupied heavy-hex device**
>   (FakeAuckland, 27 qubits) with the candidate stack (core 2026-09-29.1, layout
>   2026-09-29.c1): every lap within 1 s (median 13 ms), swap-free, 0 core fallbacks;
>   drift linear in laps (2.95e-10 at lap 100,000), memory flat. Run at home; its
>   first 30,000 laps equal an earlier pod run lap by lap to the printed digits, on a
>   different GPU, CPU and build (Addenda 258-262). The candidates are still not the
>   release.
> - **vLLM x PSF-Zero, on the record with a stop-loss rule.** An open model served by
>   vLLM writes a circuit for a described target state, PSF-Zero compiles it, and the
>   model gets state feedback for up to 6 rounds; five tasks, pre-registered go/no-go
>   criteria.
>   - First test (B200 pod; Qwen2.5-7B, Qwen2.5-72B, gpt-oss-120b): **CUT**. The best
>     model solved 12/15, but the W state on 3 qubits 0/3: every one of its W3 replies
>     used the whole 16,000-token budget on reasoning (Addenda 263-264).
>   - Three exploratory pilots then tuned the harness on W3 (Addenda 265-267), and a
>     new pre-registration tested it (H200 pod): **INVEST, at the smallest possible
>     margin**. gpt-oss-120b solved 14/15 with W3 at 2/3 against a bar of 2/3
>     (Addenda 268-269); Qwen2.5-7B did not improve (7/15). This means the line is
>     worth further testing, not that it works reliably. Model time was 5,937 s summed
>     over the 15 task-runs, against under a second of PSF-Zero compile time.
>   - On five held-out tasks that no model had seen, the same harness then solved 15/15
>     (W4, Dicke(4,2), a phased GHZ, three singlets, nine GHZ-3 states on 27 qubits);
>     a variant with three candidates per round and verbal feedback solved 14/15 at
>     three times the tokens and was not adopted (Addenda 270-271). Three runs per task;
>     three of the five tasks are close relatives of tuned ones.
> - **PSF-Zero inside that loop.** On the correct 27-qubit circuits the models wrote
>   (FakeAuckland filled with GHZ-3 states and Bell pairs), PSF-Zero compiled in
>   10-21 ms against Qiskit L3's 5.9-11.1 s (500-730x), with 17 two-qubit gates
>   against 20, in both tests (B200 and H200 pods; times are not compared across
>   them). A new tiling of nine GHZ-3 states does not take the candidate layout's
>   short-path route (workplace sandbox, 2 CPUs: 1.2-1.3 s against L3's 8.4-8.8 s);
>   the held-out evaluation confirmed it on the pod (1.2 s against 8.0 s, Addendum 271),
>   and it is the next layout improvement to pre-register.

> **Update (2026-09-29) -- heavy-hex, a layout gap in the current release, and a Qiskit
> angle cutoff** ([Part 9](../docs/findings/spare-qubit-cliff-combined-248.md), Addenda
> 248-259; notes on Addenda 246-247 in Part 8). All pre-registered; compile-only, fake
> backends, no QPU.
>
> - **A fully occupied heavy-hex device does show the cliff.** Pairs alone cannot fill a
>   heavy-hex graph (it has no perfect matching), which is why earlier heavy-hex tests
>   saw no cliff. Filling FakeKingston (156 qubits) with pairs *and* 3-qubit paths,
>   Qiskit `optimization_level=3` took a median 24-28 s at 0-8 spare qubits (0.28 s at
>   16) and added 18-39 two-qubit gates, although a swap-free layout exists.
> - **Known gap in the current release.** On those circuits PSF-Zero's layout search
>   (`psf_smart_layout` 2026-09-26.m1) wrongly reports "no layout" and falls back to
>   Qiskit's level-1 layout: fast (0.28 s) but with 45-51 more two-qubit gates than a
>   swap-free layout, more than Qiskit L3. The output is still correct (whole-circuit
>   GPU checks on a 27-qubit heavy-hex model, including SWAPs: <= 4.9e-14). Circuits
>   made of disjoint pairs are not affected. A candidate fix (2026-09-29.c1, in
>   [`patches/`](../patches/)) places them swap-free in about 0.21 s and leaves pair-only
>   outputs unchanged; it is being checked at home before it becomes the release.
> - **Qiskit's `CommutativeCancellation` drops merged Z rotations smaller than
>   4 pi x 1e-5 (about 1.26e-4 rad)**, a fixed cutoff that `approximation_degree` does
>   not change (Qiskit 2.5.2; also on main as of 2026-09-28). PSF-Zero's default path
>   (routing level 1) never runs that pass and stayed exact; passing
>   `routing_optimization_level=3` runs it and accepts the same cutoff (operator
>   errors up to about 1.3e-4 on the affected pair).
> - **A candidate Rust core (2026-09-29.1)** removes the last 15 core fallbacks of the
>   100,000-compile run (0 fallbacks, 0 repairs) and is bit-identical to the release
>   core on the 394,988 blocks where that core succeeds. Also in `patches/`, not yet
>   the release.

> **Previous release: `psf_compile.py` 2026-09-28.1 with the Rust core `CORE_VERSION`
> 2026-09-28.1 -- rebuild the core (`maturin develop --release`) when you update.** The
> core now extracts the two single-qubit factors of each block from their best-conditioned
> quaternion products; before, it failed (and Python fell back to Qiskit's synthesis) on
> about 13% of the blocks of a brick-layer training circuit, and was off by up to 8e-7 on
> a few blocks without failing. `REFINE_THRESHOLD` is now 1e-14, so that small residuals
> the new core leaves are polished instead of adding up when a circuit is compiled again
> and again. A pre-registered 100,000-compile run at home: fallbacks 43,829 -> 15, the
> compounding drift 8.33e-10 -> 1.22e-10, every checked compile exact, compile time 0.90x
> (training circuit) and 1.08x (120-qubit cliff circuit) of the previous release
> (Addenda 223-244). Outputs are equivalent to the previous release's but not
> bit-identical. The default `block_gate_floor` is 8; every emitted two-qubit block is
> checked by phase-aligned operator distance (1e-13), as since 2026-09-27.6.

> **Correctness notice (2026-09-26) -- if you use `entangling_basis="cx"`, update to
> `psf_compile.py` VERSION 2026-09-26.4 or later.** Qiskit's own
> `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` (Qiskit 2.5.2) returns a
> wrong circuit -- average gate infidelity about 7% -- for two-qubit unitaries whose
> smallest canonical coordinate lies roughly between 3e-8 and 3e-7; plain
> `transpile()` of a `UnitaryGate` to `basis_gates=["cx","rz","sx","x"]` fails the
> same way at every optimization level (a CZ basis, or a backend target with error
> data, did not). Every earlier PSF-Zero revision passed such blocks through
> unchecked on the `"cx"` path, and `verify=True` could not catch it (it checks the
> decomposition, not the emitted circuit). Random circuits essentially never reach
> this band; circuits with very small two-qubit interaction angles can. VERSION
> 2026-09-26.4 now checks every block it takes from Qiskit's decomposer and repairs
> it (48 of 272 near-degenerate test blocks were wrong without the check; the worst
> with it is 1.6e-12). Minimal Qiskit-only reproduction:
> [`benchmarks/repro_qiskit_zsx_2q_v2.py`](../benchmarks/repro_qiskit_zsx_2q_v2.py);
> full account: [`docs/findings/spare-qubit-cliff-combined-135.md`](../docs/findings/spare-qubit-cliff-combined-135.md),
> Addenda 194-197. Reported upstream as
> [Qiskit issue #17057](https://github.com/Qiskit/qiskit/issues/17057), after
> reproducing it on the latest release (2.5.2) in a fresh environment.
