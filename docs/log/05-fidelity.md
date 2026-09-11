> **Archived record — part 5 of 6.**
> This file is one slice of the original PSF-Zero README, kept **verbatim**. It is a
> lab notebook: it contains claims that were later retracted, hypotheses that were
> later refuted, and corrections written directly underneath the text they correct.
> Nothing has been deleted. For what the project currently claims, read the
> [README](../../README.md); for a settled account of one topic, read
> [`docs/findings/`](../findings/).
>
> Sections 7-8 - real IBM hardware fidelity at 15 qubits, the four-engine noisy-simulator comparison, and both root-cause threads behind the fidelity gap (`compile_for_hardware()`'s missing `basis_gates`, and the native-gate cost of RXX/RYY/RZZ).
>
>
> **One edit was made to this file when it was archived:** file-relative image paths
> were rewritten from `./docs/...` to `../../docs/...` for the new directory depth.
> Root-relative paths (`/docs/...`) were left untouched — they resolve correctly at
> any depth. No wording, number, claim or correction was altered. The 13 substitutions
> are listed in [`README.md`](README.md#the-one-edit).
> Source: README.md lines 1953-2607, as of 2026-09-11.

---

### 7. Real-device fidelity validation (15 qubits, corrected `ConsolidateBlocks`)

With the `ConsolidateBlocks` bug fixed, we re-ran the 15-qubit real-hardware
comparison referenced above — this time 10 independent job submissions instead
of one, split across two IBM backends (`ibm_marrakesh`, `ibm_fez`). Every run
logged `105/105` blocks synthesized with 0 fallbacks, confirming the fix is
exercising the intended code path rather than silently no-oping the way the
earlier, retracted single-run numbers did.

![![Native synthesis vs. TKET by scale: compile time and output depth](../../docs/090304.png)](../../docs/090401.png)

| Metric | Qiskit (L3) | PSF-Zero |
| :--- | :---: | :---: |
| Fidelity, mean ± SD (n=10) | 0.0925 ± 0.0016 | 0.0919 ± 0.0016 |
| Compile time, mean ± SD | 2.36s ± 0.09s | 0.153s ± 0.007s |
| Circuit depth, mean ± SD | 730 ± 70 | 710 ± 69 |
| 2Q gate count, mean ± SD | 644 ± 14 | 641 ± 11 |

PSF-Zero's output had higher fidelity than Qiskit's in 3 of the 10 runs; a
paired comparison across all 10 gives t = -0.78, which is not significant — on
this circuit and these two backends, we can't say PSF-Zero's real-hardware
output is either better or worse than Qiskit L3's. Depth and 2Q gate count
were a similar wash (PSF-Zero shorter/fewer in 5 of 10 runs each, in both
cases). The one result that held up cleanly on every single run was compile
time: PSF-Zero compiled 14.4x–16.2x faster than Qiskit L3 across all 10 jobs.

> **Caveat added after section 4's correction:** "consistent with the
> unconstrained-circuit results above" no longer holds — section 4's
> equivalent claim was retracted after we found it was dominated by
> measurement artifacts, including a per-process `transpile()` cold-start
> cost that this script's single-call-per-run structure could plausibly
> reproduce here too (each of these 10 runs is its own process, and
> `transpile()`/`compile_for_hardware()` are each called exactly once per
> run, so neither side benefits from a prior warm-up call the way section
> 4's corrected numbers now do). Unlike section 4, we can't just re-run this
> one with a warm-up patch — it submits real jobs to IBM hardware, and we're
> not spending real QPU time re-verifying a compile-time number without
> first checking whether the artifact applies here. Real device transpile at
> `optimization_level=3` with full routing against a ~127+-qubit backend is
> also inherently heavier than section 4's unrouted `compile()` call, so this
> number may hold up even after a warm-up fix — but we have not verified
> that, and are flagging it rather than repeating the "consistent with
> section 4" framing now that section 4 itself changed. See Roadmap.

Job IDs, in run order (for reproducibility): `daclrrjdd5gc73d68pcg`,
`dacls9e42tqs73asccbg`, `daclsstnj4cs73acqm00`, `daclu3bdd5gc73d68rs0`,
`dacluq5nj4cs73acqo70`, `daclv3m42tqs73ascfeg`, `daclvgrdd5gc73d68thg`,
`daclvre42tqs73ascgbg`, `dacm0gtnj4cs73acqq6g`, `dacm0r642tqs73aschqg`.

Code: [`benchmarks/real_device_15q_fidelity_v2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/real_device_15q_fidelity_v2)

> **Provenance note:** the listing below was reconstructed from the captured
> run log (the same log the job IDs above come from). The original file was
> searched for across the working repository (`findstr` for `transpile`,
> `Sampler`, `real_device`, `fidelity` across every `.py` file present) and
> not found — it appears to be lost, not just unexamined, and the same goes
> for `test_real_hardware_fidelity.py` (section 8). This is not guaranteed
> to be a byte-for-byte match of whatever the original was; the parameters
> (15 qubits, seed=42), control flow, and all Japanese print statements
> match the log exactly. If the real file resurfaces, replace this listing
> with it.

```python
"""
real_device_15q_fidelity_v2.py

Compares Qiskit's optimization_level=3 transpile against PSF-Zero's KAK-based
compile on a 15-qubit QuantumVolume circuit, submitted as one batched job to
a real IBM backend.
"""
import time

from qiskit import transpile
from qiskit.circuit.library import QuantumVolume
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from psf_compile import compile as psf_compile

NUM_QUBITS = 15
SEED = 42
SHOTS = 4096


def count_2q_gates(circuit):
    return sum(1 for instr in circuit.data if instr.operation.num_qubits == 2)


def classical_fidelity(counts, shots, ideal_probs):
    """Hellinger-style overlap between a measured count dict and the ideal
    probability distribution: (sum_i sqrt(p_ideal_i * p_meas_i))**2."""
    fid = 0.0
    for bitstring, p_ideal in ideal_probs.items():
        p_meas = counts.get(bitstring, 0) / shots
        fid += (p_ideal * p_meas) ** 0.5
    return fid ** 2


def main():
    print("Connecting to IBM Quantum Cloud...")
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Connection established. Using real QPU: {backend.name}")

    print(f"Generating a large entangled {NUM_QUBITS}-qubit circuit...")
    base_circuit = QuantumVolume(num_qubits=NUM_QUBITS, depth=NUM_QUBITS, seed=SEED).decompose()

    n_2q = count_2q_gates(base_circuit)
    print(
        f"-> Number of 2-qubit UnitaryGates after decompose(): {n_2q} "
        f"(If 0, PSF-Zero found no target blocks; please abort execution)"
    )

    print("Calculating the ideal probability distribution (ground truth) classically...")
    ideal_probs = Statevector(base_circuit).probabilities_dict()

    print("[1/2] Executing compilation with Qiskit (Level 3)...")
    t0 = time.perf_counter()
    qc_qiskit = transpile(base_circuit, backend=backend, optimization_level=3)
    t_qiskit = time.perf_counter() - t0
    print(f"-> Done. Qiskit processing time: {t_qiskit:.2f} seconds")

    print("[2/2] Executing compilation with PSF-Zero...")
    t0 = time.perf_counter()
    qc_psf = psf_compile(base_circuit, backend=backend)
    t_psf = time.perf_counter() - t0
    print(f"-> Done. PSF-Zero processing time: {t_psf:.2f} seconds")

    print("=== Compilation Results Comparison ===")
    print(
        f"[Qiskit] Time: {t_qiskit:.2f}s | Depth: {qc_qiskit.depth()} | "
        f"2Q Gates: {count_2q_gates(qc_qiskit)}"
    )
    print(
        f"[PSF-Zero] Time: {t_psf:.2f}s | Depth: {qc_psf.depth()} | "
        f"2Q Gates: {count_2q_gates(qc_psf)}"
    )

    print("Submitting job to the real device (QPU)...")
    sampler = Sampler(backend)
    job = sampler.run([qc_qiskit, qc_psf], shots=SHOTS)
    print(f"Job submitted successfully! Job ID: {job.job_id()}")
    print("Waiting for real device execution (this may take several minutes)...")
    result = job.result()

    counts_qiskit = result[0].data.meas.get_counts()
    counts_psf = result[1].data.meas.get_counts()

    fid_qiskit = classical_fidelity(counts_qiskit, SHOTS, ideal_probs)
    fid_psf = classical_fidelity(counts_psf, SHOTS, ideal_probs)

    print("===================================")
    print("Physical Real-Device Fidelity Comparison")
    print("===================================")
    print(f"Qiskit Level 3 : {fid_qiskit:.4f}")
    print(f"PSF-Zero       : {fid_psf:.4f}")
    print("===================================")
    print("[NOTE] Fidelity differences from a single run may fall within shot noise bounds.")
    print("It is strongly recommended to run this script multiple times (e.g., n_repeats >= 10)")
    print("and compare the mean +/- standard deviation (do not draw conclusions from a single run).")


if __name__ == "__main__":
    main()
```

**Update:** a captured log of `test_real_hardware_fidelity.py` actually being
run with a `--real` flag — i.e. against real IBM hardware, not the local
`fake_sherbrooke` snapshot below — has since turned up (11 runs, job IDs
`dadb...`). This should be read as *qualifying, not replacing* the 10-run
capture above; as with the note above, we have the run log but not a
confirmed copy of the script that produced it.

| Metric | Qiskit L3 | PSF-Zero |
| :--- | :---: | :---: |
| Fidelity, mean ± SD (n=11) | 0.0916 ± 0.0021 | 0.0909 ± 0.0017 (t = -0.68, n.s.) |
| Circuit depth, mean ± SD | 738.5 ± 47.2 | 685.0 ± 59.2 |
| 2Q gate count, mean ± SD | 648.0 ± 9.8 | 641.7 ± 17.3 |
| Compile time, mean ± SD | 2.107s | 0.159s (13.3x faster) |

![Real-device 15-qubit fidelity validation, 11 runs, corrected ConsolidateBlocks](../../docs/real_device_15q_fidelity_v3_1.png)

### 8. Fidelity across engines under a realistic noise model (mirror circuits)

**Update:** a captured log of `test_real_hardware_fidelity.py` actually being
run with a `--real` flag against real IBM hardware has since turned up (4
sweeps: 3 on `ibm_marrakesh`, 1 on `ibm_fez`). Same caveat as above: we have
the run log but not a confirmed copy of the script that produced it.

Using Qiskit's `fake_sherbrooke` (127-qubit) noise model as a local
noisy-simulator snapshot, we ran mirror circuits (which should return
all-zero with probability ~1.0 in the noiseless case — confirmed separately
for all four engines before the noisy runs below) across three circuit
families of increasing two-qubit depth per pair — `deep2q` (3 gates),
`multi_deep2q` (12 gates), and `wide` (42 gates) — for Qiskit L3, TKET
(native), PSF-Zero v6, and the Hybrid (PSF→TKET) pipeline, 5 repeats each:

| Family (2Q gates/pair) | Qiskit L3 | TKET (native) | PSF-Zero v6 | Hybrid |
| :--- | :---: | :---: | :---: | :---: |
| deep2q (3) | 0.9056 ± 0.0025 | 0.9077 ± 0.0031 | **0.8638 ± 0.0020** | 0.9076 ± 0.0031 |
| multi_deep2q (12) | 0.0849 ± 0.0015 | 0.0863 ± 0.0028 | **0.0720 ± 0.0012** | 0.0839 ± 0.0021 |
| wide (42) | 0.0033 ± 0.0004 | 0.0039 ± 0.0005 | 0.0044 ± 0.0007 | 0.0037 ± 0.0006 |

(mean P(all-zero) ± standard error, n=5)

In `deep2q` and `multi_deep2q`, PSF-Zero v6 was the lowest-fidelity engine of
the four by a margin well outside the standard errors shown — roughly 4
percentage points below Qiskit L3 on `deep2q` and about 1.3 points below on
`multi_deep2q` — which reads as a real, repeatable effect on this circuit
family rather than run-to-run noise. In `wide`, all four engines are already
near the noise floor (under 0.5% success), and PSF-Zero v6's slightly higher
mean there isn't distinguishable from the others at this sample size; we
don't read anything into it either way.

Code: [`benchmarks/test_real_hardware_fidelity.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_real_hardware_fidelity.py)
(despite the filename, this specific table is from the local `fake_sherbrooke`
noisy-simulator snapshot, not real hardware — real-hardware results are in
section 7 above)

#### Real-hardware confirmation (new)

The captured `--real` log runs the identical script against real backends:
same three families, same `block_gate_floor`-driven block counts per family
(`deep2q`: 1/1 blocks, `multi_deep2q`: 4/4 blocks, `wide`: 0/0 blocks — 0
fallbacks in every case, confirming this is the same circuit construction as
the table above, not a different one), same `mean_two_qubit_gates` per
family (3 / 12 / 42, matching the table above exactly), 5 repeats × 4
engines, batched as one job per sweep. Four independent sweeps were
captured: three against `ibm_marrakesh`, one against `ibm_fez`.

![Real compile_for_hardware(), old vs. patched: fidelity and native ecr gate count by family](/docs/section8_real_hw_vs_sim.png)

| Family | Engine | Real hardware, mean ± sd (4 sweeps) | `fake_sherbrooke` (for reference) |
| :--- | :--- | :---: | :---: |
| deep2q | Qiskit_L3 | 0.99545 ± 0.00321 | 0.9056 |
| deep2q | TKET_native | 0.99524 ± 0.00427 | 0.9077 |
| deep2q | PSF_Zero_v6 | 0.99517 ± 0.00382 | 0.8638 |
| deep2q | Hybrid | 0.99514 ± 0.00435 | 0.9076 |
| multi_deep2q | Qiskit_L3 | 0.96148 ± 0.00762 | 0.0849 |
| multi_deep2q | TKET_native | 0.96157 ± 0.00652 | 0.0863 |
| multi_deep2q | PSF_Zero_v6 | 0.96182 ± 0.00720 | 0.0720 |
| multi_deep2q | Hybrid | 0.96149 ± 0.00682 | 0.0839 |
| wide | Qiskit_L3 | 0.96113 ± 0.00769 | 0.0033 |
| wide | TKET_native | 0.96168 ± 0.00442 | 0.0039 |
| wide | PSF_Zero_v6 | 0.96190 ± 0.00668 | 0.0044 |
| wide | Hybrid | 0.96174 ± 0.00684 | 0.0037 |

("sd" here is the spread across the 4 sweep means, not the within-sweep
standard error — with only 4 sweeps this is a rough number, not a tight
confidence interval.)

Two findings, and — at the time this was first written — they appeared to
point in different directions. Section 8's own follow-up investigation
below has since substantially explained both, and reframed how they relate
to each other; the original framing is kept here for the record, with the
resolution below it.

**Finding 1 — no PSF-Zero-specific deficit on real hardware.** In every
family, the four engines' real-hardware means sit within about 0.001 of each
other, far tighter than the sweep-to-sweep spread (0.003–0.008). PSF-Zero's
rank among the four engines bounces around from sweep to sweep — 4th, 4th,
1st, 2nd on `deep2q`; 2nd, 4th, 1st, 4th on `multi_deep2q`; 2nd, 4th, 1st, 2nd
on `wide` — which looks like noise, not a systematic effect. The `deep2q`
deficit that `fake_sherbrooke` predicted for PSF-Zero specifically does not
show up here: on real hardware, across four independent sweeps on two
backends, we cannot distinguish PSF-Zero from the other three engines.

**Finding 2 — the real-hardware numbers are dramatically higher than
`fake_sherbrooke` predicted, for the identical circuits, and we do not yet
know why.** This is not a small correction. On `deep2q` all four engines
land noticeably above their `fake_sherbrooke` counterparts (~0.995 vs.
~0.86–0.91), which could plausibly be "the simulator is a bit pessimistic."
But on `multi_deep2q` and especially `wide`, the gap is not a few points —
it's close to two orders of magnitude (`wide`: ~0.96 on real hardware vs.
~0.003–0.004 predicted by `fake_sherbrooke`, for a circuit `fake_sherbrooke`
itself put "near the noise floor"). Candidates considered: `fake_sherbrooke`'s
noise snapshot being more pessimistic than either backend's current
calibration; a parameter difference between the local and real runs we
couldn't see without the actual script; or something about how
`P(all-zero)` is computed differing between the two paths.

Job/sweep provenance: 3 sweeps against `ibm_marrakesh` (156 qubits), 1 against
`ibm_fez` (156 qubits), captured 2026-09-04. Individual job IDs were not
retained for this batched-submission script (unlike section 7's per-run job
IDs) — each sweep submits one batched job of 20 circuits per family.

#### A leading (not yet fully confirmed) hypothesis for the gap

`psf_compile.py` v6's synthesizer builds each block from four local
single-qubit triples plus up to three entangling gates — but those
entangling gates are `RXX`/`RYY`/`RZZ`, not `CX`. Neither is native to real
IBM hardware (`fake_sherbrooke`'s native basis is `ecr`/`rz`/`sx`/`x`), but
we suspected they might not translate to that basis as cheaply as `CX` does.

[`benchmarks/diagnose_native_gate_inflation.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/diagnose_native_gate_inflation.py)
tests this directly: it builds the same canonical KAK circuit structure
PSF-Zero v6 emits (via Qiskit's own `TwoQubitWeylDecomposition`, since we
don't have a working build of `psf_zero_core` in every environment) for 200
random SU(4) unitaries, and a CX-basis decomposition of the same unitaries,
then transpiles both to `fake_sherbrooke` at each optimization level and
counts native `ecr` gates (0 correctness failures at any level):

| `optimization_level` | RXX/RYY/RZZ basis (PSF-Zero-like) | CX basis (Qiskit L3/TKET-like) | Ratio |
| :---: | :---: | :---: | :---: |
| 0 | 6.00 ECR | 3.00 ECR | 2.00x |
| 1 | 6.00 ECR | 3.00 ECR | 2.00x |
| 2 | 3.00 ECR | 3.00 ECR | 1.00x |
| 3 | 3.00 ECR | 3.00 ECR | 1.00x |

![Native ECR gate count after transpiling RXX/RYY/RZZ-basis vs. CX-basis circuits to fake_sherbrooke, by optimization level](../../docs/090404.png)

At `optimization_level` 0-1, the RXX/RYY/RZZ-based circuit costs exactly 2x
as many native `ecr` gates as the CX-based one for the identical unitary —
invisible to any benchmark that counts 2-qubit gates on the pre-ISA-transpile
circuit (as section 8's own `mean_two_qubit_gates` column does, which is why
it shows 3.0 for every engine). At `optimization_level` >= 2, Qiskit's
transpiler resynthesizes 2-qubit blocks from scratch regardless of input
basis, and the gap vanishes.

**Follow-up, after actually finding `compile_for_hardware()` in the real
`psf_compile.py`** (the function section 5's `test1.py` calls for
hardware-targeted output):

```python
def compile_for_hardware(qc, coupling_map, block_gate_floor=..., routing_optimization_level=0):
    qc_compressed = compile(qc, block_gate_floor=block_gate_floor)
    return transpile(qc_compressed, coupling_map=coupling_map,
                      optimization_level=routing_optimization_level)
```

[`benchmarks/diagnose_compile_for_hardware.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/diagnose_compile_for_hardware.py)
tests this exact call signature directly and shows this transpile call does
**not** decompose RXX/RYY/RZZ at any `routing_optimization_level` (0-3) —
because only `coupling_map` is given, with no `basis_gates`/`backend`, so
Qiskit only does layout and routing, never a target-basis resynthesis. So
`compile_for_hardware()`'s own output is not actually real-hardware-
submittable as written — it still contains RXX/RYY/RZZ, and IBM Runtime's
`SamplerV2` rejects non-ISA circuits — meaning there must be one more,
currently-unseen transpile-to-ISA step, wherever this output actually gets
submitted to a backend. **That still-missing step, not `compile_for_hardware()`
itself, is where the measured 2x native-gate penalty would apply**, if it
uses a low `optimization_level`.

We can't see that step (it isn't in any file we have), but `compile_for_hardware()`'s
own doc comment gives exactly the reasoning that would lead someone to pick
a low level there too: *"`routing_optimization_level` defaults to 0
(routing only) since `compile()` already did the 2-qubit optimization that
a higher `optimization_level` would otherwise redo."* That's true about
LOGICAL 2-qubit gate count. It's false about PHYSICAL native-gate count
once a real basis has to be targeted — a higher level there doesn't "redo"
work, it does work that was never done. **This is a plausible, now-measured
mechanism, made more likely by the codebase's own established habit of
defaulting to low optimization levels downstream of `compile()` — but it is
still not a confirmed diagnosis of section 7/8's actual pipeline**, since we
don't have `real_device_15q_fidelity_v2.py` / `test_real_hardware_fidelity.py`
to see what their final backend-submission transpile call actually does.
Finding that call is the one remaining check (see Roadmap).

#### Proposed fix, validated (pending confirmation of the actual root cause)

The fix this points to: give `compile_for_hardware()` a `basis_gates`
parameter, thread it through to the internal `transpile(...)` call, and
default `routing_optimization_level` to 2+ so that call actually
resynthesizes to the target basis instead of only routing:

```python
def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map,
    basis_gates: list[str] | None = None,       # new
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 2,        # was 0
) -> QuantumCircuit:
    qc_compressed = compile(qc, block_gate_floor=block_gate_floor)
    return transpile(
        qc_compressed,
        coupling_map=coupling_map,
        basis_gates=basis_gates,                # new
        optimization_level=routing_optimization_level,
    )
```

> **Note (2026-09-08):** `routing_optimization_level`'s default has since
> moved again, from the 2 set here down to 1 — see section 5's 2026-09-08
> update above. Threading `basis_gates` through was the fix that mattered
> here (translation has to actually run somewhere); once it's always
> supplied, level 2's extra re-synthesis over level 1 turned out to buy
> nothing but discarded work, on the dense-pair-block workload that
> follow-up investigation used. This section's own diagnosis and the
> `entangling_basis="cx"` fix below are unaffected by that later change.

One correction to make here, checked directly against the installed Qiskit
(2.5.2): `transpile()`'s default `optimization_level` when left unspecified
is **2**, not 1 — straight from `qiskit.compiler.transpiler.transpile`'s own
source ("Take optimization level from the configuration or 2 as default").
So if the still-missing real-hardware script simply omitted
`optimization_level` rather than setting it explicitly, it would already
have gotten level-2 (no-inflation) behavior in this Qiskit version — the
mechanism only bites if that script explicitly passed `0` or `1` (or ran an
older Qiskit release with a different default). Still not confirmable
without the file itself.

[`benchmarks/verify_compile_for_hardware_fix.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/verify_compile_for_hardware_fix.py)
validates the fix directly: N=50 random SU(4) blocks, each forced onto
non-adjacent qubits on a 6-qubit line (so every trial needs real routing),
basis `['ecr', 'rz', 'sx', 'x']`:

| `routing_optimization_level` | Correctness failures | Mean ECR gates |
| :---: | :---: | :---: |
| 0 | 0/50 | 12.00 |
| 1 | 0/50 | 6.00 |
| 2 | 0/50 | 3.00 |
| 3 | 0/50 | 3.00 |

Correctness (`Operator.from_circuit(...).equiv(...)`, which reads the
transpiled circuit's `layout` to correctly account for the routing
permutation — a naive `Operator(out).equiv(Operator(qc))` gives false
negatives here, since routing legitimately reorders physical qubits) holds
at every level: the fix doesn't break anything. Levels 0-1 are actually
worse here than in the unrouted diagnostic above (4x and 2x, vs. 2x and 2x
there) — once `basis_gates` is supplied, the routing SWAPs themselves also
need decomposing into the target basis, and low optimization levels don't
do that efficiently either. Level 2+ recovers the optimal count (3) even
with routing. **This is a validated, strictly-improving fix to
`compile_for_hardware()` — but it fixes a real bug we found in that
function regardless of whether it turns out to be the actual cause of
section 7/8's fidelity gap**, since that still depends on the one
unconfirmed piece above.

#### Independent reproduction: does the mechanism actually move fidelity?

Gate counts are one thing; section 8's actual claim is about measured
fidelity. [`benchmarks/experiment_fixed_compiler_fidelity.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/experiment_fixed_compiler_fidelity.py)
builds an independent, from-scratch mirror-circuit fidelity test directed
only by the real `psf_compile.py` source (not by section 8's own numbers)
and runs it on an `AerSimulator` noise model derived from `fake_sherbrooke`
— since the actual scripts that produced section 8 are lost, this cannot
be "re-running the real benchmark," but it is a working, from-first-
principles check of whether the mechanism found above actually behaves the
way section 8's numbers imply.

Three engines (a from-scratch `psf`-style compiler using the same KAK/
RXX-RYY-RZZ structure as `psf_compile.py` v6, plus `qiskit` L3 and `tket`
baselines) each compile the same `deep2q`/`multi_deep2q`/`wide`-style
blocks, a mirror circuit is built from each engine's own compiled output,
and then ONE common final backend-transpile is applied — standing in for
the still-missing real-hardware harness's last step — at a naive low level
(1) and a good level (3):

| Family | Engine | `optimization_level=1` | `optimization_level=3` |
| :--- | :--- | :---: | :---: |
| deep2q | psf | 0.8725 ± 0.0093 | 0.9925 ± 0.0026 |
| deep2q | qiskit | 0.9193 ± 0.0050 | 0.9930 ± 0.0018 |
| deep2q | tket | 0.9163 ± 0.0081 | 0.9916 ± 0.0039 |
| multi_deep2q | psf | 0.1438 ± 0.0047 | 0.9641 ± 0.0047 |
| multi_deep2q | qiskit | 0.1834 ± 0.0052 | 0.9651 ± 0.0017 |
| multi_deep2q | tket | 0.1702 ± 0.0167 | 0.9666 ± 0.0066 |
| wide | psf | 0.0405 ± 0.0042 | 0.9373 ± 0.0046 |
| wide | qiskit | 0.0438 ± 0.0076 | 0.9388 ± 0.0045 |
| wide | tket | 0.0435 ± 0.0032 | 0.9315 ± 0.0027 |

(mean ± stdev of P(all-zero); N=5 seeds for deep2q/multi_deep2q, N=3 for
`wide`, 2048 shots each)

![Reconstructed mirror-circuit fidelity by engine and family, naive final step at optimization_level 1 vs. 3](../../docs/allzero_by_family_sem.png)

At the naive low level, `psf` trails `qiskit`/`tket` by a real,
stdev-exceeding margin on `deep2q` (~4.5 points) and `multi_deep2q` (~3-4
points) — and `deep2q`'s gap closely matches section 8's own reported
numbers in both direction and rough size (PSF_Zero_v6 0.8638 vs. Qiskit_L3
0.9056 / TKET_native 0.9077, a ~4.2-4.4 point gap). At the good level, all
three converge on every family. On `wide` — where `psf`'s own
`block_gate_floor` logic leaves the circuit completely untouched (its
compiled op count matches the original circuit exactly) — there is no
`psf`-specific gap at either level, matching section 8's own observation
that PSF_Zero_v6 wasn't disadvantaged there. Separately, applying the
validated `basis_gates` fix to `psf`'s output immediately (rather than
relying on the later naive step) recovers most — not quite all — of the
gap under a subsequent low-level final step (`deep2q`: 0.8716 -> 0.9316),
confirming the fix helps even when what happens afterward is out of its
control.

This is the strongest evidence obtainable without the actual lost scripts:
an independent reproduction, built from nothing but the real source code,
that reproduces both section 8's qualitative pattern (gap on
`deep2q`/`multi_deep2q`, no gap on `wide`) and, for `deep2q`, its
approximate quantitative size. **It is still not section 7/8's own
benchmark re-run** — the fidelity numbers above come from a hand-built
stand-in circuit family, not the original one — so we're calling this
strong independent corroboration, not confirmation.

This script was also independently re-run, unmodified, on a second,
separate machine and Qiskit environment (a local Windows `venv`, distinct
from the sandbox that produced the table above). Every value it reported
matched the table above within run-to-run stdev — e.g. `deep2q`/`psf`/
`optimization_level=1`: 0.8760 ± 0.0021 there vs. 0.8725 ± 0.0093 here;
`multi_deep2q`/`psf`/level 1: 0.1535 ± 0.0082 vs. 0.1438 ± 0.0047 — and the
same qualitative pattern (a real gap on `deep2q`/`multi_deep2q` at level 1
that closes at level 3, no gap on `wide` at either level) held in both
runs. This doesn't change what the experiment is (still a stand-in circuit
family, not section 7/8's own script), but it does rule out the result
being an artifact of this one sandbox's environment or random seed.

#### Applying the fix to the real code, end to end

Everything above tests the mechanism using a from-scratch stand-in
compiler. This test is different: it calls the ACTUAL real
`compile_for_hardware()` function — the exact code the real repository
contains (as pasted into this project), before and after the validated fix
— and the real `compile()` / `SU4GeodesicPSFSynthesizer` block-processing
logic around it, from a reference copy of `psf_compile.py` we have in
full. The one substitution is the Rust core itself: the `.so` we were given
won't load in this environment (wrong architecture), so `geometric_decompose()`
is served by a verified stand-in
([`psf_zero_core_stub.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_zero_core_stub.py),
worst-case (1 − fidelity) = 8.88e-16 over 200 trials, matching the real
core's own claimed order of magnitude — see
[`test_psf_zero_core_stub.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_psf_zero_core_stub.py)).

Full provenance is in
[`psf_compile_patched.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/psf_compile_patched.py)'s
own header — including the caveat that we don't have the user's complete
real "v6" file, only `compile_for_hardware()` itself plus a slightly older
full reference copy (v3) of everything around it.

[`test_improved_compiler_end_to_end.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_improved_compiler_end_to_end.py)
builds the same `deep2q`/`multi_deep2q`/`wide`-style blocks as above, runs
each through the real `compile_for_hardware_buggy()` (original) and the
real, patched `compile_for_hardware()`, then applies the same naive
`optimization_level=1` final step to both and measures mirror-circuit
fidelity plus native `ecr` count under the `fake_sherbrooke` noise model:

| Family | `compile_for_hardware_buggy` (P(all-zero)) | `compile_for_hardware`, patched (P(all-zero)) | ecr: buggy → patched |
| :--- | :---: | :---: | :---: |
| deep2q | 0.8747 ± 0.0076 | 0.9190 ± 0.0039 | 12 → 6 |
| multi_deep2q | 0.1484 ± 0.0122 | 0.1909 ± 0.0044 | 48 → 18 |
| wide | 0.0448 ± 0.0007 | 0.1418 ± 0.0069 | 24 → 18 |

(mean ± stdev of P(all-zero); N=5 seeds for deep2q/multi_deep2q, N=3 for
`wide`, 2048 shots each)

![Real compile_for_hardware(), old vs. patched: fidelity and native ecr gate count by family](/docs/090201.png)

The `ecr` counts land exactly where the earlier diagnostics predicted — the
patched path needs half the native 2-qubit gates of the buggy one on
`deep2q` (6 vs. 12) and `multi_deep2q` (18 vs. 48) — and fidelity improves
in every family, not only the two where PSF-Zero's own synthesis was
active. That last point is worth stating plainly: on `wide`, `compile()`
reported "0/0 blocks" processed (every block is under `block_gate_floor`,
so PSF-Zero's synthesizer never runs), yet the fix still recovers
~9.7 points of fidelity — because the bug lives in
`compile_for_hardware()`'s own device-submission transpile call, not in
anything PSF-Zero-specific. **This means the root cause under
investigation since section 8 is not actually a PSF-Zero synthesis defect
at all — it's a generic ISA-basis-translation gap in the hardware-submission
step, one that would affect any circuit `compile_for_hardware()` is asked
to prepare, regardless of which engine produced it.** This is now the
strongest evidence in this README: not a from-scratch reimplementation, but
the real, pasted `compile_for_hardware()` code itself, patched and
measured, using a verified stand-in only for the one binary that can't run
here.

The patch itself is a single, minimal, backward-compatible change —
[`compile_for_hardware.patch`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/compile_for_hardware.patch)
— meant to be applied directly to the real repository file (add a
`basis_gates` parameter, thread it through to `transpile(...)`, default
`routing_optimization_level` to 2; existing call sites keep their old
behavior until they pass `basis_gates` explicitly).

#### A second, independent root-cause thread: RXX/RYY/RZZ's native-gate cost is real, and separately confirmed against the actual production code

A separately-obtained, more recent copy of `psf_compile.py` — this one
already carrying a `verify: bool = True` parameter and a
`compile_for_hardware()` with `basis_gates` threaded through and
`routing_optimization_level` defaulting to 2 (i.e., already incorporating
the shape of the fix proposed above) — let us test the RXX/RYY/RZZ
mechanism directly against the real production code and real
`psf_zero_core` build, not a stand-in, using the actual
`test_real_hardware_fidelity.py` script (also separately recovered, so
section 8's "lost script" caveat above no longer fully applies to this
specific check).

Running that real script locally against `fake_sherbrooke` reproduced this
section's own numbers closely (e.g. `deep2q`/PSF_Zero_v6: 0.865 here vs.
0.8638 above), confirming it's the right script. Instrumenting it to record
the actual post-mirror, post-ISA-transpile native gate count (not just the
pre-mirror `two_qubit_gates` column already in the table above) showed the
`generate_preset_pass_manager(optimization_level=1, ...)` step this script
uses for the final ISA step gives PSF-Zero's block exactly 2x the native
`ecr` gates of the CX-based engines (6 vs. 3 for a single `deep2q` block) —
the identical 2x factor found independently above via a from-scratch
`TwoQubitWeylDecomposition`-based reproduction, now confirmed against the
real production synthesizer.

Adding an opt-in `entangling_basis: str = "canonical" | "cx"` parameter to
`GeodesicPSFHyper`/`synthesize()`/`compile()`/`compile_for_hardware()` —
`"cx"` resynthesizes the entangling core through Qiskit's own
`TwoQubitBasisDecomposer(CXGate())` (already imported for the existing
degenerate-point fallback, so no new trust surface) instead of emitting
`RXX`/`RYY`/`RZZ` directly — closes the gap directly, confirmed both in
isolation and end-to-end through the real, unmodified
`test_real_hardware_fidelity.py` (patched only at its `psf_compile(qc)`
call site to add a fifth `PSF_Zero_v6_cx` engine alongside the original,
for a same-run before/after comparison). *That patched five-engine script is
a separate file from the four-engine one linked at the end of section 8 —
it is `benchmarks/test_real_hardware_fidelity_cx.py`, and it is the one the
table below was produced with:*

| Family | Engine | `fake_sherbrooke` P(all-zero) |
| :--- | :--- | :---: |
| deep2q | Qiskit_L3 | 0.9047 |
| deep2q | TKET_native | 0.9134 |
| deep2q | PSF_Zero_v6 (canonical, unchanged) | 0.8652 |
| deep2q | **PSF_Zero_v6_cx (fix)** | **0.9088** |
| deep2q | Hybrid | 0.9089 |
| multi_deep2q | Qiskit_L3 | 0.0883 |
| multi_deep2q | TKET_native | 0.0852 |
| multi_deep2q | PSF_Zero_v6 (canonical, unchanged) | 0.0692 |
| multi_deep2q | **PSF_Zero_v6_cx (fix)** | **0.0837** |
| multi_deep2q | Hybrid | 0.0872 |
| wide | all 5 engines | 0.0029–0.0048 (no PSF-specific effect either way, as expected — PSF makes no changes on `wide`) |

(3 repeats, 3000 shots; `entangling_basis="cx"` correctness re-verified
unchanged at fidelity 1.000000000000 across CX/SWAP/iSWAP/Identity and 100
random SU(4) samples, both before and after this change, matching this
project's existing standard.)

`PSF_Zero_v6_cx` lands within noise of Qiskit_L3/TKET/Hybrid on both
families where PSF-Zero's synthesis is active, closing essentially all of
the deficit this section originally reported for `deep2q`/`multi_deep2q` —
consistent with, and now confirmed on top of, the independent
`compile_for_hardware()`-level fix above. `entangling_basis` defaults to
`"canonical"` (unchanged behavior) for the same reason `verify` defaults to
`True`: `RXX`/`RYY`/`RZZ` is the *right* choice on hardware whose native
2-qubit interaction is itself an XX/YY/ZZ-type gate (e.g. trapped-ion /
neutral-atom Mølmer–Sørensen gates) — this is a target-basis choice to make
deliberately per backend, not a universal default to flip.

**Where this leaves the root-cause question:** two independent mechanisms
were found and fixed — `compile_for_hardware()` silently leaving
`RXX`/`RYY`/`RZZ` undecomposed (no `basis_gates` threaded through), and,
separately, `RXX`/`RYY`/`RZZ` costing native hardware gates that `CX`
doesn't at low transpile optimization levels even once a basis *is*
targeted. Both are real, both are now fixed, and both move measured
fidelity in the right direction on the same circuit families this section
originally flagged. We're treating this as a substantially closed
investigation rather than a fully closed one: the exact optimization level
and basis-translation path used for sections 7/8's *original* real-hardware
numbers is still not directly confirmed (see the still-missing-script
caveat above), so we can't say with certainty that this exact mechanism,
rather than some combination of it and something else, produced those
specific numbers — but we can say the mechanism is real, reproduces at
matching scale, and a validated fix for it exists and is confirmed against
the real production code.
