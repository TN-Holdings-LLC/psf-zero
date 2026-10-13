# Evaluation harness for c49 / T2P (request 14 of 2026-10-13): fixed by the evaluator before the implementation exists; the 20 new Barabasi-Albert circuits' seeds fixed; rehearsed with a stand-in

**Status: harness and circuit list fixed; c49 not written here (home implements it).** Workplace, 2026-10-13; times
in CEST from `TZ=Europe/Zurich date` (the fixing time is in the kit README). Pre-registration this measures:
`t2-placement-preregistration-draft-2026-10-13.md` (V0-V7), unchanged.

## 1. What is fixed

**Circuits (31).** HW2's eleven part R / tfimP tests, built by the HW2 kit's own `build_test`, and 20 new ones:

- QAOA on networkx's Barabasi-Albert graph, m = 2 (as Benchpress's BA files: 24 edges at n = 14), three layers,
  HW2's angle ramp (gamma_k = 0.55 (k+1)/3, beta_k = 0.45 (1 - k/3)), measured on all qubits;
- n = 14, 16, 18, 20, five instances each; **seed = 4,840,000 + 1000 n + i, i = 0..4**, i.e. 4,854,000-4,854,004,
  4,856,000-4,856,004, 4,858,000-4,858,004, 4,860,000-4,860,004 (not used anywhere in the repository before; SCALE1
  uses 4,670,000 and 4,680,000);
- each circuit's signature (SHA-256 of its decomposed QASM, 16 hex) is in `circuits/t2p_circuits.json` (networkx
  3.6.1, qiskit 2.5.2). `run` stops if a new BA circuit built on home's machine differs (another networkx or qiskit
  version could change it).

**Arms.** Each PSF-Zero arm is called as `compile_for_hardware(qc, backend=<fake device>, seed_transpiler=0, **extra)`
with the release's defaults otherwise ("C48 as it is"; level-3 candidate "auto", so T2P is also exercised in item
80's comparison):

| arm | module | extra |
|---|---|---|
| C48 | `patches/psf_compile_c48_2026-10-12/psf_compile.py` | none |
| T2P | c49 (path given with `--c49`) | `placement_score="t2p"` |
| T2P_half | c49 | `placement_score="t2p_half"` |
| C49_esp | c49 | `placement_score="esp"` (check H0) |
| QK3 | Qiskit, `generate_preset_pass_manager(3, backend, seed_transpiler=0)` | |

**Contract with c49 (the only assumption about its code):** a keyword `placement_score` with "esp" reproducing c48
bit for bit, "t2p" and "t2p_half". If home names it otherwise, `ARMS` in `t2p_eval.py` is edited before c49's
predictions are fixed and the edit is recorded. HW2 called c48 with `entangling_basis="cx", layout_search=True,
level3_candidate=False`; if home prefers that call for this test, it is a one-line change, also to be made before.

**Devices:** FakeFez, FakeMarrakesh, FakeKingston, FakeTorino (qiskit-ibm-runtime 0.50's snapshots).

## 2. How each prediction is measured (computed by the harness, not by c48/c49 code)

| quantity | definition |
|---|---|
| ESP | product over gates and measurements of (1 - error) on the fake Target |
| exposure | sum over used qubits of (end - first gate) / T2, gates as soon as possible on the Target's durations, measurements left out (HW2's P4) |
| idle term | sum over used qubits of I_q (1/(6 T1) + 1/(3 T2)), I_q = window - own gate time, T2 capped at 2 T1, missing T1/T2 = 1 us; the T2/2 form for T2P_half is reported |
| combined cost | -ln ESP + idle term |
| low-T2 qubits | used qubits with T2 below the device's 10th percentile (over all qubits with a T2) |
| exactness | total variation distance between the compiled and logical outcome distributions, Aer statevector on the active qubits (at most 24); tolerance 1e-6 |
| determinism | signature of every instruction (name, qubits, clbits, parameters to 1e-10) equal on a second call |

| ID | computed as |
|---|---|
| H0 (check, not a prediction) | C49_esp's output signature equals C48's on every compile; if not, c49 changed c48's behaviour outside the switch |
| V0 | every T2P and T2P_half output: ISA, no element with error >= 0.5, the same on a second call, exact (where <= 24 active) |
| V1 | per device, share of circuits with cost(T2P) <= cost(C48) + 1e-12; confirmed if >= 0.95 on every device, refuted if < 0.85 on any |
| V2 | per device, geometric mean of exposure T2P / C48 over the 24 BA circuits (HW2's four + the twenty new); <= 0.90 on every device / > 0.97 on any |
| V3 | per device, geometric mean of ESP T2P / C48 over all 31 and the smallest single ratio; >= 0.97 everywhere and no ratio below 0.90 / < 0.95 on any device |
| V4 | per device, low-T2 qubits summed over the 31; on devices where C48 uses any: T2P <= half everywhere / T2P >= C48 on any; no such device: not testable |
| V5 | medians of the two-qubit-count and duration ratios T2P / C48, **pooled over devices and circuits** (the draft says "median"; per-device medians are reported); both in 0.95-1.05 / either outside 0.90-1.10 |
| V6 | median compile-time ratio T2P / C48, pooled; <= 1.2 / > 1.5 |
| V7 | reported: T2P_half / T2P (exposure, ESP); T2P and C48 against QK3 (ESP, exposure, mean cost difference) |

Two choices the draft left open are fixed here and disclosed: exactness is judged on the outcome distribution, not
the state (Qiskit level 3 and PSF-Zero may drop diagonal gates before measurement, which changes the state but not
any outcome); V5 and V6 use pooled medians.

## 3. Rehearsal (a stand-in for c49, `rehearse`)

The stand-in accepts `placement_score` and ignores it (it is c48), so the run tests the harness, not T2P. Expected:
H0 passes, T2P = C48 everywhere, V1 and V5 confirmed, V2 and V4 refuted where C48 uses low-T2 qubits. The full set
(31 circuits, 4 devices, 5 arms) is in `rehearsal_output/`; the summary and timing are in the kit README.

## 4. Running it at home (after c49 exists, before any hardware)

```
python -u <kit>/t2p_eval.py run --c49 patches/psf_compile_c49_<date>/psf_compile.py --out data/<date>/t2p_eval
```

It appends rows as it goes (resumable) and writes `t2p_eval_score.md/json`. No IBM connection, no git, no pip.
