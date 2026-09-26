# Addendum 184 -- Timed compounding pilot on FakeNighthawk: the cliff recurs on every lap (Qiskit 0/10 laps within 1 s, 129 s of compile time over 10 laps) while PSF-Zero meets the deadline 10/10 (0.8 s total); but PSF-Zero's per-pair drift grows ~7e-11 per lap and exceeds the pre-registered 1e-10 bound -- D4 refuted for PSF-Zero (2026-09-26)

**Pre-registered in**:
`spare-qubit-cliff-addendum-183-preregistration-2026-09-26.md`. Run on
WSL2 (home), Python 3.12.13, Qiskit 2.5.2, PennyLane 0.45.1. The saved log
begins with the pre-run check: 7671 bytes, SHA-256 `8ffabd4e...` --
matches the locked script. Log and CSV received as files; every figure
below recomputed from them.

## 0. In one line

D1, D2 and D3 hold; **D4 is refuted for PSF-Zero**. On the cliff (spare
= 0) Qiskit's default took 12.8-13.1 s on every one of 10 laps -- feeding
its own compiled output back in does not make the cliff go away -- and met
the 1 s deadline 0 times (129.3 s total); PSF-Zero met it 10 times (0.8 s
total, ~0.06 s per lap after the first). Every lap of every run completed
with no SWAP. But PSF-Zero's maximum per-pair distance from the original
grew from 1.8e-11 to 6.5e-10 over 10 laps, above the 1e-10 bound, while
Qiskit's stayed at ~3-5e-13.

## 1. Results (10 laps each; runs executed one after another)

| spare | arm | within 1 s | compile median | compile total | max pair distance lap 1 -> 10 | routed 2q | layout changes |
|---:|---|---:|---:|---:|---|---:|---:|
| 0 | Q3 Qiskit default | 0/10 | 12.88 s | 129.3 s | 3.37e-13 -> 3.40e-13 | 180 | 0 |
| 0 | P PSF-Zero | 10/10 | 0.061 s | 0.8 s | 1.76e-11 -> 6.47e-10 | 180 | 0 |
| 8 | Q3 Qiskit default | 10/10 | 0.151 s | 1.5 s | 3.37e-13 -> 4.34e-13 | 168 | 1 (lap 2) |
| 8 | P PSF-Zero | 10/10 | 0.028 s | 0.3 s | 1.76e-11 -> 6.47e-10 | 168 | 0 |

PSF-Zero's first lap at spare = 0 took 0.247 s; laps 2-10 took 0.059-0.076
s. Its per-pair distances are identical at spare 0 and 8, lap by lap: the
worst pair is among those present in both (same seed, same pairs).

## 2. Scoring (Addendum 183)

- **D1 -- CONFIRMED.** Q3, spare 0: > 1 s on 10/10 laps; median 12.88 s
  (bar: >= 5 s). The cliff recurs on every lap.
- **D2 -- CONFIRMED.** P, spare 0: <= 1 s on 10/10 laps.
- **D3 -- CONFIRMED.** Spare 8: both arms <= 1 s on 10/10 laps.
- **D4 -- REFUTED for P; CONFIRMED for Q3.** All 40 laps completed with no
  SWAP or failure; Q3's distance after lap 10 is 3.4e-13 (spare 0) and
  4.3e-13 (spare 8), below 1e-10; P's is 6.47e-10 at both spares, above it.

## 3. What this means

- **For a training loop that recompiles**, the cliff is not a one-time
  cost: Qiskit paid ~13 s on every lap. Over 10 laps the difference was
  129.3 s versus 0.8 s (about 160x); at the same rates, 1,000 laps would be
  about 3.6 hours versus about 1 minute.
- **PSF-Zero's numerical drift is real and larger here than in Addendum
  182.** There, on 4 qubits through the per-block synthesizer and a level-1
  transpile, PSF-Zero drifted ~1.4e-13 per lap; here, through
  `compile_for_hardware`, it drifts ~7e-11 per lap -- about 500 times
  faster -- while Qiskit's default barely drifts at all. The magnitude is
  physically negligible (as a fidelity loss, of order the square, ~1e-18),
  but it is a genuine precision gap in PSF-Zero's hardware-compilation path,
  and a concrete target. Its source inside `compile_for_hardware` was not
  traced in this run.
- Together with Addenda 180 and 182: PSF-Zero is dramatically faster on the
  cliff and stays correct in the sense that matters, but it is the less
  numerically precise of the two, and the gap is largest in the path a user
  would actually call.

## 4. What this does not establish

- Behaviour beyond 10 laps.
- The cause of PSF-Zero's larger drift in `compile_for_hardware`.
- Real hardware; typical (non-saturated) training circuits.

## 5. Files

| File | What it is |
|---|---|
| `deadline_compound_chain.py` | the script (hash-locked in Addendum 183) |
| `deadline_compound_chain_2026-09-26.csv` | raw results, 40 rows |
| `deadline_chain_result.txt` | raw log, including the pre-run hash check |

## 6. Replication (second run, same day)

A second run of the same hash-locked script (log begins: 7671 bytes,
SHA-256 `8ffabd4e...`) reproduced every verdict. Compile medians changed by
at most 1% (Q3 at spare 0: 12.879 s -> 12.998 s; total 129.3 s -> 130.4 s;
P at spare 0: 0.061 s -> 0.061 s, total 0.8 s -> 0.9 s), and every
per-lap maximum pair distance, for all four runs, was identical to the
first run's to the last digit -- the drift is deterministic, not noise.
Files: `deadline_compound_chain_2026-09-26_rep2.csv`,
`deadline_chain_result_rep2.txt`.

