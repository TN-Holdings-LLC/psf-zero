# Design v2 (request 13 of 2026-10-13): T2*/ZZ measurement with enough power for E3, a kit (prepare / submit / fetch / run / score / rehearse), rehearsed on FakeFez with Aer; nothing submitted

**Status: design v2 and kit; rehearsed offline only. No real-device job was submitted.** Written at the workplace,
fixed 2026-10-13 09:2x CEST (the exact time is in the kit README, from `TZ=Europe/Zurich date`). v1 is
`t2star-zz-experiment-design-2026-10-13.md` (kept unchanged); this file lists what changed and why. Runs only with
the owner's go-ahead.

## 1. Why v1 had to change: the ZZ estimate was underpowered

Monte Carlo of the analysis with binomial shot noise, readout error 1.5%, spectator T1 150 us, post-selection on the
spectator (`t2zz_power.py`, output `t2zz_power_output.md`). Standard error of zeta in kHz, true zeta = 0:

| design | rough QPU (s) | T2* 15 us | 30 us | 60 us | 120 us |
|---|---|---|---|---|---|
| v1 grid (tau 0/5/10/20/40), 1,000 shots, fit <= 20 us | 63 | 0.91 | 0.57 | 0.42 | 0.37 |
| A: v1 grid, 2,000 shots, fit <= 40 us | 84 | 0.61 | 0.32 | 0.20 | 0.15 |
| B: tau 30 added, 1,000 shots, fit <= 40 us | 75 | 0.80 | 0.38 | 0.23 | 0.17 |
| **v2 = C: tau 30 added, 2,000 shots, fit <= 40 us** | **100** | **0.56** | **0.27** | **0.16** | **0.13** |

(All rows use v2's weighted estimator; v1's unweighted fit is no better.) With v1, 3 standard errors were 1.1-2.7
kHz, so E3's "|zeta| >= 1 kHz with >= 3 SE" could hardly be confirmed even if zeta were 1.5 kHz (probability 0.39 at
T2* = 30 us, 0.75 at 60 us); its refutation band (|zeta| < 0.5 kHz) caught a true zero only 63-83% of the time per
pair. With v2, a true 1.5 kHz is confirmed with probability 0.97 at T2* = 30 us and 1.00 at 60 us, and a true zero
falls in the refutation band with probability 0.94-1.00 per pair (0.62 at T2* = 15 us).

**What gives the power:** mainly fitting the weighted slope over all delays up to 40 us (the long delays carry the
phase; weights put low-coherence points where they belong), then doubling the shots. Adding tau = 30 us is the
smaller gain. Literature (request 16) puts a Heron pair's residual ZZ plausibly at 0.1-10 kHz, most likely around or
below 1 kHz, so a standard error near 0.2 kHz is what the question needs.

## 2. Changes from v1 (everything else as v1)

| item | v1 | v2 | reason |
|---|---|---|---|
| delays | 0, 5, 10, 20, 40 us | 0, 5, 10, 20, **30**, 40 us | power (home's option), fills the fit window |
| shots | 1,000 | **2,000** | power (home's option) |
| ZZ fit | least squares through the origin, tau <= 20 us | **weighted** least squares through the origin (weights 1 / var from shot noise, delta method), **0 < tau <= 40 us** | power; section 1 |
| phase offset at tau = 0 | not considered | d(0) reported with its SE; **if \|d(0)\| >= 3 SE, E3 uses the free-intercept fit** (also always reported) | a spectator-dependent phase from the preparation layer would bias a fit through the origin |
| rule (a) | lowest T2 above 10 us | lowest T2 **at or above 25 us** | on FakeFez the v1 rule picked a qubit with T2 = 10.0 us: echo coherence 0.15 at 20 us, ZZ SE 1.1 kHz, E2 ratio unreadable |
| pair separation | "at least two couplers apart" | shortest-path distance **>= 3** between any qubits of different pairs | no shared neighbour between pairs |
| layers | not specified | barriers: preparation \| wait \| final rotation \| measure | the spectator's X must precede the wait under any device timing (under ALAP without a barrier it would move to the end) |
| executed schedule | not requested | SamplerV2 `experimental.execution.scheduler_timing = True`; `score` checks that the spectator's X starts before the target's wait and whether measurements are aligned | answers by observation; no extra QPU (request 16) |
| QPU | 42 circuits, 42,000 shots, rough 63 s, cap 75 s | **50 circuits, 100,000 shots, rough 100 s (about 40 s by HW1's ratio 0.40), cap 110 s** | |

Predictions E1-E6 keep v1's wording and thresholds; E3 uses the estimate named by the d(0) rule.

## 3. Kit (`t2zz_kit/`)

| file | what |
|---|---|
| `t2zz_real.py` | `rehearse`, `prepare` (device rule as HW2/HW3: ibm_fez or ibm_marrakesh, operational with fewer pending jobs; qubit rule; ISA circuits with explicit delays in dt; manifest before submission; refuses if prepared), `submit` (one job; DD and twirling off; schedule requested), `fetch`, `run` (= submit + fetch), `score` |
| `test_t2zz.py` | 4 offline tests: pair rule and separation; 50 ISA circuits and programmed delays; noiseless expectations and bit order (Ramsey X = +1, Y = 0, spectator always read as prepared); exact recovery of zeta by the fit |
| `t2zz_power.py`, `t2zz_power_output.md` | section 1 |
| `rehearsal_output/` | the FakeFez rehearsal (section 4) |

The script needs only qiskit, qiskit-aer, qiskit-ibm-runtime and numpy (not the psf-zero repository). It reads no
key and writes none; it runs no git and no pip.

## 4. Rehearsal on FakeFez + Aer (no IBM connection)

`rehearse` prepares on FakeFez, then simulates the prepared circuits with Aer and FakeFez's noise model (T1/T2
relaxation on gates and delays, gate and readout errors), adding a **known** truth per pair, injected after each
delay on the target: a static ZZ (RZZ, zeta = 4.0, 1.5, 0 kHz for pairs a, b, c) and a quasi-static detuning that
changes between 10 batches (spread 6, 3, 0 kHz; an echo removes it, a Ramsey does not).

FakeFez's qubit rule gave (a) 140/141 (T2 27 us), (b) 89/88 (T2 86 us), (c) 1/2 (T2 303 us). Result with the
default seed (`rehearsal_output/rehearse_log.txt`):

| pair | zeta injected (kHz) | recovered (kHz) | C_R/C_E at 20 us | expected from the injected detunings |
|---|---|---|---|---|
| a | 4.00 | +3.75 +- 0.38 (-0.7 SE) | 0.770 | 0.775 |
| b | 1.50 | +1.70 +- 0.16 (+1.2 SE) | 0.975 | 0.937 |
| c | 0.00 | -0.11 +- 0.10 (-1.1 SE) | 1.010 | 1.000 |

E1 (echo against exp(-20 us / T2)) agrees within 0.03 on all three pairs, as it should with FakeFez's own T2 in the
noise model; E2 comes out NOT DECIDED (only pair a is below 0.90), which matches the injected truth (expected ratios
0.78, 0.94, 1.00); E3, E4 and E5 come out as the injected truth implies. Two more seeds (development runs, before the last code change) gave deviations between -2.0 and +2.1 SE
over 9 pair checks; a separate check with 200,000 shots found no spectator-dependent phase at tau = 0 in the
simulation (|d(0)| <= 0.005 rad), and the Monte Carlo (2,000 repetitions) found the estimator unbiased with
calibrated standard errors (zeta 4 kHz at T2* 19 us: mean 3.98, SD 0.43, mean SE 0.42 kHz). The schedule check
reports "returned for 0 of 50" in the rehearsal, as it must (Aer returns no schedule).

## 5. Limits added in v2

- The weighted fit trusts the delta-method variances; below a coherence of about 0.05 they are poor, but such points
  then carry little weight.
- A spectator-dependent phase that is not linear in tau (for example a spectator that is driven, or two-level
  defects) is not modelled; the residuals of the fit are reported in `zz_points`.
- The rule change for (a) and the d(0) rule are made before any run and are disclosed here; if home prefers v1's
  rule (a), set `MIN_T2_A = 10e-6` and re-fix.
