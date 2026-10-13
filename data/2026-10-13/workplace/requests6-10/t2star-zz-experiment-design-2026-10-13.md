# Design (request 9 of 2026-10-13): a small hardware experiment that measures unechoed dephasing (T2*) and residual ZZ directly, with predictions; to run only with the owner's go-ahead

**Status: design and predictions; nothing submitted.** Written at the workplace, 2026-10-13 (CEST).

## 1. Question

HW1 found the device more sensitive to time than an estimate built on the published T1 and T2. Two mechanisms act on
a qubit that waits without dynamical decoupling and are invisible in the published numbers:

- **H-T2\*:** the published T2 is a Hahn-echo time; an unechoed waiting qubit dephases on its Ramsey time T2*, which
  can be shorter.
- **H-ZZ:** a residual ZZ coupling to a neighbour shifts the waiting qubit's frequency by an amount that depends on the
  neighbour's state; over tens of microseconds this is a phase error. A Hahn echo removes it while the neighbour's
  state stays fixed.

## 2. Circuits

**Qubits, chosen by a rule fixed now** (on the device chosen for the session, from its Target at `prepare`):

- three target qubits: (a) the operational qubit with the lowest reported T2 above 10 us, (b) the qubit whose T2 is
  closest to the device median, (c) the operational qubit with the highest T2;
- for each, one neighbour as spectator: the coupled neighbour with the highest T1;
- the three (target, spectator) pairs must be at least two couplers apart from each other (otherwise take the next
  qubit in the same rule); they run in parallel in every circuit.

**Sequences** (target qubit; spectator prepared by X or nothing at the start and left alone; both measured):

| name | sequence | reads |
|---|---|---|
| Ramsey X | H, delay(tau), H, measure | <X> |
| Ramsey Y | H, delay(tau), Sdg, H, measure | <Y> |
| Echo X | H, delay(tau/2), X, delay(tau/2), H, measure | <X> |
| Echo Y | H, delay(tau/2), X, delay(tau/2), Sdg, H, measure | -<Y> (sign of Y flipped by the X) |

- **Delays:** tau = 0, 5, 10, 20, 40 us, as explicit `delay` instructions in units of the device's dt (rounded to the
  device's granularity), so that the executed timing does not depend on the hardware's default scheduling.
- **Spectator states:** |0> and |1>.
- **Readout reference:** two circuits with every used qubit in |0> and in |1>.
- **Count:** 4 sequences x 5 delays x 2 spectator states + 2 = 42 circuits, 1,000 shots each, dynamical decoupling and
  twirling off.

**QPU budget.** HW2's rough rule (0.5 ms per shot + 1 s per circuit) gives 63 s; HW1 used 83 QPU seconds against a
rough 208 s (ratio 0.40), so about 25 s is expected. `prepare` should refuse a job whose rough estimate exceeds 75 s
(about 30 s by HW1's ratio).

## 3. Analysis (fixed now)

- Per target, spectator state and tau: <X> and <Y> from the target's counts. Shots in which the spectator is not read
  in its prepared state are dropped (this removes most spectator decays during the wait); the share dropped is
  reported.
- Coherence C(tau) = sqrt(<X>^2 + <Y>^2) / C(0) (the normalisation removes readout and preparation errors to first
  order); phase phi(tau) = atan2(<Y>, <X>).
- ZZ rate: zeta = slope of [phi(tau | spectator 1) - phi(tau | spectator 0)] against tau over tau <= 20 us (least
  squares through the origin), divided by 2 pi; reported in kHz with its standard error from the shot noise.
- Shot noise: with 1,000 shots, <X> has a standard error of about 0.03, and a phase at C = 0.8 about 0.04 rad.

## 4. Predictions

| ID | prediction | confirmed | refuted |
|---|---|---|---|
| E1 | the echo matches the published T2: C_echo(20 us) against exp(-20 us / T2_reported), spectator in \|0> | within +-0.10 on at least 2 of 3 targets | off by more than 0.20 on all 3 |
| E2 | H-T2\*: C_Ramsey(20 us) / C_echo(20 us) <= 0.90, spectator in \|0> | on at least 2 of 3 targets | >= 0.98 on all 3 |
| E3 | H-ZZ: \|zeta\| >= 1 kHz in Ramsey | on at least 1 of 3 pairs, with \|zeta\| >= 3 standard errors | \|zeta\| < 0.5 kHz on all 3 pairs |
| E4 | the echo removes ZZ: echo phase difference between spectator \|1> and \|0> at 20 us | <= 0.10 rad on every pair where E3 holds | > 0.30 rad on any such pair |
| E5 | target (a), the low-T2 qubit, has the largest Ramsey-to-echo gap at 20 us | yes | no (reported either way) |
| E6 | reported: an effective idle time constant per target from the Ramsey coherence at 10, 20 and 40 us, against T2 and T2 / 2 (HW2-PLUS's P2 assumes T2 / 2) | - | - |

## 5. What it would mean

- **E2 confirmed:** idle windows without DD should be charged with a Ramsey-type time, not the published echo T2;
  this supports P2-type predictors and the "T2P-half" variant of the placement draft.
- **E3 confirmed, E4 confirmed:** residual ZZ is a real cost of waiting next to excited neighbours, and DD (HW3) should
  recover it; a placement or schedule that avoids long waits next to busy neighbours would follow.
- **E2 and E3 refuted:** the time sensitivity of HW1 is not explained by unechoed dephasing or static ZZ on these
  qubits; the remaining candidates are crosstalk during simultaneous gates, leakage and drift.

## 6. Limits

Three pairs on one device, one calibration. The spectator is static (a busy neighbour that is driven during the wait
can produce more than static ZZ). Delays are rounded to the device's timing granularity. Nothing here tests a compiled
circuit.
