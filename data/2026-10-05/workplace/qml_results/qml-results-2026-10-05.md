# Workplace results: the QML test (pre-registration `qml-preregistration-2026-10-01.md`, Amendment 1 `qml-amendment1-2026-10-04.md`). Five of six confirmed, Q3 ambiguous: a better compiler keeps more of the classifier's signal and trains it to a lower loss, but the deployed accuracy does not move (2026-10-05)

**Status: results of the workplace pre-registration of 2026-10-01, scored by the locked `qml_eval.py score`.**

- **Setting:** workplace sandbox (2 CPUs), Qiskit 2.5.2, qiskit-aer 0.17.2, core 2026-09-29.1; fake devices and Aer
  noise only.
- **Re-runs:** only the two A5 training runs stopped by the infrastructure interruption of 2026-10-01, re-run
  identically under Amendment 1. Every logged value of the interrupted runs was reproduced exactly (A5-911 at
  iterations 0 and 20: test accuracy 0.8167, loss 0.75055; A5-912 at iteration 0).
- **Relation to home:** home ran an independent test of the same question the same day (Addenda 290-291, 296-297).
  This test reaches the same picture and adds little beyond it. It is recorded for completeness.

## 1. Verdicts

| ID | Verdict | Numbers |
|---|---|---|
| C0 | OK | versions 2026-09-28.1, 2026-10-01.c2, 2026-10-01.a5 in all 10 files |
| Q1 | **CONFIRMED** | 32 of 32 compiled model circuits exact |
| Q2 | **CONFIRMED** | median retention A5 vs REL: Auckland 0.814 vs 0.719, Torino 0.900 vs 0.772 |
| Q3 | **AMBIGUOUS** | pooled deployed accuracy A5 - REL = -0.010 (confirm >= +0.02, refute < -0.02) |
| Q4 | **CONFIRMED** | retention A5 vs L3T: 0.814 vs 0.820, 0.900 vs 0.900 |
| Q5 | **CONFIRMED** | mean final test loss under noisy training: A5 0.674, REL 0.778 (L3T 0.710) |
| Q6 | **CONFIRMED** | mean final test accuracy: A5 0.833, REL 0.758 (L3T 0.792) |

## 2. Numbers

**Deployment** (teachers 911-914; ideal test accuracy 0.875 on both devices):

| device | noisy acc REL / C2 / A5 / L3T | median retention REL / C2 / A5 / L3T |
|---|---|---|
| FakeAuckland | 0.871 / 0.871 / 0.871 / 0.879 | 0.719 / 0.719 / 0.814 / 0.820 |
| FakeTorino | 0.896 / 0.883 / 0.875 / 0.871 | 0.772 / 0.811 / 0.900 / 0.900 |

**Training under noise** (SPSA, 80 iterations, FakeAuckland; final test loss / accuracy):

| arm | teacher 911 | teacher 912 |
|---|---|---|
| REL | 0.773 / 0.733 | 0.784 / 0.783 |
| C2 | 0.773 / 0.733 | 0.784 / 0.783 |
| A5 | 0.631 / 0.900 | 0.717 / 0.767 |
| L3T | 0.725 / 0.767 | 0.695 / 0.817 |

## 3. Reading

- **Signal, not accuracy.** A5 and L3T keep 9-13 points more of the model's output than the released compiler, but
  on this well-separated test set noise rarely flips a sign, so deployed accuracy is level (Q3). Home's Addendum 291
  found the same, and Addendum 297 showed that a fragile test set is needed to see accuracy move.
- **Training.** With the noisy device inside the loss, A5 reached the lowest test loss on both teachers. With two
  teachers and one SPSA run each, the accuracy difference (Q6) is within run-to-run noise. The loss difference (Q5)
  is the more reliable of the two.
- **C2 equals REL on FakeAuckland** (identical circuits), and improves retention on FakeTorino (0.772 → 0.811), as in
  Addendum 291.
- **By construction.** A5's estimate shares the simulator's physics, so Q2, Q4 and Q5 favour it (Addenda 286-297).

## 4. Files (workplace sandbox `qml/run/`)

`d_auck.json`, `d_tor.json`, `t_{REL,C2,A5,L3T}_{911,912}.json`, their logs, `score.txt`, and `interrupted/` (the logs of
the two interrupted runs).
