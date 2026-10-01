# qml_home_eval score (SMOKE -- not a result)

teacher seed 23; theta* noiseless test accuracy 0.938; versions {"rel": "2026-09-28.1", "c2": "2026-10-01.1", "layout": "2026-10-01.1", "a5": "2026-10-01.a5", "core": "2026-09-29.1", "qiskit": "2.5.2", "aer": "0.17.2"}

P0: numpy vs Statevector 8.88e-16 (<= 1e-9); compiled noiseless vs logical 4.27e-15 (<= 1e-6); theta* test accuracy 0.938 (>= 0.9)

## Q1: theta* through each compiler (noisy test accuracy / mean margin / median 2q)

| device | REL | C2 | A5 | L3T |
|---|---|---|---|---|
| FakeAuckland | 1.000 / 0.4894 / 17 | 1.000 / 0.4894 / 17 | 1.000 / 0.4930 / 17 | 1.000 / 0.4822 / 17 |

- H1 (C2 margin >= REL - 0.005 on every device): **CONFIRMED**
- H2 (A5 margin >= L3T - 0.005 on >= 2 of 3 devices): **AMBIGUOUS** (1 of 1)
- H3 (noisy accuracy >= theta* - 2/32 for every arm on Torino and Kingston): **CONFIRMED**

## Q2: learning with the compiler in the loop (mean over seeds)

IDEAL (noiseless, same seeds): test accuracy 0.781, seed 941: 0.781 / loss 0.452

| device | arm | noisy test acc | noisy test margin | noisy train loss | noiseless acc of result | compiles | compile s (home) | median 2q |
|---|---|---|---|---|---|---|---|---|
| FakeAuckland | REL | 0.781 | 0.3770 | 0.4739 | 0.781 | 112 | 2.9 | 17 |
| FakeAuckland | C2 | 0.781 | 0.3770 | 0.4739 | 0.781 | 112 | 3.1 | 17 |
| FakeAuckland | A5 | 0.781 | 0.3796 | 0.4700 | 0.781 | 112 | 19.4 | 17 |
| FakeAuckland | L3T | 0.781 | 0.3680 | 0.4769 | 0.781 | 112 | 3.0 | 17 |

- H4 (every arm's noisy test accuracy >= IDEAL - 0.10): **CONFIRMED**
- H5 (train loss C2 <= REL + 0.01 and A5 <= L3T + 0.01 on both devices): **CONFIRMED**
