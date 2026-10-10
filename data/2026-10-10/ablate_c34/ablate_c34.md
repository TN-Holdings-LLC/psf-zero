# ABLATE-C34 (exploratory, nothing predicted)

git head 0b7f790, uncommitted tracked changes: none; 106 FakeTorino tests on FakeTorino; 530 jobs, 4 at a time; Qiskit 2.5.2. 106 tests with every arm; 52 where the best ESP >= 0.01.

| arm | cold (s, summed) | warm (s, summed) | small tests: median cold / warm (ms) | q2 / C33D (gmean, +1) | q2 / QK3 | ESP / C33D (gmean, both > 0) | ESP 10% better / worse than C33D | same output as C33D | outputs on failed elements |
|---|---|---|---|---|---|---|---|---|---|
| C33D | 223.3 | 208.7 | 221.1 / 81.8 | 1.0000 | 1.106 | 1.000 | 0 / 0 | 106 of 106 | 0 |
| C34D | 206.8 | 207.8 | 111.4 / 66.3 | 1.0000 | 1.106 | 1.000 | 0 / 0 | 106 of 106 | 0 |
| C34Q | 183.7 | 182.3 | 100.7 / 59.9 | 1.0000 | 1.106 | 1.005 | 0 / 0 | 15 of 106 | 0 |
| C34N | 126.1 | 125.5 | 85.7 / 52.7 | 1.1000 | 1.217 | 0.918 | 0 / 14 | 15 of 106 | 0 |
| C34X | 174.9 | 174.7 | 96.4 / 64.0 | 1.1123 | 1.231 | 0.956 | 0 / 7 | 37 of 106 | 0 |

For scale, ESP-FT's Qiskit level 2 on the same tests: 59.6 s (one call, cold, in its own process).
