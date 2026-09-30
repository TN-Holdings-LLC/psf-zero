## Solved task-runs (a missing result counts as not solved)

| arm | tuned (not run) | held-out /15 | total /15 | w4 | dicke42 | ghz3i | singlet3 | fill27g9 | missing |
|---|---|---|---|---|---|---|---|---|---|
| v9 | 0 | 15 | 15 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 |
| v10 | 0 | 14 | 14 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 0 |

- H1 (held-out, v10): **go** (14/15)
- H2 (v10 - v9, 15 held-out task-runs each): **no clear difference** (-1)
- H3 (G2 on held-out, v10): **go** (solved 14; <= baseline 14, worse 0)
- H4 (G3 on fill27g9): **stop** ({"circuits": 3, "median_psf_s": 1.2097026229894254, "median_ratio": 6.647002158375925, "min_ratio": 6.42751198552426, "max_ratio": 6.788390286910269, "psf_le_l3": 2, "psf_gt_l3": 1, "psf_2q": [18, 18, 24], "l3_2q": [18, 18, 18]})

## Default harness: **KEEP v9**
## Line: **CONTINUE**

