## Per model (solved task-runs; a missing result counts as not solved)

| model | solved /15 | ghz5 | w3 | bell3 | qft3 | fill27 | errors | HTTP 400 | missing | model s | PSF compile s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gptoss120b | 12 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 24 | 0 | 0 | 2731 | 0.68 |
| qwen72b | 9 | 3/3 | 0/3 | 3/3 | 0/3 | 3/3 | 4 | 0 | 0 | 1731 | 1.31 |
| qwen7b | 7 | 3/3 | 0/3 | 2/3 | 0/3 | 2/3 | 4 | 0 | 0 | 377 | 1.43 |

## Best model: gptoss120b

- G1: **ambiguous** (12/15 solved; w3 0/3, qft3 3/3)
- G2: **go** (solved task-runs 12; model <= StatePreparation baseline on 11, worse on 1)
- G3: **go** ({"circuits": 7, "median_psf_s": 0.016858596121892333, "median_ratio": 611.9010979320107, "min_ratio": 500.6019783523636, "max_ratio": 658.5489571405265, "psf_le_l3": 7, "psf_gt_l3": 0, "psf_2q": [17, 17, 17, 17, 17, 17, 17], "l3_2q": [20, 20, 20, 20, 20, 20, 20]})

## Decision: **CUT**

## G2 detail (best model)

| task | run | model 2q (PSF) | baseline 2q (PSF) | baseline 2q (L3) |
|---|---|---|---|---|
| bell3 | 1 | 3 | 3 | 3 |
| bell3 | 2 | 3 | 3 | 3 |
| bell3 | 3 | 3 | 3 | 3 |
| fill27 | 1 | 17 | 52 | 46 |
| fill27 | 2 | 17 | 52 | 46 |
| fill27 | 3 | 17 | 52 | 46 |
| ghz5 | 1 | 4 | 47 | 34 |
| ghz5 | 2 | 4 | 47 | 34 |
| ghz5 | 3 | 4 | 47 | 34 |
| qft3 | 1 | 0 | 7 | 3 |
| qft3 | 2 | 3 | 7 | 3 |
| qft3 | 3 | 12 | 7 | 3 |
