## Per model (solved task-runs; a missing result counts as not solved)

| model | solved /15 | ghz5 | w3 | bell3 | qft3 | fill27 | errors | HTTP 400 | missing | model s | PSF compile s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gptoss120b | 14 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 | 5937 | 0.44 |
| qwen7b | 7 | 3/3 | 0/3 | 3/3 | 0/3 | 1/3 | 0 | 0 | 0 | 278 | 0.42 |

## Best model: gptoss120b

- G1: **go** (14/15 solved; w3 2/3, qft3 3/3)
- G2: **go** (solved task-runs 14; model <= StatePreparation baseline on 14, worse on 0)
- G3: **go** ({"circuits": 2, "median_psf_s": 0.010606724499666598, "median_ratio": 697.9320624558952, "min_ratio": 664.5265137531188, "max_ratio": 731.3376111586715, "psf_le_l3": 2, "psf_gt_l3": 0, "psf_2q": [17, 17], "l3_2q": [20, 20]})

## Decision: **INVEST**

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
| qft3 | 2 | 0 | 7 | 3 |
| qft3 | 3 | 3 | 7 | 3 |
| w3 | 1 | 6 | 7 | 7 |
| w3 | 3 | 6 | 7 | 7 |
