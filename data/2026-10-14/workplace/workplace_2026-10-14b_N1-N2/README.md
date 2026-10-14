# Workplace -> home, 2026-10-14 (b): N1, N2, and the tensor/swap-network question

Packed 2026-10-14 (time in the reply, from `TZ=Europe/Zurich date`). Offline only; no IBM connection, no push. Common
rule followed (no search, no sweeps, no seeds varied, no ESP/simulation choices; first-match complete embedding used
as allowed). Base: bundle psf-zero_2026-10-14a (5ae4947), c49 sha256 5544d7c9...

| folder | content | result |
|---|---|---|
| `n1/` | item 86 patch proposal on c49 (`patch/make_c49_n1.py` builds it from c49; the built `psf_compile.py`; `test_n1.py`), report, end-to-end check on W2's 48 cells | CZ <= QK3-BW in 48/48, duration <= QK3-BW in 46/48 (0.80-1.02x); FakeFez n=100/10 steps 5.30 us (c48 471 us); the 2 misses are FakeMarrakesh n=100 (+0.2%, +1.6%, every 100-qubit path crosses 80 ns couplers). test_n1 9/9; c49's tests 52/54 (the 2 are the version string) |
| `n2/` | ring formula (folded line + interior odd-even swaps, mirrored layers), with a14/a15 on W3's circuits | shortest of N2/a15/a14/c48/QK3 on 61 of 64 cells (the 3 others are L = 1); at L = 16 0.69-0.87x the best other; 7L CZ (n=4), 15L CZ (n=6) |
| `q_tensor/` | owner's question: spectral (tensor-style) placement and swap network on BA QAOA and R4 | no gain over QK3 (stopped) |

Check: `python check_intake.py`. To rebuild the patch: `python n1/patch/make_c49_n1.py patches/psf_compile_c49_2026-10-14/psf_compile.py n1/patch/n1_code.py <out>`.

Running the checks: from the repository root after this bundle's files are in place. The scripts import W2's
`w2_tfim.py` (and W5's `w5_place.py`) from `data/2026-10-14/workplace/workplace_2026-10-14_W1-W2-W3-W5-W7/`; the
search path line was added to the kit copies for that (the runs were made from the workplace folders, otherwise the
same code). `n2_ring.py` and `n1_e2e.py` expect the built file in `patches/psf_compile_c49n1_2026-10-14/`.
