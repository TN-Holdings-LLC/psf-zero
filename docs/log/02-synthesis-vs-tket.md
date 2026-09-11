> **Archived record — part 2 of 6.**
> This file is one slice of the original PSF-Zero README, kept **verbatim**. It is a
> lab notebook: it contains claims that were later retracted, hypotheses that were
> later refuted, and corrections written directly underneath the text they correct.
> Nothing has been deleted. For what the project currently claims, read the
> [README](../../README.md); for a settled account of one topic, read
> [`docs/findings/`](../findings/).
>
> Sections 1-3 - correctness at scale (N=300), native synthesis vs. TKET across 10-160 qubits with its three independent runs, and the Trotter-block Hamiltonian families.
>
>
> **One edit was made to this file when it was archived:** file-relative image paths
> were rewritten from `./docs/...` to `../../docs/...` for the new directory depth.
> Root-relative paths (`/docs/...`) were left untouched — they resolve correctly at
> any depth. No wording, number, claim or correction was altered. The 13 substitutions
> are listed in [`README.md`](README.md#the-one-edit).
> Source: README.md lines 161-272, as of 2026-09-11.

---

### 1. Correctness at scale (N=300)

300 randomly sampled 2-qubit unitaries, each synthesized independently by all
four pipelines and checked for unitary equivalence against the original block
(all 300/300 passed for every pipeline):

| Metric | Qiskit (L3) | TKET | PSF-Zero | Hybrid (PSF→TKET) |
| :--- | :---: | :---: | :---: | :---: |
| Circuit depth — every one of 300 samples | 15 | 7 | 9 | 7 |
| Compile time, median | 6.0ms | 153.5ms | 1.5ms | 45.1ms |

![N=300 statistical benchmark: depth is identical for all 300 samples, and compile-time distributions by compiler](../../docs/090303.png)

The depth numbers aren't averages with some spread rounded off — they are
*exactly* 15 / 7 / 9 / 7 for every single one of the 300 randomly sampled
unitaries, with zero variance. That's expected, not surprising: PSF-Zero's
synthesis of a generic SU(4) unitary always resolves to the same canonical
(Weyl-chamber) form, so depth doesn't depend on which random unitary you feed
it — this is a direct consequence of doing an exact decomposition rather than
a search, not a claim about optimality. TKET's search-based peephole optimizer
reliably finds a shallower circuit (7 vs. 9) on this circuit family; we're not
aware of a way to close that gap without giving up the determinism and the
constant-time guarantee, and we think that's a fair trade-off to state plainly
rather than paper over. On compile time, PSF-Zero was the fastest of the four
in every sample, and also the most consistent (tightest distribution) —
Qiskit's L3 pass had a long tail, including two outlier samples that took
15–17x longer than its own median.

Code: [`benchmarks/test_psf_vs_tket.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_psf_vs_tket.py)

### 2. Native synthesis vs. TKET, by scale

Same circuit family (dense, same-pair 2-qubit interaction chains, built to
avoid the TKET `Unitary2qBox` incompatibility by pre-decomposing each block into
standard gates), run at 10, 20, 40, 80, and 160 qubits:

| Qubits | TKET time | PSF-Zero time | Speedup | TKET depth | PSF-Zero depth |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 1.064s | 0.007s | 152x | 7 | 9 |
| 20 | 2.135s | 0.009s | 237x | 7 | 9 |
| 40 | 4.193s | 0.016s | 262x | 7 | 9 |
| 80 | 8.244s | 0.032s | 258x | 7 | 9 |
| 160 | 16.840s | 0.062s | 272x | 7 | 9 |

![Native synthesis vs. TKET by scale: compile time and output depth](../../docs/090304.png)

The depth gap (TKET 7 vs. PSF-Zero 9) is flat across every scale we tested —
the same trade-off as the N=300 result above, on a different circuit family.
Note that the speedup factor here behaves differently from the Qiskit
comparison in section 4 below: against TKET's `FullPeepholeOptimise`, the
speedup holds roughly steady (150x–270x) rather than shrinking as qubit count
grows, because TKET's own compile time is scaling worse than linearly on this
circuit family over the range we tested. We're reporting both comparisons
because they don't tell the same story, and we'd rather show that than pick
whichever one looks better.

Independently re-run on 2026-09-07 (`pytest test_scale_explosion_war2.py -s`,
same machine): TKET 1.155s/2.136s/4.120s/8.339s/16.998s and PSF-Zero
0.007s/0.009s/0.017s/0.034s/0.064s at 10/20/40/80/160 qubits respectively —
every value within ~1–8% of the table above (ordinary run-to-run noise, not
a trend), and the 7-vs-9 depth split reproduced exactly at every scale, with
every block reported as processed by the Rust core (`0 fell back`) rather
than a no-op. One more data point for the reproducibility this project is
now leaning on.

Code: [`benchmarks/test_scale_explosion_war2.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_scale_explosion_war2.py)

#### Update (2026-09-10): third independent run, on the project's slower machine, with one flagged outlier

The identical, unmodified script was run a third time — this time on the
project's slower machine (the one used for section 5's 2026-09-09
confirmation), against a freshly-built, real `psf_zero_core` wheel:

| Qubits | TKET time | PSF-Zero time | PSF-Zero depth |
| :---: | :---: | :---: | :---: |
| 10 | 0.989s | 0.002s | 9 |
| 20 | 1.958s | 0.003s | 9 |
| 40 | 3.897s | 0.004s | 9 |
| 80 | 7.719s | **0.081s** | 9 |
| 160 | 15.579s | 0.022s | 9 |

TKET's times track the two earlier runs closely (same order of magnitude at
every scale). Depth reproduced at exactly 9 for every scale, again — the
one number in this table that doesn't depend on timing noise.

The 80-qubit PSF-Zero time does not fit the trend (0.002s / 0.003s / 0.004s
/ **0.081s** / 0.022s is not monotonic, and 0.081s is roughly 20x its
neighbors). This script measures one call per scale, not an average over
seeds, so there is nothing here to average the spike away with. This project
has already seen single-measurement noise of this shape before (section 5's
500-qubit outlier, 0.825s against a 0.173s re-run of the identical seed) and
we are treating this one the same way: **flagged as probable transient
system noise, not re-run yet, and not folded into the speedup claims above**
until either a repeat measurement confirms or contradicts it.

### 3. Hamiltonian simulation (Trotter blocks)

Using the standard XX/YY/ZZ/exchange/full two-qubit interaction blocks used in
Trotterized time evolution (VQE, condensed-matter simulation):

![Trotter interaction blocks: output circuit depth by compiler, original vs. Qiskit L3 vs. TKET vs. PSF-Zero](../../docs/090302.png)

Across all five interaction types, Qiskit Level 3 produced circuits of depth
15, PSF-Zero produced circuits of depth 9, and TKET's peephole optimizer
produced circuits of depth 7 — every interaction type gave the identical
15/7/9 split, the same three-way signature as the two benchmarks above, now
confirmed on a third, independently-motivated circuit family. PSF-Zero's
compile time was consistently the fastest of the three in every interaction
type tested (sub-3ms vs. Qiskit's ~5–40ms and TKET's ~50–57ms).

Code: [`benchmarks/test_official_hamiltonians_war.py`](https://github.com/TN-Holdings-LLC/psf-zero/blob/main/benchmarks/test_official_hamiltonians_war.py)
