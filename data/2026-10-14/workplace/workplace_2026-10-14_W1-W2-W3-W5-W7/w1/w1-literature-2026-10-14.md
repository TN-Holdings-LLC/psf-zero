# W1 literature check, points 1-3 of the revised request (2026-10-14)

**Scope as revised by home on 2026-10-14:** (1) pre-registered compiler comparisons on hardware, (2) higher hardware F from reordering the same gates, (3) Qiskit's "cliff". The earlier points on figures of merit and on failed elements were cancelled and are not reported here. Prior work only; no implementation ideas. Two key citations (arXiv:2607.02031, arXiv:2001.02826) were re-opened and checked by the workplace session after the review.

Scope: a desk review for PSF-Zero, a Qiskit-based compiler that is compared against Qiskit's preset transpiler on IBM Heron devices. Every claim below comes from a page that was opened during this review: an arXiv abstract or HTML/PDF page, a journal page, IBM documentation, or a GitHub issue. Where a detail could not be found on the opened page, it is marked "not found" or "not stated". Each source has at most one quote, and every quote is under 15 words.

Method note: direct `curl` access to arxiv.org and export.arxiv.org was blocked by the session's egress policy (HTTP 403). All arXiv pages were read through the web-fetch tool instead, so the arXiv API was not used. The text extraction sometimes garbles numbers that the source renders as LaTeX. Where that happened, it is noted next to the number.

---

## 1. Pre-registered compiler comparisons on real hardware, scored on all cases

**Status: not found for strict pre-registration. Hardware compiler comparisons exist, but none fixed predictions or an analysis plan in advance.**

Evidence:

- **Benchpress.** Nation, Saki, Brandhofer, Bello, Garion, Treinish, Javadi-Abhari (IBM Quantum). arXiv:2409.08844, published in *Nature Computational Science* 5, 427–435 (2025), DOI 10.1038/s43588-025-00792-y.
  - Scale: 1,066 tests (1,032 transpilation tests) across 7 SDKs (Qiskit 1.2.0, Tket, BQSKit, Cirq, Braket, Staq, Qiskit Transpiler Service). Circuits go up to 930 qubits.
  - Device target: "FakeTorino", a snapshot of a 133-qubit Heron. Metrics are 2Q gate count, 2Q depth and runtime, with a 3,600 s timeout.
  - **No hardware runs.** The paper states: "we do not execute the resulting circuits on hardware."
  - Failures and timeouts are reported for every SDK, so all cases are scored. Pre-registration is not stated. Qiskit was the only SDK that passed every transpilation test.
  - Sources: https://arxiv.org/html/2409.08844v2 and https://www.nature.com/articles/s43588-025-00792-y
- **Mills, Sivarajah, Scholten, Duncan, "Application-Motivated, Holistic Benchmarking of a Full Quantum Computing Stack."** *Quantum* 5, 415 (2021), arXiv:2006.01273.
  - **Real hardware**: ibmqx2, ibmq_ourense, ibmq_16_melbourne and ibmq_singapore, between 2020-01-29 and 2020-02-10.
  - Five compilation strategies were compared: noise-aware and noise-unaware pytket, noise-aware and noise-unaware Qiskit, and pytket routing only.
  - Three circuit classes (shallow IQP, square/QV-like, deep Pauli-gadget). Each class and width has 200 circuits run at 8,192 shots. Metrics are heavy-output probability, cross-entropy difference and l1 distance.
  - Finding: no single winner. The best strategy depends on device, circuit class and width.
  - Pre-registration is not mentioned. The paper says it shows only a subsample of results in its figures, but the full dataset is on Zenodo (10.5281/zenodo.3832121).
  - Source: https://quantum-journal.org/papers/q-2021-03-22-415/pdf/
  - This is the **closest match** found: a hardware compiler comparison that releases complete data, though it is not pre-registered.
- **Arline Benchmarks.** Kharkov, Ivanova, Mikhantiev, Kotelnikov, arXiv:2202.14025 (2022).
  - Compared Qiskit, Tket, Cirq and PyZX on gate counts, depth, a fidelity-weighted cost and runtime.
  - **Compile-only**: gate fidelities were taken from vendor publications, and no circuits were run on hardware.
  - Source: https://arxiv.org/pdf/2202.14025
- **Salm, Barzen, Leymann, Weder, Wild, "Automating the Comparison of Quantum Compilers for Quantum Circuits."** SummerSOC 2021 (Springer), pp. 64–80.
  - Compared t|ket> against the Qiskit transpiler targeting ibmq_16_melbourne, and t|ket> against Quilc on a mocked Rigetti device.
  - Only 2 use cases, with width, depth and an executability check. No hardware results are reported.
  - Source: https://www.iaas.uni-stuttgart.de/publications/Salm2021_CompilerComparison.pdf
- **SupermarQ.** Tomesh et al., arXiv:2202.11045.
  - An application-level benchmark suite with results on IBM, IonQ and AQT@LBNL hardware.
  - From the abstract page, it benchmarks devices, not compilers. Pre-registration is not mentioned.
  - Source: https://arxiv.org/abs/2202.11045v2
- **QED-C application-oriented benchmarks.** Lubinski et al., arXiv:2110.03137, DOI 10.1109/TQE.2023.3253761.
  - A volumetric, application-oriented benchmark suite for hardware.
  - The abstract page does not say that compilers were compared or that anything was pre-registered.
  - Source: https://arxiv.org/abs/2110.03137v3
- **Pati and Simmhan, "Per-Phase Fidelity Attribution for Quantum Compilers using HBR Decomposition."** arXiv:2605.07876.
  - Compares Qiskit, PennyLane and TKET with an analytic fidelity model.
  - Real hardware use was limited to **2 circuits on IBM Fez** (BV n=12, Grover n=4). Everything else ran on a noisy simulator (FakeFez).
  - Parameters such as the seed and shot count were fixed in advance, but the paper does not describe a registered analysis plan.
  - Source: https://arxiv.org/html/2605.07876
- **Methodology papers.** Neither paper below mentions pre-registration.
  - Lorenz, Monz, Eisert et al., "Systematic benchmarking of quantum computers: status and recommendations" (arXiv:2503.04905). It stresses transparency and published evaluation code, but does not mention pre-registration or cherry-picking. Source: https://arxiv.org/pdf/2503.04905
  - Proctor, Young, Baczewski, Blume-Kohout, "Benchmarking quantum computers" (arXiv:2407.08828). It discusses gaming in Goodhart terms ("such benchmarks can be gamed.") but not pre-registration. Source: https://arxiv.org/pdf/2407.08828
- An explicit web search for "preregistered" or "pre-registered" quantum compilation experiments returned nothing relevant.

Relation to PSF-Zero: no prior compiler-vs-compiler study on hardware was found that fixed its predictions and scoring rules before running and then scored every case. Within what this review covered, PSF-Zero's pre-registered, all-cases design appears to be new. Mills et al. (2021) is the closest precedent, because it ran on hardware and released its full data.

---

## 2. Improving real-device fidelity only by reordering the same gates

**Status: partly known. Reordering is widely used, but almost always together with gate-count changes. Only a few hardware studies hold the gates fixed and change only timing or order, and none of them used Trotter or Pauli-block reordering on Heron.**

Studies that change only timing or order on hardware:

- **Murali, McKay, Martonosi, Javadi-Abhari, "Software Mitigation of Crosstalk on NISQ Computers."** arXiv:2001.02826, ACM DOI 10.1145/3373376.3378477 (the task lists it as ASPLOS 2020; the opened page does not name the venue).
  - XtalkSched takes the same hardware-compliant circuit and assigns gate start times, enforced with barriers. It serializes high-crosstalk pairs while balancing decoherence.
  - Hardware: IBMQ Poughkeepsie, Johannesburg and Boeblingen (20 qubits each).
  - Results versus IBM's parallel scheduler: SWAP-circuit error rate improved up to 5.6x (geometric mean about 2x), and QAOA cross-entropy loss improved up to 3.6x.
  - The paper notes that naive full serialization hurts because of decoherence.
  - This is the clearest **"same gates, different time order → higher hardware fidelity"** result found. Its motivation is crosstalk, not Trotter reordering.
  - Sources: https://arxiv.org/abs/2001.02826v1 and https://ar5iv.labs.arxiv.org/html/2001.02826
- **Liu, Maruyama, Oshima, Sato (Fujitsu / RIKEN), "Idling error suppression through gate scheduling."** arXiv:2607.02031 (July 2026).
  - Moves only the H gates that reverse CNOT direction within an idle window (front, middle or back), with "no additional gate operations."
  - Hardware: the 64-qubit RIKEN RQC-Fujitsu superconducting device, using 2 physical qubits. Circuits are single-qubit idling (states |1>, |->) and two-qubit Bell idling, at 10,000 shots per single-qubit point. The metric is ideal-state probability.
  - Midpoint placement was best or near best for short idle times, but gains were not uniform for long idles. Hardware numbers appear only in plots, and the "up to 12%" figure comes from simulation.
  - This is the only small-scale demonstration found. It does not cover Pauli blocks.
  - Source: https://arxiv.org/pdf/2607.02031
- **Dangwal et al., "COMPASS: Compiler Pass Selection for Improving Fidelity of NISQ Applications."** arXiv:2306.15020.
  - Includes ASAP and ALAP scheduling as one of 72 pass combinations, measured on IBM Auckland and Hanoi across 19 benchmarks.
  - The scheduler's effect is not isolated or reported separately in the text that was read.
  - Source: https://arxiv.org/pdf/2306.15020v2

Studies that reorder commuting terms but also change gate counts:

- **Tomesh, Gui, Gokhale, Shi, Chong, Martonosi, Suchara, "Optimized Quantum Program Execution Ordering to Mitigate Errors in Simulations of Quantum Systems."** ICRC 2021.
  - Reorders Hamiltonian Pauli terms with a clique cover, TSP ordering within cliques and commutator-based ordering between cliques.
  - Hardware: 6 molecules, each with 4 data qubits and 1 ancilla. Ethene ran on a 7-qubit device and five others on an 11-qubit IonQ device. The metric is Hellinger infidelity.
  - Their ordering (mctsp) had lower infidelity in 5 of 6 cases. For C2H2 it was 57.7% versus 57.2% for magnitude ordering.
  - **The CNOT count also changes** (for example, 55/49/41), so this is not a fixed-gate comparison.
  - Source: https://mrmgroup.cs.princeton.edu/papers/tomesh-term-grouping.pdf
- **Alam, Ash-Saki, Ghosh, "Circuit Compilation Methodologies for QAOA."** Appears to be MICRO 2020 based on the PDF header.
  - Reorders commuting CPHASE gates (instruction parallelization, incremental compilation and a variation-aware variant).
  - Hardware: ibmq_16_melbourne, 40 graphs of 12 nodes, 40,960 shots. The metric is approximation-ratio gap. IC beat IP by 8.53%, and VIC beat IC by 7.36%.
  - Reordering also changes the SWAP insertion, so the gates are not fixed.
  - Source: https://par.nsf.gov/servlets/purl/10292780
- **Paulihedral.** Li, Wu, Shi, Javadi-Abhari, Ding, Xie, arXiv:2109.03371 (the task lists it as ASPLOS 2022; the opened page does not name the venue).
  - Block scheduling combined with synthesis, cancellation and mapping.
  - Hardware: IBM 16-qubit Melbourne, 8 QAOA programs, 40,960 shots. Real success probability improved 1.24x on average (up to 1.87x), and ESP improved 2.11x.
  - **There is no ablation that changes order only.**
  - Source: https://www.arxiv.org/pdf/2109.03371
- **2QAN.** Lao and Browne, arXiv:2108.02099.
  - Exploits the freedom to permute 2-local Trotter terms during routing, unitary unification and scheduling.
  - QAOA experiments ran on IBMQ Montreal. The opened page claims the highest fidelity but gives no hardware numbers in the text that was read.
  - Compile-level results: SWAP count down by up to 11.5x. There is no isolated reordering experiment.
  - Source: https://ar5iv.arxiv.org/html/2108.02099
- **Tetris.** Jin, Li, Hua, Hao, Zhou, Huang, Zhang, ISCA 2024 (arXiv:2309.01905).
  - Block scheduling plus cancellation-aware synthesis.
  - **No hardware runs.** Evaluation used compile metrics and Aer noise simulation on 18 benchmarks. Average CNOT count fell 30.68% versus Paulihedral (JW encoding).
  - There is a scheduler-only comparison (Fig. 14), but it reports compile metrics.
  - Source: https://par.nsf.gov/servlets/purl/10532467
- **PHOENIX.** Yang, Ding, Zhu, Chen, Xie, arXiv:2504.03529.
  - Pauli IR grouping, simplification in binary symplectic form (BSF), and ordering of IR groups.
  - **No hardware runs**, only compile metrics, and no ablation for ordering alone.
  - Source: https://arxiv.org/pdf/2504.03529
- **Rustiq.** Goubault de Brugière and Martiel, arXiv:2404.03280, "Faster and shorter synthesis of Hamiltonian simulation circuits."
  - Its heuristics can "maintain or loosen" the ordering of rotations, and it reports up to 4x lower entangling depth.
  - The abstract page mentions no hardware experiments.
  - Source: https://arxiv.org/abs/2404.03280

Theory and simulation of Trotter ordering, with no hardware:

- **Tranter, Love, Mintert, Wiebe, Coveney.** *Entropy* 21, 1218 (2019), arXiv:1912.07555. Classical Trotter-error analysis of term orderings across 44 molecular Hamiltonians. Source: https://www.arxiv.org/abs/1912.07555
- **Childs, Su, Tran, Wiebe, Zhu, "A Theory of Trotter Error."** *PRX* 11, 011020, arXiv:1912.08854. Theory only. Source: https://arxiv.org/abs/1912.08854v2

Relation to PSF-Zero: hardware gains from timing or serialization alone are established for crosstalk (Murali 2020) and for single-gate idle placement (Liu et al. 2026). Every Pauli and Trotter reordering compiler found here also changes gate counts and reports either no hardware ablation or only compile metrics. A pre-registered hardware test that holds the gate multiset fixed and changes only the order of commuting blocks was not found. It would add new evidence beyond these works.

---

## 3. Qiskit level 2/3 becoming very slow when the circuit nearly fills the coupling map (the README's "cliff")

**Status: partly known. The Qiskit developers acknowledge excessive VF2Layout/VF2PostLayout runtime for symmetric circuits on large coupling maps, and have made several fixes. No issue or paper was found that describes a runtime cliff as a function of the number of spare qubits.**

Evidence:

- **Qiskit SDK 2.3 release notes.** The exact-match VF2PostLayout call and trial limits at optimization_level=3 were lowered to "avoid excessive runtimes for highly symmetric trial circuits" mapped to large coupling maps.
  - VF2 `call_limit` now accepts a 2-tuple. `max_trials` "now has much less effect as a runtime limiter."
  - Partial-score tracking "can drastically reduce the cost of scoring layouts" in high-symmetry cases.
  - Source: https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.3
- **Qiskit SDK 2.2 release notes.** VF2PostLayout was re-enabled in the level 3 optimization stage. In 2.2.1, its limits were reduced "to avoid excessive runtimes."
  - SabreLayout now includes a "dense subset" (DenseLayout-style) trial.
  - Source: https://quantum.cloud.ibm.com/docs/api/qiskit/release-notes/2.2
- **Qiskit SDK 2.1 release notes.** VF2PostLayout was added at level 3 in 2.1.0, then removed from the optimization stage in 2.1.2 "to save the runtime cost."
  - Source: https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/2.1
- **Issue #15064, "Potential performance regression ... optimization level 3 in version 2.2.0."**
  - A 20-qubit `efficient_su2` circuit on FakeFez took 0.121 s in 2.1.2 and 2.012 s in 2.2.0.
  - The cause was VF2PostLayout in strict-direction mode, because Fez has many layouts for a 20-qubit line. PR #15068 brought it to 0.209 s.
  - Source: https://github.com/Qiskit/qiskit/issues/15064
- **Issue #14653, "Significant performance regression of transpilation in 2.1.0 compared to 2.0.2."**
  - A batch of `zz_feature_map` circuits (4–28 qubits) on FakeFez at level 3 took about 34.3 s, versus 3.7 s before. Level 2 was unaffected.
  - Maintainers attributed it to VF2PostLayout, recommended level 2 when runtime matters, and closed it as expected behaviour.
  - Source: https://github.com/Qiskit/qiskit/issues/14653
- **Issue #9367, "Long transpile time for simple circuits."**
  - A 4-qubit circuit on ibm_sherbrooke was slow in VF2Layout because it enumerated about 10.3 million candidate layouts. The maintainer pointed to PR #9148.
  - This case is the opposite of the project's: a small circuit on a large device.
  - Source: https://github.com/Qiskit/qiskit/issues/9367
- **rustworkx issue #1679.**
  - The reporter shows that `vf2_mapping` with VF2++ ordering fails to find perfect-matching embeddings on grids of 24 or more nodes within the call limit. With identity order it finds one in under 1 ms.
  - The reporter cites a **42-qubit circuit on a 42-qubit grid (0 spare qubits) taking about 12.8 s at level 3**, and a 38-on-42 case.
  - A maintainer (jakelishman) **disputed** that Qiskit's current VF2Layout uses this code path, because the path moved to Rust in PR #14860. The issue is about a near-full or full device, but its link to Qiskit is contested.
  - Source: https://github.com/Qiskit/rustworkx/issues/1679
- **VF2Layout source (`vf2_layout.py`).** Defines `call_limit`, `time_limit` and `max_trials`. It says nothing about complexity, beyond VF2++ subgraph isomorphism.
  - Source: https://github.com/Qiskit/qiskit/blob/main/qiskit/transpiler/passes/layout/vf2_layout.py
- **SABRE (Li, Ding, Xie, arXiv:1809.02573).**
  - Preprocessing costs O(N^3) for the distance matrix and O(g) for the DAG. The search considers only SWAPs adjacent to the front layer, which avoids exponential search over mappings.
  - There is no analysis of how runtime depends on spare qubits.
  - Sources: https://arxiv.org/abs/1809.02573v2 and https://ar5iv.labs.arxiv.org/html/1809.02573
- **LightSABRE (Zou, Treinish, Hartman, Ivrii, Lishman et al., arXiv:2409.08368).**
  - About 200x faster than the original Qiskit SABRE, with a "release valve" for stuck routing. Runtime scaling was tested on BV circuits up to 19,998 qubits.
  - The summary page says nearly full devices are not covered.
  - Source: https://alphaxiv.org/resources/2409.08368v1 (a third-party summary, not the arXiv PDF)

Relation to PSF-Zero: Qiskit release notes and issues confirm that level 3 runtime is dominated by VF2Layout/VF2PostLayout search in high-symmetry or many-layout situations, with fixes in 2.1.2, 2.2.1 and 2.3. That makes it a plausible mechanism for the project's "cliff." A documented runtime cliff that depends on the number of spare qubits was not found.

---

## Summary table

| # | Topic | Status | Closest prior work |
|---|---|---|---|
| 1 | Pre-registered hardware compiler comparison scored on all cases | **Not found** (no pre-registration; hardware comparisons exist) | Mills et al. 2021 (Quantum 5, 415): tket vs Qiskit on 4 IBM devices, full data released. Benchpress (arXiv:2409.08844): all cases scored but compile-only |
| 2 | Higher hardware fidelity from reordering the same gates | **Partly known** | Murali et al. 2020 (arXiv:2001.02826): same gates, rescheduled, up to 5.6x lower error. Liu et al. 2026 (arXiv:2607.02031): H-gate placement on 2 qubits. Pauli and Trotter reorderers (Paulihedral, 2QAN, Tetris, PHOENIX, Tomesh 2021, Alam 2020) also change gate counts |
| 3 | Qiskit level 2/3 slowdown near a full coupling map | **Partly known** | Qiskit 2.1–2.3 release notes and issues #15064 and #14653 (VF2PostLayout runtime). rustworkx #1679 (42-on-42 case, link to Qiskit disputed) |

---

## Not verified

- Venues listed in the task but not shown on the opened pages: Paulihedral as ASPLOS 2022, 2QAN as ISCA 2022, Murali et al. as ASPLOS 2020 (the page gives an ACM DOI only), SupermarQ as HPCA 2022, Tannu and Qureshi as ASPLOS 2019, and Alam et al. as MICRO 2020 (inferred from a garbled PDF header).
- 2QAN's hardware fidelity numbers on IBMQ Montreal and its number of circuits. The opened ar5iv page was cut off before them.
- Paulihedral's per-benchmark real-system values, which are only in its Fig. 11.
- The LightSABRE details came from an alphaxiv summary page, not the arXiv PDF (section 3).
- Whether the rustworkx #1679 timing (42-on-42 grid, about 12.8 s) applies to current Qiskit. A maintainer disputed the claimed code path.
- Whether the slow cases reported in Qiskit issues depend on the number of spare qubits. No issue reports that dependence.
- The Tomesh et al. (ICRC 2021) hardware provider for ethene, a "7-qubit device", is not named in the extracted text.
- The arXiv API and direct arXiv downloads were blocked (HTTP 403 from the proxy policy), so arXiv metadata was not cross-checked through the API.
