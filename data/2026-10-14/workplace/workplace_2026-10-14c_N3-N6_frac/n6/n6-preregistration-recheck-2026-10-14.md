# N6: Has anyone published a pre-registered comparison of quantum compilers on real hardware? (re-check, 2026-10-14)

**Question.** Is there a published comparison of quantum compilers or transpilers, run on
**real quantum hardware**, where the predictions or analysis plan were fixed before the
run (ideally time-stamped, hashed, or registered on OSF, AsPredicted or similar) and every
case was scored?

**Earlier finding being re-checked.** "Not found". The closest work was Mills, Sivarajah,
Scholten, Duncan, *Application-Motivated, Holistic Benchmarking of a Full Quantum Computing
Stack*, Quantum 5, 415 (2021), arXiv:2006.01273. It compares tket and Qiskit on hardware and
releases the full data, but it is not pre-registered.

**Result: NOT FOUND.** No paper or report was found that combines all three: (a) a compiler
comparison, (b) real QPU execution, and (c) a plan or predictions fixed before the run.
Two works found in this re-check meet (a) and (c) but not (b). One of them (TOQB) appears to
be written by the PSF-Zero authors, so it is not independent prior art (see below).

---

## Method

All checks were made on 2026-10-14 using web search and page fetches only.

### 1. Works citing Mills et al. 2021

- **Semantic Scholar API** (`api.semanticscholar.org/.../arXiv:2006.01273/citations`) and
  **OpenAlex API** (`api.openalex.org/works/doi:10.22331/q-2021-03-22-415`): **not reachable.**
  Shell access was refused by the egress proxy (HTTP 403 on CONNECT). The page-fetch tool
  refused these URLs because of its URL-provenance rule. The Semantic Scholar web page for
  the paper (found by search) returned a fetch error.
- **Used instead: the "Cited by" list on the Quantum journal page**
  (https://quantum-journal.org/papers/q-2021-03-22-415/). It combines Crossref and SAO/NASA
  ADS data, last updated 2026-10-13, and notes that the list may be incomplete.
  - **54 entries scanned.** Two pairs are journal and arXiv versions of the same work, so
    there are **52 distinct works**.
  - About 13 entries show no title (mostly conference and book chapters). Those were judged
    by venue only and were not opened.
- Entries that looked relevant to compiler or hardware benchmarking were opened or checked
  through their abstract pages:
  - Dimitrijevs et al. 2026 (arXiv:2602.16014)
  - Acuaviva et al. (arXiv:2407.10941v4), full text
  - Proctor et al. *Benchmarking quantum computers* (arXiv:2407.08828), full text
  - Also considered from the list: QuSquare (arXiv:2512.19665), MQT Predictor, Lubinski
    et al. (QED-C), Kordzanganeh et al., Hines & Proctor, Seritan et al. (arXiv:2508.05754),
    Siekierski et al. (arXiv:2511.02134), Bandic et al., Salm et al. 2021.
  - **None of the titled entries is a pre-registered hardware compiler comparison.** Most
    are device benchmarks, benchmark-suite or methodology papers, or reviews.

### 2. Web and arXiv searches

About 16 search queries were run, covering:

- `"pre-registered" quantum compiler benchmark hardware`
- `"preregistered" OR "preregistration" quantum computing transpiler experiment arXiv`
- `"registered report" quantum computing benchmark`
- `quantum benchmarking "pre-registration" methodology hardware results`
- `osf.io quantum computing preregistration IBM quantum hardware experiment`
- `aspredicted OR "osf.io" "quantum circuit" compilation benchmark preregistration`
- `"predictions" "before running" quantum compiler comparison real hardware tket qiskit`
- `"pre-registered" "quantum" "transpiler" OR "qubit mapping" OR "routing"`
- `arXiv quant-ph 2026 "preregistered" hardware IBM Heron ... predictions frozen`
- `quantum compiler "blind" benchmark ... prediction hash commitment`
- comparisons of Qiskit, tket and BQSKit on real IBM hardware (2025–2026)

About 15 of the hits were opened and checked for pre-registration language and for hardware
execution.

### 3. Methodology and standards documents

Each was checked for "pre-registration", "registered report", or fixing predictions or
analysis in advance:

- Lorenz et al., *Systematic benchmarking of quantum computers: status and recommendations*
  (arXiv:2503.04905), full text including references
- Proctor et al., *Benchmarking quantum computers* (arXiv:2407.08828)
- Acuaviva et al. (arXiv:2407.10941v4), full text in two parts
- DARPA QBI program page
- QED-C pages
- *Quantum Computer Benchmarking: An Explorative Systematic Literature Review*
  (arXiv:2509.03078), first 100k characters

---

## Result: not found

**No source meets all criteria.** The closest works are listed below, with the criterion
each one fails.

### Closest works

| Work | Link | Why it does not qualify |
|---|---|---|
| Mills, Sivarajah, Scholten, Duncan, Quantum 5, 415 (2021) | https://quantum-journal.org/papers/q-2021-03-22-415/ | Hardware tket vs Qiskit with full data on Zenodo, but **not pre-registered**. Still the closest independent match. |
| Yang, Li, Deng, *Representation-Dependent Recoverability in Quantum Compilation* (arXiv:2609.29210, ~Sept 2026) | https://arxiv.org/html/2609.29210 | Uses a "preregistered protocol frozen before the run", tests Qiskit and TKET, and reports two failed preregistered predictions. But there is **no QPU execution**: it measures compiler outputs in software, and no OSF entry or hash was seen. It is also not a performance comparison of compilers. |
| TOQB (Task-Oriented Quantum Benchmark), TN-Holdings-LLC, GitHub, v0.1 (contamination notice dated 2026-10-10) | https://github.com/TN-Holdings-LLC/TOQB | Predictions and scoring scripts are hash-locked before each scored run, and all failures are published. It compares Qiskit O1/O2/O3, TKET and PSF-Zero. But results so far are **simulation on fake IBM backends only**, with hardware runs only "planned". It is not a paper. **The README says the TOQB authors also wrote PSF-Zero**, so this looks like the user's own work, not independent prior art. |
| Pati & Simmhan, *Per-Phase Fidelity Attribution for Quantum Compilers using HBR Decomposition* (arXiv:2605.07876) | https://arxiv.org/html/2605.07876 | Compares Qiskit, PennyLane and TKET. Only 2 circuits ran on IBM Fez, the rest on FakeFez. Seed and shots were fixed in advance, but there is **no registered analysis plan**. Taken from the earlier W1 note, not re-opened today. |
| Nation et al., Benchpress (arXiv:2409.08844; Nat. Comput. Sci. 2025) | https://arxiv.org/html/2409.08844 | Large multi-SDK compiler benchmark. It does **not run circuits on hardware**: "we do not execute the resulting circuits on hardware". No pre-registration. |
| Kharkov et al., Arline Benchmarks (arXiv:2202.14025) | https://arxiv.org/pdf/2202.14025 | Compiler comparison using **compile metrics only**. Not pre-registered. |
| Merilehto, *A 200-Line Python Micro-Benchmark Suite for NISQ Circuit Compilers* (arXiv:2509.16205) | https://arxiv.org/html/2509.16205v1 | Laptop and local simulator only, no pre-registration. Its Cirq and Braket paths appear to return the input circuit uncompiled. |
| Sharma & Lau, *From Circuits to Hardware* (arXiv:2607.11637) | https://arxiv.org/html/2607.11637v1 | Real hardware, but only one compilation was run on the QPU. The other compilation settings were offline counterfactuals. No pre-registration. |
| Dimitrijevs, Palchiha, Yakaryilmaz, *Benchmarking the Lights Out Problem on Real Quantum Hardware* (arXiv:2602.16014) | https://www.arxiv.org/abs/2602.16014 | Cites Mills et al. and runs on real hardware (IBM, IQM), but compares **devices, not compilers**. No pre-registration. |
| Eker et al., QANTIS (arXiv:2607.06760) | https://arxiv.org/abs/2607.06760 | IBM Heron hardware, but compares amplification strategies, **not compilers**. No pre-registration is mentioned. |
| Salm et al. 2021, *Automating the Comparison of Quantum Compilers* | https://www.iaas.uni-stuttgart.de/publications/Salm2021_CompilerComparison.pdf | Cites Mills et al. Compile-time comparison only, **no hardware results**, not pre-registered. Taken from the earlier W1 note. |
| QED-C / Basu, *Testing of fidelity and execution run time hypotheses using QED-C suite* | https://quantumconsortium.org/?p=4648 | The title mentions hypotheses, but the accessible page does not say they were registered in advance or that compilers were compared. Full text was not accessible. |

### Methodology and standards documents

None of them recommends or reports pre-registration:

- **Lorenz et al. 2025 (arXiv:2503.04905):** The word "pre-registration" does not appear.
  The closest it comes is asking for evaluation code, documentation and hosted results that
  allow "comparison and cross-checks" later.
- **Proctor et al. (arXiv:2407.08828):** No pre-registration. Box 4 covers how compilation
  choice affects scores and warns that benchmarks can be "gamed".
- **Acuaviva et al. (arXiv:2407.10941v4):** No pre-registration. It recommends fixing allowed
  compilation rules in advance (SPEC-style base and peak runs) for reproducibility. That is
  a fixed protocol, not registered predictions.
- **DARPA QBI:** Relies on independent verification and validation (IV&V), described as
  "unbiased third-party verification and validation". The program page does not mention
  pre-registration or pre-specified predictions.
- **arXiv:2509.03078 (systematic review):** No registered review protocol, and no primary
  study is reported as pre-registered.

---

## Limits of this search

- **Citation coverage is partial.** Semantic Scholar and OpenAlex were not reachable, because
  of the proxy block and the fetch tool's provenance rule. The citing list scanned is the
  Crossref+ADS list on the Quantum page (52 distinct works), which says itself that it may be
  incomplete. About 13 untitled entries were not opened. Google Scholar was not tried.
- **Full texts were not read for every hit.** Several judgements rest on abstract pages
  (arXiv:2508.10781, arXiv:2511.04608, arXiv:2607.03275). Long PDFs were read in parts, and
  arXiv:2509.03078 only for its first ~100k characters.
- **Search engine limits.** Searches were US-only and keyword-based. Pre-registration stated
  only in a supplement, a GitHub repository or an OSF page without the word "compiler" could
  be missed. OSF and AsPredicted were not searched directly; only web search was used.
- **Some hits could not be checked.**
  - arXiv:2609.09495 (*Certifying bipartite entanglement ... corrected QAOA cost layer*)
    appeared under an OSF/preregistration query, but its PDF had no extractable text. By its
    title it is not a compiler comparison.
  - TOQB's `prereg/` folder could not be listed (robots.txt and no GitHub API access), so its
    contents and dates rest on the README.
- **Dates.** arXiv:2609.29210 had no version stamp in the fetched text. Its date (~Sept 2026)
  is inferred from the arXiv ID.
