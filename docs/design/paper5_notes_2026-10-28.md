# Paper 5: material collected on 2026-10-28 (not a draft)

Owner's decisions of 2026-10-28:

- the Occam argument goes into Paper 5, not into Paper 4 v1.2;
- the direction after flow L is the workplace's (A), the kicked Ising ring with an exact answer (flow M, done) plus
  device point D3 (declared, not run).

Records: TOQB `results/flow-l-2026-10-28` and `results/flow-m-2026-10-28`; declarations in TOQB
`prereg/drafts-2026-10-28/`.

## 1. Discussion paragraph: formula schedules and selection bias (the workplace's Occam argument, corrected at home)

Status: a theoretical remark with no evidence of its own. The 2026-10-22 device day is one anecdote, not evidence.

Draft wording:

> PSF's schedules are fixed by formula. They are never chosen as the best of several candidates by a calibration-,
> error-product- or simulation-based score; the only use of the calibration is a threshold that excludes failed
> elements before placement. For a compiler that selects the best of K candidates by an estimate with error scale
> sigma (sub-Gaussian, any correlation), the selected circuit's predicted quality is optimistic by up to
> sigma sqrt(2 ln K) in expectation. This is the optimizer's curse [Smith and Winkler 2006]; it has the structure of
> the Occam bound of learning theory [Blumer et al. 1987]. The optimism vanishes for K = 1, which holds for the formula
> schedules on a day with no failed element.
>
> This is a statement about the reliability of a prediction, not about which compiler keeps more signal; the latter is
> measured separately. Fewer two-qubit gates did not always mean more signal:
>
> - flow G: at n = 18, the zipper placed elsewhere kept about as much as the rule's holes, with almost twice the gates;
> - flow K: on the same qubits, the rule kept 1.13-1.81 times the zipper's signal.

Corrections to the workplace's memo (occam_argument_2026-10-28.md, verified at home):

- **K = 1 holds only on a day with no failed element.** `fold:rule` uses the calibration as a failed-element threshold.
  That is a cut, not an argmax, but it is not estimate-free.
- **The direct reference is the optimizer's curse.** Blumer et al. is an analogy.
- **Flow K belongs beside flow G** in the "fewer gates is not always better" point.
- **The table of sqrt(2 ln K) and sqrt(2 ln(2K/delta)) values was recomputed and is correct.**

Testable form (not runnable under the current rules):

- On a day different from the selection snapshot, a search-selected circuit falls short of its prediction by more
  than a formula circuit does.
- It needs a search arm. The 10/27 decision keeps Qiskit out of long runs; a short device job is outside that decision.

References:

- J. E. Smith and R. L. Winkler, "The optimizer's curse: Skepticism and postdecision surprise in decision analysis",
  Management Science 52(3), 311-322 (2006), doi:10.1287/mnsc.1050.0451.
- A. Blumer, A. Ehrenfeucht, D. Haussler and M. K. Warmuth, "Occam's razor", Information Processing Letters 24,
  377-380 (1987).

## 2. Other material for Paper 5 (from 2026-10-28)

- **Flow L (twin).** The Clifford proxy diverges from the random classifier's lam with depth: ratio 0.92, 0.75, 0.56
  at L = 8, 16, 24.
- **Flow M (twin).** The kicked Ising ring (J = pi/4, h = 0.6, exact free-fermion reference) is CLIFFORD-SIDE at
  L = 16 and 24 and keeps 0.046 +- 0.010 at L = 100.
- **The workplace's noisy free-fermion recursion.** It is exact under Pauli noise; reproduced at home, 6.1e-16 against
  the density matrix at n = 4-6.
  - It explains the slow tail: weight-2 conserved modes decay as (1 - 4p/3)^2 per layer.
  - Caveat found at home: the slope over 24 inputs is unstable at n = 40-80 (it was −0.047 and 0.70 in the formula).
- **The workplace's slow-mode probe.** Exact 2-local conserved charges; their decay is (1 - 4p/3)^(2L) under uniform
  depolarizing, reproduced at n = 6.
  - It is a noise-characterisation tool, a crowded field (layer fidelity, cycle benchmarking).
  - Not run; a literature check comes first.
- **Prior art to read in full.** Harris, Lively and Schuhmacher, arXiv:2603.10224 (verifiable benchmark circuits for
  QEM bias, a 100-qubit kicked Ising chain on ibm_fez). Its non-Clifford benchmark construction may coincide with the
  J = pi/4 free-fermion idea.
