# Workplace literature check (request 16 of 2026-10-13): residual ZZ on IBM Heron, and how SamplerV2 times circuits that carry no explicit delays (2026-10-13)

**Status: desk research, a follow-up to `heron-idle-literature-2026-10-13.md` (request 3).** Searched 08:49-08:52 CEST
(`TZ=Europe/Zurich date`). Where nothing citable was found, it says so.

## 1. Residual ZZ between neighbouring qubits on Heron: typical value in kHz

**No published, Heron-specific measured value was found** (IBM documentation, arXiv, IBM blog and secondary pages
searched again).

| source | what it says | weight |
|---|---|---|
| Stehlik et al., IBM, "Tunable coupling architecture for fixed-frequency transmons", [arXiv:2101.07746](https://ar5iv.labs.arxiv.org/html/2101.07746) | "ZZ<10 kHz with the coupler in its off state", for a typical pair (detuning 351 MHz); no other pair given | primary, the precursor architecture of Heron, not a Heron device |
| Lange, Heunisch, Fehske, DiVincenzo, Hartmann, "Cross-talk in superconducting qubit lattices with tunable couplers", [arXiv:2504.10298](https://arxiv.org/html/2504.10298) (Quantum Sci. Technol., [IOP](https://iopscience.iop.org/article/10.1088/2058-9565/ae2358)) | theory: with tunable transmon couplers, nearest-neighbour ZZ is reduced "by about an order of magnitude" relative to the same lattice without them, where it reaches "the order of 10 kHz" at their parameters; exact values only in a figure | theory, generic parameters, not IBM |
| AbuGhanem, "IBM quantum computers: evolution, performance, and future directions", [arXiv:2410.00916](https://arxiv.org/html/2410.00916v1) | Heron "virtually eliminating cross-talk" (restating IBM) | secondary, no number |
| Kang et al., crosstalk detector, [arXiv:2502.14225](https://arxiv.org/pdf/2502.14225) | ZZ for ibm_torino only as a dimensionless simulation parameter | no kHz value |

**Reading.** The only primary number is an upper bound of 10 kHz on the precursor architecture; the theory paper
suggests about 1 kHz as a scale for tunable-coupler transmon lattices. A defensible prior for a Heron pair is
therefore "0.1 to 10 kHz, most likely around or below 1 kHz", with no source that fixes it. Scale: 1 kHz gives a
conditional phase of 0.13 rad over 20 us, 10 kHz gives 1.3 rad.

**Consequence for the T2*/ZZ experiment (request 13).** The pre-registered E3 threshold (|zeta| >= 1 kHz) sits in the
middle of the plausible range, so the measurement is informative either way only if its standard error is well below
1 kHz; v2 of the design sizes the run for a standard error of about 0.25 kHz (see `t2star-zz-design-v2-2026-10-13.md`).
A refuted E3 (|zeta| < 0.5 kHz on all three pairs) is a plausible outcome on Heron and would itself be a result.

## 2. How SamplerV2 times circuits without explicit delays (ASAP or ALAP)

**No official statement found.** Pages checked:

- [DynamicalDecouplingOptions](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-dynamical-decoupling-options):
  `scheduling_method`: "Whether to schedule gates as soon as ("asap") or as late as ("alap") possible", default
  "alap"; this applies when DD is enabled (`enable` defaults to False). Nothing is said for DD off.
- [PadDynamicalDecoupling (qiskit-ibm-runtime)](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-scheduling-pad-dynamical-decoupling):
  "You need to call ALAPScheduleAnalysis before running dynamical decoupling" to "guarantee your circuit satisfies
  acquisition alignment constraints". This is about the user-side pass, not the hardware default.
- [ALAPScheduleAnalysis (qiskit-ibm-runtime)](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-scheduling-alap-schedule-analysis):
  a pass "designed to work for the unique scheduling constraints of the dynamic circuits backends"; it does not say
  that the backend applies it to submitted circuits.
- [Scheduling passes overview](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/transpiler-passes-scheduling),
  [Classical feedforward and control flow](https://quantum.cloud.ibm.com/docs/en/guides/classical-feedforward-and-control-flow):
  nothing on the default timing of ordinary circuits.

**What is documented, and answers the question by measurement:** the guide
[Visualize circuit timing](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-runtime-circuit-timing) shows how to
have the service return the executed schedule with the results of a Sampler job:

```python
sampler.options.experimental = {"execution": {"scheduler_timing": True}}
# per PUB: result[i].metadata["compilation"]["scheduler_timing"]["timing"]
# one line per instruction and channel: branch, instruction, channel, T0 (dt), duration (dt), pulse type
```

The guide says "IBM Quantum Compute Service returns the accurate circuit timing information inside the job results
when requested"; the function is marked experimental (preview), and the total circuit time it reports "is NOT the
time used for billing". It needs no extra circuits or shots.

**Recommendation.** Request `scheduler_timing` in every remaining hardware job (HW2 if not yet submitted, with
disclosure since the kit is fixed; HW3; the T2*/ZZ kit, where v2 requests it). From the returned T0 values, a qubit's
last gate before the common measurement shows directly whether the device ran the circuit as soon as possible
(early-finishing qubits wait before measurement) or as late as possible (gates pushed towards the measurement). This
replaces an assumption in NS1/NS2/P1/P2 with an observation, and the predictors that assumed the other timing can be
reported as such.

## 3. Not found (stated so that nobody quotes it)

- A Heron T2*/T2E ratio (unchanged from request 3).
- A per-pair residual ZZ table for any IBM Heron device.
- Any IBM statement on whether measurements of different qubits are aligned to a common time when no barrier is
  present.
