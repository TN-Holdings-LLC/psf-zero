# Idea memo: sim-to-real selection (S2R) -- many candidates in simulation, a few on the device (2026-10-16)

**Status: idea and design draft, not built, not pre-registered.** The owner's idea (2026-10-16, about 16:31 CEST:
"from PSF-Zero through PennyLane, 100,000 in parallel on GPUs, 100,000 answers from the device, looped a few times"),
reshaped by the authors so that it fits the QPU budget. Waiting for three decisions by the owner (section 6).

## 1. Why the idea as first stated does not fit

At IBM's price model used in TOQB (0.036 s per circuit + 400 shots x (0.27 ms + duration)), one PSF-Zero circuit of
the 6-qubit, 8-layer classifier costs 0.1475 s. 100,000 circuits are 4.1 hours of QPU, about $23,600, per loop. The
owner's real budget, 2 minutes ($192), buys about 810 circuits. Compiling is not the limit (a template bind is under
1 ms); QPU time is.

## 2. The question

Candidates chosen by a cheap simulator are sent to the expensive device in small numbers. **Do PSF-Zero's shorter
circuits keep "good in simulation" closer to "good on the device", so that the same device budget reaches a better
model?** This extends the reading of the budget and long runs (Addenda 509-511): PSF-Zero does not make a smarter
model; it gets the same model for less QPU money.

## 3. One loop

1. Spread N candidates (parameter vectors) around the current best (N = 100,000 is the goal).
2. Evaluate all of them in the simulator (noisy, on 8-16 training points).
3. Evaluate only the top k (8-16) on the "device", billed at IBM's price.
4. Update the search distribution from the device's ranking (an evolution strategy, e.g. CMA-ES style).
5. Repeat until the device budget ($192) is spent: with k = 16 and 8 training points, 128 circuits, about 19 s of QPU
   per loop, about 6 loops.

## 4. The "device" before the device

If the simulator and the "device" share one noise model, there is nothing to transfer. The simulator is TOQB's FakeFez
noise model; the stand-in device is a real calibration (the Marrakesh snapshot) with mismatches the simulator does not
know: T1/T2 off by up to 30%, a systematic over-rotation, ZZ crosstalk between neighbours. Only the last stage uses
the real device.

## 5. Arms, baseline and measures

- **Arms:** PSF-Zero and Qiskit level 2, the same compiled circuit in the simulator and on the "device".
- **Baseline:** no simulator, device-only training (the budget run's design).
- **Display:** the "device" test accuracy reached for the same device budget.
- **Essential:** whether the simulator's ranking holds on the "device" (rank correlation of the top k), and the device
  dollars to a target accuracy.

| stage | size | where | purpose |
|---|---|---|---|
| 0 | N = 10,000, 20 seeds per arm | home PC or the authors' cloud (no GPU) | is there an effect at all; stop here if not |
| 1 | N = 100,000 | a rented GPU pod (cost estimated and approved first) | does the effect grow with N |
| 2 | small k, 2 minutes | the real device | only if stages 0-1 show a difference |

Stage 0 runs on TOQB's fast engine (new: the candidate generator and selection, the stand-in device's noise model;
about half a day of work). Stage 1 needs the engine on a GPU. PennyLane is not needed.

## 6. Decisions the owner is asked for (before anything is built)

1. The task: today's classifier, or a regression on the value of <Z>.
2. The stand-in device's mismatches: T1/T2, over-rotation and ZZ crosstalk, or others.
3. Stage 0's size: N = 10,000, k = 16, $192, 20 seeds, or others.

## 7. The authors' honest expectation

Plausible but not more: shorter circuits should suffer less from a noise model's errors, so the simulator's choice
should transfer better. It may also be small for both compilers, and the long run showed the advantage fading once
training converges; so the effect, if any, is expected within small budgets. Stage 0 decides cheaply.
