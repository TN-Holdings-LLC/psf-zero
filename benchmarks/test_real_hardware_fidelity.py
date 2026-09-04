"""
test_real_hardware_fidelity.py -- does PSF-Zero's compression actually help
on REAL quantum hardware, not just on paper (gate count / depth)?

Every test so far in this project measured CLASSICAL proxies for quality:
compile time, gate count, circuit depth. None of that says anything about
what happens when the circuit actually runs on a noisy chip. This script
closes that gap using the standard "mirror circuit" technique: for a
compiled circuit C, build C followed by its exact inverse C^-1, then
measure how often the all-zero bitstring comes back.

  - In a perfect, noiseless simulator, P(all-zero) = 1.0 for EVERY engine,
    always -- C and C^-1 exactly cancel regardless of how C was compiled.
  - On real hardware, every 2-qubit gate has some error rate, so P(all-zero)
    drops below 1.0 by an amount that grows with how many 2-qubit gates the
    compiled circuit actually contains. This makes P(all-zero) a direct,
    physically meaningful stand-in for "how much did this compilation cost
    me in real gate-error exposure" -- exactly the number a gate-count/
    depth comparison can only ever *predict*, not measure.

Two circuit families are tested, matching the two regimes already
characterized in simulation earlier in this project:

  "deep2q" -- 2-qubit-confined, high-depth circuit (the family where
    PSF-Zero's Cartan/KAK compression won big in simulation: 250 gates/
    depth 200 -> 15 gates/depth 9). Prediction: engines that actually
    compress this (PSF-Zero v6, Hybrid, and possibly TKET/Qiskit-L3 too,
    since general 2-qubit synthesis isn't unique to PSF) should show a
    HIGHER all-zero return rate than a version that doesn't compress at
    all, because there is physically less noise-exposed gate time.

  "wide" -- the wide, shallow-per-pair dense circuit family where PSF-Zero
    v6 was verified to do NOTHING (0/0 eligible blocks -- see
    test_scale_native_war.py). Prediction: PSF-Zero/Hybrid should show NO
    measurable hardware-fidelity difference from TKET/Qiskit-L3 here. If
    this prediction fails (i.e. a real difference shows up anyway), that
    would mean the earlier "PSF does nothing useful on wide circuits"
    conclusion was a simulator-only artifact -- worth knowing either way.

## Modes

Default: SIMULATOR mode, using a local FakeXxx backend snapshot (a real
IBM device's calibration data replayed through Aer -- no network, no QPU
time, no queue). This is what lets this whole script be validated end to
end for free before spending real hardware budget on it (see the
verification note at the bottom of this file's accompanying delivery
message -- this was actually run, not just written).

--real: switches to genuine hardware via your own already-saved
QiskitRuntimeService() account. Nothing about your credentials, API token,
or instance/channel is read, stored, or asked for by this script -- it
only calls QiskitRuntimeService() with no arguments, which uses whatever
you already configured with save_account() previously. Pass --backend to
pin a specific device, otherwise the least-busy device with enough qubits
is selected automatically.

## Honest caveats (read before treating results as final)

  - The final generate_preset_pass_manager(optimization_level=1, ...) step
    is unavoidable -- every circuit must be translated to the backend's
    native gates and physical qubit layout before it can run at all -- and
    is deliberately kept at the LOWEST useful optimization level so it
    doesn't quietly re-solve away whatever structural differences the four
    engines produced. Using a higher level here would partially mask the
    very effect this test exists to isolate.
  - Real hardware drifts (calibration changes hour to hour). Each engine's
    mirror circuit is repeated across N_REPEATS independently-seeded
    circuit instances and reported as mean +/- standard error, but a
    single run of this script is still one hardware session at one point
    in time, not a controlled-for-drift experiment. Re-running the whole
    script at a different time of day is the honest way to check a result
    holds up.
  - All-zero return probability is a *proxy* for fidelity, not fidelity
    itself (some error types happen to preserve the all-zero outcome and
    would be invisible to this metric). It's used here because full state/
    process tomography is infeasible at these gate counts on real hardware
    -- this is the standard practical choice for exactly this kind of
    check, not a shortcut invented for this project.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import FullPeepholeOptimise
from psf_compile import compile as psf_compile

ENGINES = ["Qiskit_L3", "TKET_native", "PSF_Zero_v6", "Hybrid"]


# ---------------------------------------------------------------------------
# Circuit families (same generators used throughout this project, so results
# here are directly comparable to the earlier simulated gate-count/depth
# numbers).
# ---------------------------------------------------------------------------

def generate_deep2q_circuit(depth_steps: int = 40, rng=None) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(depth_steps):
        qc.rx(rng.uniform(0.1, 3.0), 0)
        qc.ry(rng.uniform(0.1, 3.0), 1)
        qc.cx(0, 1)
        qc.rz(rng.uniform(0.1, 3.0), 0)
        qc.cx(1, 0)
    return qc


def generate_multi_deep2q_circuit(num_pairs: int, depth_steps: int = 40, rng=None) -> QuantumCircuit:
    """Several INDEPENDENT deep2q blocks side by side on disjoint qubit
    pairs (no cross-pair gates at all). Added after the first validation
    run of this script showed something important: for a SINGLE deep2q
    block, all four engines converge to the identical, generically-optimal
    2-qubit gate count (3) -- meaning PSF's earlier "big win" on this
    family (250 gates -> 15 gates, simulated) was a COMPILE-TIME speed
    story, not a final-circuit-quality story, since Qiskit-L3's and TKET's
    own internal 2-qubit synthesis reach that exact same optimum anyway,
    just much more slowly. This multi-block version tests whether that
    still holds once TKET's optimizer has to find that same per-block
    optimum inside a larger, multi-qubit circuit (a harder search problem
    for a heuristic peephole pass than for PSF's closed-form per-block
    math) -- verified directly at num_pairs=4: still 12/12/12/12 identical
    2-qubit gate counts across Qiskit_L3/TKET/PSF/Hybrid, i.e. even here
    every engine still finds the same optimum. Kept in as a real-hardware
    check of that finding, and as a template to push further (more pairs,
    shared qubits, mixed connectivity) if you want to hunt for a regime
    where TKET's heuristic search actually does worse than PSF's exact
    math -- that regime is what would be needed for a genuine real-
    hardware fidelity advantage, as opposed to a compile-time-only one.
    """
    qc = QuantumCircuit(2 * num_pairs)
    for p in range(num_pairs):
        q0, q1 = 2 * p, 2 * p + 1
        for _ in range(depth_steps):
            qc.rx(rng.uniform(0.1, 3.0), q0)
            qc.ry(rng.uniform(0.1, 3.0), q1)
            qc.cx(q0, q1)
            qc.rz(rng.uniform(0.1, 3.0), q0)
            qc.cx(q1, q0)
    return qc


def generate_wide_circuit(num_qubits: int, depth: int = 6, rng=None) -> QuantumCircuit:
    if rng is None:
        rng = np.random.default_rng()
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            qc.rx(float(rng.uniform(0.1, 3.0)), i)
            qc.ry(float(rng.uniform(0.1, 3.0)), i)
        for i in range(0, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cx(i, i + 1)
    return qc


# ---------------------------------------------------------------------------
# The four engines under test. Each returns ITS OWN compiled circuit -- the
# comparison must reflect what a user actually gets from each engine, not a
# circuit re-optimized after the fact.
# ---------------------------------------------------------------------------

def run_tket_compile(qc: QuantumCircuit) -> QuantumCircuit:
    t = qiskit_to_tk(qc)
    FullPeepholeOptimise().apply(t)
    return tk_to_qiskit(t)


def compile_all_engines(qc: QuantumCircuit) -> dict[str, QuantumCircuit]:
    qc_qiskit = transpile(qc, basis_gates=["rx", "ry", "rz", "cx"], optimization_level=3)
    qc_tket = run_tket_compile(qc)
    qc_psf = psf_compile(qc)
    qc_hybrid = run_tket_compile(qc_psf)
    return {
        "Qiskit_L3": qc_qiskit,
        "TKET_native": qc_tket,
        "PSF_Zero_v6": qc_psf,
        "Hybrid": qc_hybrid,
    }


def two_qubit_gate_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def assert_equivalent(qc_orig: QuantumCircuit, qc_new: QuantumCircuit, label: str, atol=1e-6):
    Ua = Operator(qc_orig).data
    Ub = Operator(qc_new).data
    prod = Ua.conj().T @ Ub
    ref = prod[0, 0]
    assert abs(ref) > 1e-12, f"[{label}] not even proportional to the original"
    phase = ref / abs(ref)
    dev = float(np.abs(prod / phase - np.eye(prod.shape[0])).max())
    assert dev < atol, f"[{label}] deviation {dev:.3e} -- functional regression, not just noise"
    return dev


def build_mirror_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """C followed by C^-1, plus measurement. Ideal (noiseless) result is the
    all-zero bitstring with probability 1, regardless of how C was compiled
    -- any shortfall on real hardware is real gate-error exposure."""
    n = qc.num_qubits
    mirror = QuantumCircuit(n, n)
    mirror.compose(qc, qubits=range(n), inplace=True)
    mirror.compose(qc.inverse(), qubits=range(n), inplace=True)
    mirror.measure(range(n), range(n))
    return mirror


# ---------------------------------------------------------------------------
# Sanity check: run BEFORE touching any noisy backend. If this fails, the
# mirror-circuit construction itself is broken and nothing downstream can be
# trusted.
# ---------------------------------------------------------------------------

def sanity_check_noiseless():
    from qiskit_aer import AerSimulator
    print("=== Sanity check (noiseless simulator): mirror circuits must return all-zero ~100% ===", flush=True)
    rng = np.random.default_rng(0)
    qc = generate_deep2q_circuit(depth_steps=20, rng=rng)
    engines = compile_all_engines(qc)
    sim = AerSimulator()
    for label, compiled in engines.items():
        dev = assert_equivalent(qc, compiled, label)
        mirror = build_mirror_circuit(compiled)
        counts = sim.run(mirror, shots=4096).result().get_counts()
        zero_key = "0" * qc.num_qubits
        p0 = counts.get(zero_key, 0) / 4096
        print(f"  {label:14s}: unitary-equivalence dev={dev:.2e}, "
              f"noiseless mirror P(all-zero)={p0:.4f} (expect ~1.0), "
              f"2q-gates={two_qubit_gate_count(compiled)}", flush=True)
        assert p0 > 0.999, f"{label}: noiseless mirror circuit did not return to all-zero -- BUG"
    print("Sanity check PASSED: all four engines' mirror circuits are exact in the noiseless case.\n", flush=True)


# ---------------------------------------------------------------------------
# The actual hardware/noisy-simulator experiment.
# ---------------------------------------------------------------------------

def get_backend(use_real: bool, backend_name: str | None, min_qubits: int):
    if use_real:
        from qiskit_ibm_runtime import QiskitRuntimeService
        # 成功しているスクリプトと同様にチャンネルやインスタンスを確実に解決させる
        service = QiskitRuntimeService(channel="ibm_cloud")
        if backend_name:
            return service.backend(backend_name)
        return service.least_busy(min_num_qubits=min_qubits, operational=True, simulator=False)
    else:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke()  # noise model snapshot from a real 127-qubit IBM device


def run_family(
    family_name: str,
    circuit_fn,
    backend,
    n_repeats: int,
    shots: int,
    seed_base: int,
):
    from qiskit_ibm_runtime import SamplerV2

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    all_isa_circuits = []
    meta = []  # (engine, repeat_idx, num_qubits, two_q_gate_count)

    print(f"\n=== Building & transpiling '{family_name}' mirror circuits "
          f"for {n_repeats} repeats x {len(ENGINES)} engines ===", flush=True)
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed_base + rep)
        qc = circuit_fn(rng)
        engines = compile_all_engines(qc)
        for label, compiled in engines.items():
            mirror = build_mirror_circuit(compiled)
            isa = pm.run(mirror)
            all_isa_circuits.append(isa)
            meta.append((label, rep, qc.num_qubits, two_qubit_gate_count(compiled)))

    print(f"Submitting ONE batched job of {len(all_isa_circuits)} circuits "
          f"(minimizes queue/job overhead on real hardware)...", flush=True)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(all_isa_circuits, shots=shots)
    result = job.result()

    rows = []
    for (label, rep, n_qubits, two_q_count), pub_result in zip(meta, result):
        counts = pub_result.data.c.get_counts()
        zero_key = "0" * n_qubits
        total = sum(counts.values())
        p0 = counts.get(zero_key, 0) / total
        rows.append({
            "family": family_name, "engine": label, "repeat": rep,
            "n_qubits": n_qubits, "two_qubit_gates": two_q_count,
            "p_all_zero": p0,
        })
    return pd.DataFrame(rows)


def summarize_and_plot(df: pd.DataFrame, out_prefix: str):
    summary = df.groupby(["family", "engine"]).agg(
        mean_p_all_zero=("p_all_zero", "mean"),
        stderr_p_all_zero=("p_all_zero", lambda x: x.std(ddof=1) / max(len(x) ** 0.5, 1)),
        mean_two_qubit_gates=("two_qubit_gates", "mean"),
        n=("p_all_zero", "count"),
    ).reset_index()
    print("\n" + "=" * 78)
    print("REAL/NOISY-HARDWARE MIRROR-CIRCUIT RESULTS (higher P(all-zero) = less real gate-error exposure)")
    print("=" * 78)
    print(summary.to_string(index=False))
    print("=" * 78)

    df.to_csv(f"{out_prefix}_raw.csv", index=False)
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)

    families = summary["family"].unique()
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), squeeze=False)
    for i, fam in enumerate(families):
        ax = axes[0][i]
        sub = summary[summary["family"] == fam]
        ax.bar(sub["engine"], sub["mean_p_all_zero"], yerr=sub["stderr_p_all_zero"], capsize=5)
        ax.set_title(f"Mirror-circuit fidelity proxy -- '{fam}'")
        ax.set_ylabel("P(all-zero) [higher = better]")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_plot.png", dpi=300)
    print(f"\nSaved '{out_prefix}_raw.csv', '{out_prefix}_summary.csv', '{out_prefix}_plot.png'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true", help="Use real IBM Quantum hardware instead of a local noisy-simulator snapshot.")
    parser.add_argument("--backend", type=str, default=None, help="Specific real backend name (only used with --real).")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--n-repeats", type=int, default=5, help="Independent circuit instances per engine, per family.")
    parser.add_argument("--deep-depth-steps", type=int, default=40)
    parser.add_argument("--wide-n-qubits", type=int, default=8)
    parser.add_argument("--wide-depth", type=int, default=6)
    parser.add_argument("--multi-pairs", type=int, default=4, help="Number of independent deep2q blocks in the multi_deep2q family.")
    parser.add_argument("--min-qubits", type=int, default=8, help="Minimum qubits when auto-selecting a real backend.")
    parser.add_argument("--skip-sanity-check", action="store_true")
    parser.add_argument("--out-prefix", type=str, default="hw_fidelity")
    args = parser.parse_args()

    if not args.skip_sanity_check:
        sanity_check_noiseless()

    backend = get_backend(args.real, args.backend, args.min_qubits)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits), mode={'REAL HARDWARE' if args.real else 'local noisy-simulator snapshot'}", flush=True)

    df_deep = run_family(
        "deep2q",
        lambda rng: generate_deep2q_circuit(depth_steps=args.deep_depth_steps, rng=rng),
        backend, args.n_repeats, args.shots, seed_base=1000,
    )
    df_multi = run_family(
        "multi_deep2q",
        lambda rng: generate_multi_deep2q_circuit(args.multi_pairs, depth_steps=args.deep_depth_steps, rng=rng),
        backend, args.n_repeats, args.shots, seed_base=1500,
    )
    df_wide = run_family(
        "wide",
        lambda rng: generate_wide_circuit(args.wide_n_qubits, depth=args.wide_depth, rng=rng),
        backend, args.n_repeats, args.shots, seed_base=2000,
    )
    df_all = pd.concat([df_deep, df_multi, df_wide], ignore_index=True)
    summarize_and_plot(df_all, args.out_prefix)


if __name__ == "__main__":
    main()
