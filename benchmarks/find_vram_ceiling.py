"""find_vram_ceiling.py -- Addendum 143, Part 1.

Finds the largest n for which lightning.gpu can allocate and run a simple
statevector circuit, at n=25..29, catching out-of-memory failures rather
than crashing. A wall here is a valid, pre-registered result, not a
failure.

Usage (inside psf_zero_wsl_env_312):
    python find_vram_ceiling.py
"""
import pennylane as qml

CANDIDATES = (25, 26, 27, 28, 29)


def try_n(n):
    dev = qml.device("lightning.gpu", wires=n)

    def circuit():
        qml.Hadamard(wires=0)
        for w in range(n - 1):
            qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev)
    return qnode()


def main():
    ceiling = None
    for n in CANDIDATES:
        try:
            val = try_n(n)
            print(f"n={n}: OK  value={val}")
            ceiling = n
        except Exception as exc:
            print(f"n={n}: FAILED  {type(exc).__name__}: {exc}")
            print(f"\nCeiling found: largest n that succeeded = {ceiling}")
            return
    print(f"\nAll candidates up to {CANDIDATES[-1]} succeeded; ceiling is at least {CANDIDATES[-1]}.")


if __name__ == "__main__":
    main()
