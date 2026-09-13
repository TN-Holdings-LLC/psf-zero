"""benchmarks/verify_determinism_variance.py"""
import time
import hashlib
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import FullPeepholeOptimise, DecomposeBoxes

from psf_compile import compile as psf_compile

def get_circuit_signature(qc: QuantumCircuit) -> str:
    """回路の構造（ゲート種別、パラメータ、対象量子ビット）から一意なハッシュを生成する"""
    ops = []
    for inst in qc.data:
        op_name = inst.operation.name
        qubits = ",".join(str(qc.find_bit(q).index) for q in inst.qubits)
        # 連続パラメータの微小な浮動小数点誤差を丸めて同一構造として扱う
        params = ",".join(
            f"{float(p):.4f}" if isinstance(p, (float, int, np.number)) else str(p) 
            for p in inst.operation.params
        )
        ops.append(f"{op_name}({params})-[{qubits}]")
    
    sig_string = "|".join(ops)
    return hashlib.sha256(sig_string.encode('utf-8')).hexdigest()

def main():
    N_ITERATIONS = 1000
    
    # ターゲットとして固定のSU(4)ユニタリ（大域位相修正済み）を生成
    np.random.seed(42)
    u = random_unitary(4, seed=42).data
    u = u / np.linalg.det(u) ** 0.25 
    
    qc_base = QuantumCircuit(2)
    qc_base.unitary(u, [0, 1])

    records = []

    print(f"Running Determinism & Variance Benchmark ({N_ITERATIONS} iterations)...")
    
    for i in range(N_ITERATIONS):
        # --------------------------------------------------
        # 1. Qiskit (optimization_level=3)
        # --------------------------------------------------
        t0 = time.perf_counter()
        qc_qiskit = transpile(qc_base, basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=3)
        t_qiskit = time.perf_counter() - t0
        
        records.append({
            "iteration": i, "engine": "Qiskit_L3", 
            "time_s": t_qiskit, "depth": qc_qiskit.depth(), 
            "cx_count": qc_qiskit.count_ops().get('cx', 0), 
            "signature": get_circuit_signature(qc_qiskit)
        })

        # --------------------------------------------------
        # 2. TKET (FullPeepholeOptimise)
        # --------------------------------------------------
        tk_circ = qiskit_to_tk(qc_base)
        t0 = time.perf_counter()
        
        DecomposeBoxes().apply(tk_circ) 
        FullPeepholeOptimise().apply(tk_circ)
        qc_tket = tk_to_qiskit(tk_circ)
        t_tket = time.perf_counter() - t0
        
        records.append({
            "iteration": i, "engine": "TKET", 
            "time_s": t_tket, "depth": qc_tket.depth(), 
            "cx_count": qc_tket.count_ops().get('cx', 0), 
            "signature": get_circuit_signature(qc_tket)
        })

        # --------------------------------------------------
        # 3. PSF-Zero
        # --------------------------------------------------
        t0 = time.perf_counter()
        qc_psf = psf_compile(qc_base, verify=False)
        t_psf = time.perf_counter() - t0
        
        records.append({
            "iteration": i, "engine": "PSF-Zero", 
            "time_s": t_psf, "depth": qc_psf.depth(), 
            "cx_count": qc_psf.count_ops().get('cx', 0), 
            "signature": get_circuit_signature(qc_psf)
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1} / {N_ITERATIONS}")

    # 集計と出力
    df = pd.DataFrame(records)
    df.to_csv("data/determinism_variance_2026-09-13.csv", index=False)
    
    print("\n--- Summary ---")
    for engine in ["Qiskit_L3", "TKET", "PSF-Zero"]:
        df_eng = df[df["engine"] == engine]
        mean_time = df_eng["time_s"].mean() * 1000
        std_time = df_eng["time_s"].std() * 1000
        total_time = df_eng["time_s"].sum()
        unique_circuits = df_eng["signature"].nunique()
        
        print(f"[{engine}]")
        print(f"  Time/iter : {mean_time:.3f} ms ± {std_time:.3f} ms")
        print(f"  Cumulative: {total_time:.2f} s")
        print(f"  Variants  : {unique_circuits} unique circuit patterns emitted")

if __name__ == "__main__":
    main()