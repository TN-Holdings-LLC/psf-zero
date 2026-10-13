"""Exploratory: the ASAP variant of NS1 for one circuit (qaoa_n20_p2 PSF), whose first run was interrupted."""
import json, os, sys, time, importlib.util
sys.path.insert(0, "benchmarks")  # repository form: ns_sim.py is in benchmarks/
import ns_sim as N
from qiskit import qpy
sys.path[:0] = ["benchmarks", "."]
spec = importlib.util.spec_from_file_location("hw1", os.path.join("benchmarks", "hw1_real.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
H.KIT = os.path.join("data", "2026-10-12", "hw_kit")
W = os.path.join("data", "2026-10-12", "hw1")
man = json.load(open(os.path.join(W, "hw1_manifest.json"))); sc = json.load(open(os.path.join(W, "hw1_score.json")))
circs = qpy.load(open(os.path.join(W, "hw1_circuits.qpy"), "rb"))
cal = N.calibration_from_csv(os.path.join("data", "2026-10-12", "hw_analysis", "ibm_kingston_calibrations_2026-10-12T11_15_24Z.csv"))
for p, c in zip(man["pubs"], circs):
    e = man["entries"][p["entry"]]
    if e["part"] == "HW1" and p["repeat"] == 1 and e["test"] == sys.argv[1] and e["arm"] == sys.argv[2]:
        qc, _ = H.build_test(e["test"]); d, _ = H.measured_distribution(H._unrolled(qc))
        t0 = time.time()
        fs, se, dur, nact, ntraj = N.simulate_xeb(c, d, qc.num_qubits, cal, method="asap")
        r = dict(test=e["test"], arm=e["arm"], f_sim=fs, se_sim=se, dur_us=dur * 1e6, active=nact, esp=e["esp"],
                 f_real=sc["F"][f"{e['test']}|{e['arm']}"]["f"], traj=ntraj, secs=round(time.time() - t0, 1))
        print(json.dumps(r), flush=True)
        break
