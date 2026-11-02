"""Five-minute tour of psf-os on a fake device (no IBM account needed).

SIMULATION ONLY. The defects below are planted by this script into an Aer simulation of IBM's FakeFez; psf-os is
not told where they are. Nothing printed here is a device result. The device results are in Papers 5 and 6 and in the
TOQB records (see README.md).

What it does, using psf-os's own functions only (no new rule or threshold):
  1. reads FakeFez's coupling graph and failed elements (no quality number);
  2. plants a two-qubit depolarizing error (probability 0.15 after every CZ) on two couplers, one on the standard
     n = 40 ring and one on the standard n = 72 ring;
  3. scans twice (the mirror echo per ring site; at its Clifford point h = pi/4 so that a laptop can simulate 72
     qubits; the device scan uses h = 0.6), and ingests both scans into the registry: the planted paths become LISTED;
  4. solves two practice problems with exact answers, P2 (cluster-state parity) and P1 (random Clifford circuit read
     through output stabilizers), at n = 40 on the standard ring and on the ring placed around the registry, with the
     same planted errors, plus the sentinel on the avoiding ring;
  5. prints the scores side by side.

Usage:  pip install -e <psf-os>[sim]   then   python quickstart_os.py [OUT_DIR]
"""
import json
import os
import sys
import time

from psf_os import device as Dv
from psf_os import job as J
from psf_os import plan as P
from psf_os import registry as R
from psf_os import sentinel as Se
from psf_os import sim

BANNER = "SIMULATION with planted defects on FakeFez -- not a device result"
SCAN = dict(kind="echo_clifford", D=24, shots=1000)        # the scan at its Clifford point (simulable at n = 72)
SENTINEL = dict(kind="echo_clifford", D=24)
PLANT = dict(p=0.15, kind="depol2")
PROBLEMS = [("P2 cluster parity, n = 40", dict(kind="cluster_parity", n=40, shots=1000), "right"),
            ("P1 random Clifford (k = 4), read stabilizers, n = 40", dict(kind="clifford_stab", n=40, k=4, seed=1,
                                                                            shots=1000), "right")]


def rings_of(runs):
    out = {}
    for f in sorted(x for x in os.listdir(runs) if x.endswith(".json")):
        r = json.load(open(os.path.join(runs, f)))
        out[r["task"]["n"]] = [int(r["echo"]["measured_qubits"][str(j)]) for j in range(r["task"]["n"])]
    return out


def main(out):
    t0 = time.time()
    os.makedirs(out, exist_ok=False)
    print(BANNER)
    dev = Dv.fetch_fake("FakeFez")
    now = P.now_cet()
    print(f"1. device record: {dev.name}, {len(dev.couplers)} couplers, {len(dev.failed_couplers)} failed, "
          f"{len(dev.failed_qubits)} failed qubits (no quality number read)")

    s1 = P.scan_job(dev, os.path.join(out, "scan1"), spec=SCAN)
    rings = rings_of(s1["runs"])
    planted = [(rings[40][12], rings[40][13]), (rings[72][50], rings[72][51])]
    print(f"2. planted (known to this script only): couplers {planted}, two-qubit depolarizing 0.15")

    base = None
    for k, s in enumerate([s1, P.scan_job(dev, os.path.join(out, "scan2"), spec=SCAN)]):
        sim.submit(s["job"], "pauli", seed=1000 + k, planted=planted, **PLANT)
        reg, obs = P.scan_ingest(dev.name, s["job"], s["runs"], base, f"scan{k + 1}")
        base = os.path.join(out, f"registry_{k + 1}.json")
        R.save(reg, base)
    reg = R.load(base)
    listed = [e["centre"] for e in R.entries(reg["observations"], now=now) if e["status"] == "LISTED"]
    print(f"3. two scans ({s1['circuits']} circuits each); LISTED paths: {listed}")

    print("4. practice problems (exact answers recorded at build time), standard ring against avoiding ring:")
    rows = []
    for label, job, key in PROBLEMS:
        std = P.plan(dev, job, R.empty(), os.path.join(out, f"std_{job['kind']}"), now, standard=True)
        sim.submit(std["user_job"]["job"], "pauli", seed=2000, planted=planted, **PLANT)
        (rs,) = J.score(std["user_job"]["job"], std["user_job"]["runs"]).values()
        av = P.plan(dev, job, reg, os.path.join(out, f"avoid_{job['kind']}"), now, sentinel_spec=SENTINEL)
        sim.submit(av["user_job"]["job"], "pauli", seed=2000, planted=planted, **PLANT)
        (ra,) = J.score(av["user_job"]["job"], av["user_job"]["runs"]).values()
        sv = Se.verdict(av["user_job"]["job"], av["user_job"]["runs"])
        rows.append(dict(problem=label, standard=rs[key], avoiding=ra[key], sentinel=sv["status"],
                         masked=av["masked_couplers"], qpu_seconds_model=av["user_job"]["qpu_seconds_model"]))
        print(f"   {label}: fraction right {rs[key]:.3f} (standard ring) -> {ra[key]:.3f} (avoiding ring); "
              f"sentinel on the avoiding ring: {sv['status']}; masked couplers {av['masked_couplers']}")
    dt = time.time() - t0
    json.dump(dict(banner=BANNER, planted=planted, listed=listed, rows=rows, seconds=round(dt, 1)),
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"5. done in {dt:.0f} s on this machine. {BANNER}.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quickstart_out")
