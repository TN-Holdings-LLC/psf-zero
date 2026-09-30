"""indep_check_2026-09-30.py -- independent re-check (numpy only, no PennyLane/Qiskit): recompute the logical fidelity of every
parsed circuit in every rounds.jsonl and compare with the recorded value.

    python indep_check_2026-09-30.py OUT     (OUT = the unpacked invest_outputs_0930.zip)
"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # e2e_vllm_psf_v6.py next to this file
from e2e_vllm_psf_v6 import TASKS, ALIASES, _safe_eval  # task targets and parameter parsing only

I = np.eye(2); X = np.array([[0,1],[1,0]]); Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1,-1]); H = np.array([[1,1],[1,-1]])/math.sqrt(2)
S = np.diag([1,1j]); T = np.diag([1,np.exp(1j*math.pi/4)])
def rx(t): return np.array([[math.cos(t/2),-1j*math.sin(t/2)],[-1j*math.sin(t/2),math.cos(t/2)]])
def ry(t): return np.array([[math.cos(t/2),-math.sin(t/2)],[math.sin(t/2),math.cos(t/2)]])
def rz(t): return np.diag([np.exp(-1j*t/2),np.exp(1j*t/2)])
ONE = {"h":H,"x":X,"y":Y,"z":Z,"s":S,"sdg":S.conj().T,"t":T,"tdg":T.conj().T}
def apply1(psi,n,q,U):
    psi=psi.reshape((2,)*n); psi=np.moveaxis(np.tensordot(U,psi,axes=([1],[q])),0,q); return psi.reshape(-1)
def apply_c(psi,n,c,t,U):
    psi=psi.reshape((2,)*n).copy(); idx=[slice(None)]*n; idx[c]=1; sub=psi[tuple(idx)]
    tt=t if t<c else t-1
    sub=np.moveaxis(np.tensordot(U,sub,axes=([1],[tt])),0,tt); psi[tuple(idx)]=sub; return psi.reshape(-1)
def simulate(spec,n):
    psi=np.zeros(2**n,complex); psi[0]=1
    for g in spec["gates"]:
        nm=ALIASES.get(str(g["name"]).lower().strip(),str(g["name"]).lower().strip()); q=[int(x) for x in g["qubits"]]
        p=g.get("params") or []; p=p if isinstance(p,list) else [p]; p=[_safe_eval(v) for v in p]
        if nm in ONE: psi=apply1(psi,n,q[0],ONE[nm])
        elif nm in ("rx","ry","rz"): psi=apply1(psi,n,q[0],{"rx":rx,"ry":ry,"rz":rz}[nm](p[0]))
        elif nm=="cx": psi=apply_c(psi,n,q[0],q[1],X)
        elif nm=="cz": psi=apply_c(psi,n,q[0],q[1],Z)
        elif nm=="cry": psi=apply_c(psi,n,q[0],q[1],ry(p[0]))
        elif nm=="crz": psi=apply_c(psi,n,q[0],q[1],rz(p[0]))
        elif nm=="swap": psi=np.swapaxes(psi.reshape((2,)*n),q[0],q[1]).reshape(-1)
        else: raise ValueError(nm)
    return psi
def target(task):
    n,_,groups=TASKS[task]; order=[];v=np.array([1],complex)
    for w,g in groups: order+=w; v=np.kron(v,g)
    return np.transpose(v.reshape((2,)*n),[order.index(i) for i in range(n)]).reshape(-1)
root=sys.argv[1]; worst=0; cnt=0; solved_mismatch=0
for f in sorted(glob.glob(os.path.join(root,"*","run*","*","rounds.jsonl"))):
    task=f.split(os.sep)[-2]; n=TASKS[task][0]
    if n>20: continue  # fill27 checked by groups below
    tg=target(task)
    for r in map(json.loads,open(f)):
        if "spec" in r and "fidelity_logical" in r:
            fi=abs(np.vdot(tg,simulate(r["spec"],n)))**2; worst=max(worst,abs(fi-r["fidelity_logical"])); cnt+=1
            solved_mismatch += (fi>=0.9999) != (r["fidelity_compiled"]>=0.9999)
print("small tasks: circuits", cnt, "max |F_indep - F_recorded|", worst, "solved disagreements", solved_mismatch)
# fill27: simulate each group separately when no gate crosses groups
cnt=0;worst=0;cross=0
for f in sorted(glob.glob(os.path.join(root,"*","run*","fill27","rounds.jsonl"))):
    for r in map(json.loads,open(f)):
        if "spec" not in r or "fidelity_logical" not in r: continue
        n,_,groups=TASKS["fill27"]; gid={}
        for k,(w,_) in enumerate(groups):
            for q in w: gid[q]=k
        gates=r["spec"]["gates"]
        if any(len({gid[int(q)] for q in g["qubits"]})>1 for g in gates): cross+=1; continue
        F=1.0
        for k,(w,gv) in enumerate(groups):
            loc={q:i for i,q in enumerate(w)}
            sub={"gates":[dict(g,qubits=[loc[int(q)] for q in g["qubits"]]) for g in gates if gid[int(g["qubits"][0])]==k]}
            F*=abs(np.vdot(gv,simulate(sub,len(w))))**2
        worst=max(worst,abs(F-r["fidelity_logical"])); cnt+=1
print("fill27: circuits", cnt, "max diff", worst, "cross-group circuits skipped", cross)
