import numpy as np, itertools
X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]);I=np.eye(2)
def k(a,b): return np.kron(b,a)  # a on q0 (LSB), b on q1
def N(al,be,ga):
    from scipy.linalg import expm
    return expm(1j*(al*k(X,X)+be*k(Y,Y)+ga*k(Z,Z)))
def rz(t): return np.diag([np.exp(-.5j*t),np.exp(.5j*t)])
def ry(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-s],[s,c]])
P0=np.diag([1,0]);P1=np.diag([0,1])
CX10=k(I,P0)+k(X,P1)  # control q1, target q0
CX01=k(P0,I)+k(P1,X)  # control q0, target q1
def same(U,W):
    i=np.argmax(abs(U)); ph=U.flat[i]/W.flat[i]; return abs(abs(ph)-1)<1e-9 and np.max(abs(U-ph*W))<1e-9
rng=np.random.default_rng(1)
tests=[tuple(rng.uniform(-1,1,3)) for _ in range(3)]
H=np.pi/2
found=[]
for perm in itertools.permutations(range(3)):
  for c1,c2,c3 in itertools.product((-1,0,1),repeat=3):
    for d1,d2,d3 in itertools.product((-1,1),repeat=3):
      for s0,r0,s4,r4 in itertools.product((-1,0,1),repeat=4):
        ok=True
        for abc in tests:
            al,be,ga=[abc[i] for i in perm]
            t1=c1*H+d1*2*ga; t2=c2*H+d2*2*al; t3=c3*H+d3*2*be
            C=k(rz(s4*H),rz(r4*H))@CX10@k(I,ry(t3))@CX01@k(rz(t1),ry(t2))@CX10@k(rz(s0*H),rz(r0*H))
            if not same(N(*abc),C): ok=False;break
        if ok:
            found.append((perm,(c1,c2,c3),(d1,d2,d3),(s0,r0,s4,r4))); print(found[-1]); 
            if len(found)>3: raise SystemExit
print(len(found))
