import importlib.util,numpy as np,pennylane as qml,itertools,sys
s=importlib.util.spec_from_file_location("k","13_qpinn_kirsch.py");K=importlib.util.module_from_spec(s);s.loader.exec_module(K)
A,R=K.A_HOLE,K.R_OUT
rng=np.random.default_rng(0);n=1500
r=np.sqrt(A**2+rng.random(n)*(R**2-A**2));th=rng.random(n)*2*np.pi
u,v=K.uv_exact(r*np.cos(th),r*np.sin(th))
ur=u*np.cos(th)+v*np.sin(th);ut=-u*np.sin(th)+v*np.cos(th)
xi=0.95*(2*(A/r-A/R)/(1-A/R)-1);a1=np.arccos(xi)
def qfeat(nq,L,seed):
    dev=qml.device("default.qubit",wires=nq);g=np.random.default_rng(seed);W=g.random((2,L,nq,3))*2*np.pi
    obs=[qml.PauliZ(i) for i in range(nq)]+[qml.PauliX(i) for i in range(nq)]+[qml.PauliZ(i)@qml.PauliZ(j) for i,j in itertools.combinations(range(nq),2)]
    @qml.qnode(dev)
    def c(x1,x2):
        for e in range(2):
            for w in range(nq): qml.RY(x1,wires=w); qml.RZ(x2,wires=w)
            qml.StronglyEntanglingLayers(W[e],wires=range(nq))
        return [qml.expval(o) for o in obs]
    return np.stack(c(a1,th),1)
def cfeat(m,seed):
    g=np.random.default_rng(seed);X=np.stack([a1,np.cos(th),np.sin(th)],1)
    Wm=g.normal(size=(3,m))*1.5;b=g.random(m)*2*np.pi;return np.tanh(X@Wm+b)
def err(F):
    F=np.concatenate([F,np.ones((n,1))],1);tr=slice(0,1000);te=slice(1000,n);e=[]
    for y in (ur/r,ut/r):
        c,*_=np.linalg.lstsq(F[tr],y[tr],rcond=None);e.append(F[te]@c*r[te])
    pu=e[0]*np.cos(th[te])-e[1]*np.sin(th[te]);pv=e[0]*np.sin(th[te])+e[1]*np.cos(th[te])
    return np.sqrt(((pu-u[te])**2+(pv-v[te])**2).sum()/(u[te]**2+v[te]**2).sum())
for nq in (4,6):
    for sd in (0,1):
        F=qfeat(nq,2,sd);m=F.shape[1]
        print(f"QELM nq={nq} feats={m} seed={sd} err={err(F):.2e} | classical RF same m: {err(cfeat(m,sd)):.2e}",flush=True)
