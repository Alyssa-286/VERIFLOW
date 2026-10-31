import numpy as np
rng=np.random.default_rng(0)
M=5; TPR=0.9; FPR=0.3
q0=rng.dirichlet(np.ones(M)); pc0=0.4
fA=np.full(M,FPR)
fB=rng.uniform(0,0.8,M); fB=fB-(q0@fB)+FPR; assert (fB>=0).all() and (fB<=1).all()
print("FPR A,B",q0@fA,q0@fB,"V_B",q0@(fB-q0@fB)**2,"max_B",fB.max())
def rk4(F,x,T,n=20000,**k):
    h=T/n; out=[x.copy()]; 
    for i in range(n):
        a=F(x,**k);b=F(x+h/2*a,**k);c=F(x+h/2*b,**k);d=F(x+h*c,**k); x=x+h/6*(a+2*b+2*c+d)
        if (i+1)%(n//8)==0: out.append(x.copy())
    return np.array(out)
def rep(p,f): r=np.r_[TPR,f]; return p*(r-p@r)
def vpg(th,f): p=np.exp(th-th.max()); p/=p.sum(); r=np.r_[TPR,f]; return p*(r-p@r)
p0=np.r_[pc0,(1-pc0)*q0]; T=np.linspace(0,8,9)
for name,f in [("A",fA),("B",fB)]:
    s=rk4(rep,p0,8.0,f=f); pc=s[:,0]
    K=np.log(np.exp(np.outer(T,f))@q0)
    pred=1/(1+np.exp(-(np.log(pc0/(1-pc0))+TPR*T-K)))
    print(name,"max|replicator - closed form| =",np.abs(pc-pred).max())
    th=rk4(vpg,np.log(p0),8.0,f=f); p=np.exp(th); p/=p.sum(1,keepdims=True)
    print(name,"replicator acc",np.round(pc,3)); print(name,"vanillaPG  acc",np.round(p[:,0],3))
f=fB; eps=1e-5; s=rk4(rep,p0,eps,n=8,f=f); q=s[-1,1:]/s[-1,1:].sum()
print("dFPR/dt numeric",(q@f-q0@f)/eps,"  V0",q0@(f-q0@f)**2)
