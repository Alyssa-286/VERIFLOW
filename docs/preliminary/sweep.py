import numpy as np
rng=np.random.default_rng(1)
M=5; TPR=0.9; FPR=0.3; pc0=0.4; N=400
def run(rule,q0,f,T,n=4000,beta=0.0,lr=1.0):
    h=T/n; p=np.r_[pc0,(1-pc0)*q0]; th=np.log(p); r=np.r_[TPR,f]; lref=th.copy()
    m=np.zeros_like(th); v=np.zeros_like(th); accs=[]
    for i in range(n):
        p=np.exp(th-th.max()); p/=p.sum(); rb=p@r
        if rule=="npg": g=r-rb
        elif rule=="vpg": g=p*(r-rb)
        elif rule=="adam":
            g0=p*(r-rb); m=0.9*m+0.1*g0; v=0.999*v+0.001*g0**2
            g=(m/(1-0.9**(i+1)))/(np.sqrt(v/(1-0.999**(i+1)))+1e-8)*0.05
        if beta>0: g=g-beta*(th-lref) if rule=="npg" else g
        th=th+h*lr*g; accs.append(p[0])
    return np.mean(accs[n//2:]), p[0]
for T,label in [(2,"short"),(8,"medium"),(30,"long")]:
  for rule in ["npg","vpg","adam"]:
    rows=[]
    for k in range(N if rule!="adam" else 120):
        q0=rng.dirichlet(np.ones(M))
        f=rng.uniform(0,1,M); f=f-(q0@f)+FPR
        if (f<0).any() or (f>1).any(): continue
        V=q0@(f-FPR)**2; mx=f.max(); K=np.log(q0@np.exp(f*T))
        acc,_=run(rule,q0,f,T,n=1500)
        rows.append((acc,V,mx,K,(q0**2)@((f-FPR)**2)))
    a=np.array(rows); y=a[:,0]
    from numpy import corrcoef as cc
    rk=lambda x:np.argsort(np.argsort(x))
    s=lambda j:cc(rk(y),rk(a[:,j]))[0,1]
    print(f"{label:6s} {rule:4s} n={len(a)} spearman(acc, V0)={s(1):+.2f} (acc,maxf)={s(2):+.2f} (acc,K_T)={s(3):+.2f} (acc,V_q2)={s(4):+.2f}")
