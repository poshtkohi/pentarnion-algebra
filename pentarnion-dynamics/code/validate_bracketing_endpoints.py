"""Independent Python/Numba validation of eta=0 and eta=1 endpoints."""
from pathlib import Path
import argparse, json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from pentarnion_core import A,B,P0,pmul,rhs,tangent_rhs,integrate,full_lyapunov


def jacobian(q,eta):
    I=np.eye(5);return np.column_stack([np.asarray(tangent_rhs(q,I[:,j],eta)) for j in range(5)])

def refine_periodic_orbit(q_guess,T_guess,eta=1.0):
    I=np.eye(5)
    def f(t,q):return np.asarray(rhs(q,eta))
    def flowvar(q,T):
        def combined(t,z):
            y=z[:5];M=z[5:].reshape(5,5);return np.r_[f(t,y),(jacobian(y,eta)@M).ravel()]
        sol=solve_ivp(combined,(0,T),np.r_[q,I.ravel()],method='DOP853',rtol=5e-13,atol=1e-15,max_step=.005)
        return sol.y[:5,-1],sol.y[5:,-1].reshape(5,5)
    q=q_guess.copy();T=float(T_guess)
    for _ in range(8):
        y,M=flowvar(q,T);R=np.r_[y-q,f(0,q)[0]]
        K=np.zeros((6,6));K[:5,:5]=M-I;K[:5,5]=f(0,y);K[5,:5]=jacobian(q,eta)[0]
        dz=np.linalg.solve(K,-R);q+=dz[:5];T+=dz[5]
        if np.linalg.norm(R)<1e-13:break
    y,M=flowvar(q,T)
    return q,T,y-q,np.linalg.eigvals(M)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--out',default='python_validation_results');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dt=.01 if args.quick else .005;trans=300 if args.quick else 1000;measure=800 if args.quick else 5000
    summary={'A':A.tolist(),'B':B.tolist(),'p0':P0.tolist(),'initial_associator_norm':float(np.linalg.norm(pmul(pmul(B,P0),P0)-pmul(B,pmul(P0,P0))))}
    for eta in (0.0,1.0):
        spec,div,ht,hs,q,m=full_lyapunov(eta,dt,trans,measure,.25,100)
        t,x=integrate(eta,dt,trans,500 if eta==0 else 100,1)
        np.savez_compressed(out/f'endpoint_eta{int(eta)}.npz',t=t,x=x,spectrum=spec,mean_divergence=div,history_time=ht,history_spectrum=hs,maxnorm=m)
        summary[f'eta{int(eta)}']={'spectrum':spec.tolist(),'mean_divergence':float(div),'maxnorm':float(m)}
    # initial periodic estimate from endpoint record
    d=np.load(out/'endpoint_eta1.npz');t=d['t'];x=d['x'];
    peaks=np.where((x[1:-1,0]>x[:-2,0])&(x[1:-1,0]>=x[2:,0]))[0]+1
    q,T,closure,eig=refine_periodic_orbit(x[peaks[-2]],t[peaks[-1]]-t[peaks[-2]])
    np.savez(out/'periodic_orbit.npz',q=q,T=T,closure=closure,eig=eig)
    summary['periodic_orbit']={'phase_point':q.tolist(),'period':float(T),'closure_norm':float(np.linalg.norm(closure)),'floquet':[{'real':float(z.real),'imag':float(z.imag),'modulus':float(abs(z))} for z in eig]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
