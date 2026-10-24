"""Shooting and pseudo-arclength continuation of the rational model's periodic branch.

Start from a refined periodic orbit at eta=1 (periodic_orbit.npz produced by
validate_bracketing_endpoints.py). The script follows the stable branch downward in eta and
then uses pseudo-arclength continuation around the saddle-node fold.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from pentarnion_core import rhs,tangent_rhs
I=np.eye(5)

def jac(q,e):return np.column_stack([np.asarray(tangent_rhs(q,I[:,j],e)) for j in range(5)])
def f(q,e):return np.asarray(rhs(q,e))
def flow(q,T,e,var=False):
    if not var:
        s=solve_ivp(lambda t,y:f(y,e),(0,T),q,method='DOP853',rtol=3e-10,atol=3e-12,max_step=.04);return s.y[:,-1]
    def c(t,z):
        y=z[:5];M=z[5:].reshape(5,5);return np.r_[f(y,e),(jac(y,e)@M).ravel()]
    s=solve_ivp(c,(0,T),np.r_[q,I.ravel()],method='DOP853',rtol=5e-11,atol=5e-13,max_step=.02)
    return s.y[:5,-1],s.y[5:,-1].reshape(5,5)
def shoot(zg,e):
    def R(z):
        q=z[:5];T=z[5];return np.r_[flow(q,T,e)-q,f(q,e)[0]]
    return root(R,zg,method='hybr',options={'xtol':2e-9,'maxfev':200})
def metrics(z,e):
    q=z[:5];T=z[5];_,M=flow(q,T,e,True);ev=np.linalg.eigvals(M);neutral=np.argmin(np.abs(ev-1));mt=np.max(np.abs(np.delete(ev,neutral)))
    return [e,T,*q,mt,*np.sort(np.abs(ev))[::-1]]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--periodic',default='python_validation_results/periodic_orbit.npz');ap.add_argument('--out',default='periodic_continuation.csv');args=ap.parse_args()
    d=np.load(args.periodic);z=np.r_[d['q'],float(d['T'])];rows=[];solutions=[]
    for e in np.arange(1,.507,-.0025):
        r=shoot(z,e)
        if not r.success or np.linalg.norm(r.fun)>1e-6:break
        z=r.x;rows.append(metrics(z,e));solutions.append(np.r_[z,e])
    # If the last two regular-continuation points straddle the fold closely,
    # follow with secant pseudo-arclength.
    if len(solutions)>=2:
        z0,z1=solutions[-2],solutions[-1]
        for _ in range(60):
            tangent=z1-z0;tangent/=np.linalg.norm(tangent);ds=np.clip(np.linalg.norm(z1-z0),.002,.02);pred=z1+ds*tangent
            def H(z):
                q=z[:5];T=z[5];e=z[6]
                return np.r_[flow(q,T,e)-q,f(q,e)[0],np.dot(z-pred,tangent)]
            r=root(H,pred,method='hybr',options={'xtol':2e-9,'maxfev':300})
            if not r.success or np.linalg.norm(H(r.x))>1e-6:break
            z2=r.x;rows.append(metrics(z2[:6],z2[6]));z0,z1=z1,z2
    header='eta,period,q0,q1,q2,q3,q4,max_transverse_multiplier,mod1,mod2,mod3,mod4,mod5'
    np.savetxt(args.out,np.asarray(rows),delimiter=',',header=header,comments='')
if __name__=='__main__':main()
