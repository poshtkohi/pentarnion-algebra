#!/usr/bin/env python3
"""Independent DOP853 check of a SHORT spin tangent-space spectrum calculation.

Use the CSV made with lyapunov_scan --transient-periods 5 --periods 20
--omegas .3,1.2. This validates the field/tangent implementation over that finite
horizon; it is not a convergence certificate for a long-time Lyapunov exponent.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'part-I/code/spin_dynamics'))
from mechanical_spin import Y0,rhs,jacobian

def integrate(w:float,trans:int,periods:int) -> np.ndarray:
    T=2*np.pi/w;y=Y0.copy();e=np.cross(y,[1.,0.,0.]);e/=np.linalg.norm(e);M=np.column_stack((e,np.cross(y,e)));Y=np.r_[y,M.ravel()];sums=np.zeros(2);t=0.
    def f(t,Y):return np.r_[rhs(t,Y[:3],w),(jacobian(t,Y[:3],w)@Y[3:].reshape(3,2)).ravel()]
    # A different QR schedule from the fixed-step implementation.
    nper=int(np.ceil(T/.25));h=T/nper
    for n in range((trans+periods)*nper):
        end=(n+1)*h;sol=solve_ivp(f,(t,end),Y,method='DOP853',rtol=1e-11,atol=1e-13)
        if not sol.success:raise RuntimeError(sol.message)
        Y=sol.y[:,-1];y=Y[:3];scale=np.linalg.norm(y);y/=scale;M=Y[3:].reshape(3,2);M=(M-y[:,None]*(y@M)[None,:])/scale
        q,r=np.linalg.qr(M,mode='reduced');sign=np.sign(np.diag(r));sign[sign==0]=1.;q=q*sign[None,:]
        if n>=trans*nper:sums+=np.log(np.abs(np.diag(r)))
        Y=np.r_[y,q.ravel()];t=end
    return np.sort(sums/(periods*T))[::-1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=pd.read_csv(a.csv);rows=[]
    for _,r in data.iterrows():
        if r.transient_periods>10 or r.accumulation_periods>30:raise ValueError('Use a short comparison record as specified in the help')
        adaptive=integrate(float(r.Omega),int(r.transient_periods),int(r.accumulation_periods));fixed=np.array([r.lambda_max,r.lambda_min]);err=float(np.max(abs(adaptive-fixed)))
        if err>2e-6:raise AssertionError((r.Omega,adaptive,fixed,err))
        rows.append({'Omega':r.Omega,'adaptive':adaptive.tolist(),'fixed':fixed.tolist(),'max_absolute_difference':err});print('PASS Omega=',r.Omega,'difference=',err)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'PASS','comparison':'independent DOP853 versus projected RK4; short finite horizon','results':rows},indent=2)+'\n')
if __name__=='__main__':main()
