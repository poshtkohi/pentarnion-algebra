#!/usr/bin/env python3
"""Construct fixed-phase diagnostics from two archived dense trajectories.

For each t_k=2*pi*k/omega within the record, start from the immediately preceding
archived state and integrate for at most one output interval using DOP853. This
phase alignment uses the actual p coordinates and does not regenerate the long
chaotic record, establish convergence, or interpolate/smooth its geometry.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp
from verify_section9 import make_model

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    package=Path(__file__).resolve().parents[3]
    parser.add_argument('--package',type=Path,default=package)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args();out=args.output or args.package/'outputs/numerical/validation';out.mkdir(parents=True,exist_ok=True)
    g,x,f,_=make_model();fun=sp.lambdify((g,*x),f,'numpy',cse=True)
    records=[]
    for name,stem,c,a,w in [
        ('I','protocol_I_omega_0p8_dense',.68,.08,.8),
        ('II','protocol_II_omega_0p2_dense',.525,.475,.2)]:
        source=args.package/'data/riccati_dense'/f'{stem}_timeseries.csv'
        if not source.exists(): source=source.with_suffix('.zip')
        df=pd.read_csv(source)
        times=df.t.to_numpy();p=df[['w','x','y','z','u_active']].to_numpy()
        T=2*np.pi/w;ks=np.arange(int(np.ceil(times[0]/T)),int(np.floor(times[-1]/T))+1)
        rows=[]
        for k in ks:
            tk=float(k*T);j=max(0,int(np.searchsorted(times,tk,side='right')-1))
            if tk<=times[j]+1e-12:pp=p[j]
            else:
                sol=solve_ivp(lambda t,y:np.asarray(fun(c+a*np.sin(w*t),*y),float).ravel(),
                              (times[j],tk),p[j],method='DOP853',rtol=1e-11,atol=1e-13)
                if not sol.success:raise RuntimeError(sol.message)
                pp=sol.y[:,-1]
            rows.append([int(k),tk,*pp,float(tk-times[j])])
        data=pd.DataFrame(rows,columns=['forcing_cycle','t','p0','p1','p2','p3','p4','local_alignment_interval'])
        data.to_csv(out/f'protocol_{name}_archived_stroboscopic.csv',index=False)
        records.append({'protocol':name,'omega':w,'samples':len(data),'forcing_phase':0,
            'first_time':float(data.t.iloc[0]),'last_time':float(data.t.iloc[-1]),
            'source':f'data/riccati_dense/{stem}_timeseries.csv',
            'max_local_alignment_interval':float(data.local_alignment_interval.max()),
            'method':'DOP853 local integration from preceding archived p-state; rtol=1e-11 atol=1e-13',
            'limitation':'Derived finite-record diagnostic, not an independent long-time integration.'})
    (out/'archived_phase_sampling.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
