#!/usr/bin/env python3
"""Independent numerical checks for Section 9 of main-R1(4).tex.

Usage:
  python verify_section9.py --package /path/to/code_data --output ./checks
  python verify_section9.py --package /path/to/code_data --output ./checks --extended

Requires numpy, scipy, sympy, pandas. The extended checks integrate Protocol II
at omega=2 through t=1200. These are numerical checks, not interval certificates.
The original continuation archive is read only; all results go to --output.
The default output is part-II/outputs/numerical/validation.
The extended run also writes forcing-phase samples and a late-response record.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def make_model():
    g = sp.symbols('gamma', real=True)
    x = sp.Matrix(sp.symbols('w x y z u', real=True))
    k, sig, r = 1/sp.sqrt(1+g*g), g/sp.sqrt(1+g*g), 1/sp.sqrt(2)
    def product(a, b):
        q = sp.Matrix([a[0]*b[0]-sum(a[i]*b[i] for i in range(1,5))]
                      + [a[0]*b[i]+a[i]*b[0] for i in range(1,5)])
        for i,j,l in [(1,2,3),(2,3,1),(3,1,2)]:
            c=a[i]*b[j]-a[j]*b[i]; q[l]+=k*c; q[4]+=sig*c
        for i,j,l in [(1,2,3),(2,3,1),(3,1,2)]:
            c=a[4]*b[i]-a[i]*b[4]; q[j]+=r*c; q[l]+=r*c
        return q
    A=sp.Matrix([-sp.Rational(1,2),-2,sp.Rational(1,2),-1,-g/2])
    B=sp.Matrix([1,0,2,0,-g/2])
    f=(product(A,x)+product(B,product(x,x))).applyfunc(sp.expand)
    return g,x,f,f.jacobian(x)


def write_json(path, obj):
    def convert(x):
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,indent=2,default=convert),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[3],help='Part II root containing data/')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args(); out=(args.output or args.package/'outputs/numerical/validation').resolve();out.mkdir(parents=True,exist_ok=True)
    ref=(args.package.resolve()/'data').resolve()
    if out==ref or ref in out.parents: parser.error('Output must be outside reference data')
    bp=args.package.resolve()/'data/riccati/periodic_branches'
    if not bp.is_dir(): parser.error(f'Missing continuation data: {bp}')
    g,x,f,J=make_model(); L=J.subs(dict.fromkeys(x,0));lam=sp.symbols('lambda')
    b=3*(16+g*g-g**4)/(8*(1+g*g))
    c=(44-g*g-3*g**4+sp.sqrt(2)*g*(g**4+g*g+194))/(16*(1+g*g))
    H=100+10*g*g-6*g**4-sp.sqrt(2)*(g**5+g**3+194*g)
    expected=(lam**2+lam+(22+g*g)/4)*(lam**3+sp.Rational(3,2)*lam**2+b*lam+c)
    residual=sp.simplify(L.charpoly(lam).as_expr()-expected)
    if residual != 0: raise AssertionError(f'Characteristic polynomial mismatch: {residual}')
    gh=brentq(sp.lambdify(g,H,'numpy'),0,1,xtol=1e-15)
    D=sp.Rational(3,2)*b-c
    crossing=-sp.diff(D,g)/(2*(b+sp.Rational(9,4)))
    fn=sp.lambdify((g,*x),f,'numpy',cse=True)
    jn=sp.lambdify((g,*x),J,'numpy',cse=True)
    def field(y,ga): return np.asarray(fn(ga,*y),dtype=float).ravel()

    # Kuznetsov convention: f=Lx+B2(x,x)/2; q*q=1 and p*q=1
    # (the products marked * in this comment are Hermitian inner products).
    AA=np.array(L.subs(g,gh),float)
    B2=np.array([np.array(sp.hessian(fi,list(x)).subs(g,gh),float) for fi in f])
    eig,V=np.linalg.eig(AA); idx=int(np.argmax(eig.imag));omega=eig[idx].imag
    q=V[:,idx]/np.linalg.norm(V[:,idx])
    ea,VA=np.linalg.eig(AA.T);p=VA[:,np.argmin(abs(ea+1j*omega))]
    p=p/np.conj(np.vdot(p,q))
    def bb(a,b): return np.einsum('ijk,j,k->i',B2,a,b)
    h20=np.linalg.solve(2j*omega*np.eye(5)-AA,bb(q,q))
    h11=-np.linalg.solve(AA,bb(q,q.conj()))
    G21=np.vdot(p,bb(q.conj(),h20)+2*bb(q,h11))
    exact={'charpoly_residual':str(residual),'gammaH':gh,'omegaH':omega,
           'crossing_speed':float(crossing.subs(g,gh)),
           'first_Lyapunov_coefficient':float(G21.real/(2*omega)),
           'l1_convention':'f=Lx+B2(x,x)/2; ||q||_2=1; conjugate(p).T q=1; l1=Re(G21)/(2 omega)',
           'l1_reference':'Y. A. Kuznetsov, Andronov-Hopf bifurcation, Scholarpedia 1(10):1858 (2006)',
           'eigenvector_residual':float(np.linalg.norm(AA@q-1j*omega*q)),
           'adjoint_residual':float(np.linalg.norm(AA.T@p+1j*omega*p)),
           'divergence':str(sp.simplify(sp.trace(J)))}
    write_json(out/'symbolic_and_hopf_checks.json',exact);print(exact,flush=True)
    records=[]
    for name,target in [('P1',.4),('P2',.4875),('Broad_P2',.4875),('Broad_P2',.7),
                        ('Broad_P2',.728),('Window_C1',.728),('P32',.49318)]:
        df=pd.read_csv(bp/f'{name}_floquet.csv')
        row=df.iloc[np.argmin(abs(df.gamma-target))];ga=float(row.gamma);T=float(row['T'])
        p0=np.array([row[f'x{i}'] for i in range(5)])
        def aug(t,Y):
            p=Y[:5];M=Y[5:].reshape(5,5)
            return np.r_[field(p,ga),(np.asarray(jn(ga,*p))@M).ravel()]
        sol=solve_ivp(aug,(0,T),np.r_[p0,np.eye(5).ravel()],method='DOP853',
                      rtol=1e-11,atol=1e-13,max_step=.03,dense_output=True)
        if not sol.success: raise RuntimeError(f'{name}: {sol.message}')
        ev,V=np.linalg.eig(sol.y[5:,-1].reshape(5,5));flow=field(p0,ga)
        alignment=np.abs(V.conj().T@flow)/(np.linalg.norm(V,axis=0)*np.linalg.norm(flow))
        neutral=int(np.argmax(alignment));rho=float(max(abs(np.delete(ev,neutral))))
        result={'branch':name,'gamma':ga,'period':T,
                'adaptive_closure_residual':float(np.linalg.norm(sol.y[:5,-1]-p0)),
                'transverse_radius':rho,'archive_transverse_radius':float(row.max_transverse_modulus),
                'neutral_error':float(abs(ev[neutral]-1))}
        records.append(result);print(result,flush=True)
        if (name,target) in [('P2',.4875),('Broad_P2',.4875),('Broad_P2',.7),('Broad_P2',.728),('Window_C1',.728)]:
            ts=np.linspace(0,T,2401)
            dd=pd.DataFrame(sol.sol(ts)[:5].T,columns=['p0','p1','p2','p3','p4'])
            dd.insert(0,'t',ts);dd.to_csv(out/f'orbit_{name}_{ga:.4f}.csv',index=False)
        if name=='Broad_P2' and target==.7:
            def sec(t,p): return p[1]
            sec.direction=1
            def peak(t,p): return field(p,ga)[3]
            peak.direction=-1
            orbit=solve_ivp(lambda t,p:field(p,ga),(0,T),p0,method='DOP853',
                            rtol=1e-11,atol=1e-13,max_step=.01,events=[sec,peak])
            section={'gamma':ga,'period':T,
                     'upward_p1_zero_crossings_per_cycle':len(orbit.t_events[0]),
                     'p3_local_maxima_per_cycle':len(orbit.t_events[1]),
                     'upward_crossing_times':orbit.t_events[0].tolist(),
                     'peak_times':orbit.t_events[1].tolist()}
            write_json(out/'broad_branch_section_check.json',section)

    pd.DataFrame(records).to_csv(out/'independent_orbit_checks.csv',index=False)
    if not args.extended:return
    extended=[];endpoint=None
    for mode,rtol in [('p',1e-10),('p',1e-12),('q',1e-11)]:
        def rhs(t,y):
            ga=.525+.475*np.sin(2*t);gd=.95*np.cos(2*t)
            if mode=='p':return field(y,ga)
            p=y.copy();p[4]*=ga;z=field(p,ga);z[4]=(z[4]-gd*y[4])/ga;return z
        def event(t,y):
            p=y.copy()
            if mode=='q':p[4]*=.525+.475*np.sin(2*t)
            return 1e6-np.linalg.norm(p)
        event.terminal=True;event.direction=-1
        p0=np.array([.5,-.5,-.5,.5,-.5*(.525 if mode=='p' else 1)])
        sol=solve_ivp(rhs,(0,1200),p0,method='DOP853',rtol=rtol,atol=rtol/100,
                      max_step=.05,events=event,dense_output=True)
        actual=sol.y.copy()
        if mode=='q': actual[4]*=.525+.475*np.sin(2*sol.t)
        result={'coordinates':mode,'rtol':rtol,'atol':rtol/100,'max_step':.05,
                'end_time':float(sol.t[-1]),'success':bool(sol.success),
                'radius_limit_reached':bool(sol.t_events[0].size),
                'maximum_radius_at_adaptive_nodes':float(np.linalg.norm(actual,axis=0).max()),
                'final_radius':float(np.linalg.norm(actual[:,-1])),
                'final_actual_state':actual[:,-1].tolist()}
        extended.append(result);print(result,flush=True)
        if mode=='p' and rtol==1e-12 and sol.t[-1]==1200:
            endpoint=actual[:,-1]
            ts=np.arange(int(1200/np.pi)+1)*np.pi
            pp=sol.sol(ts).T
            diag=pd.DataFrame(pp,columns=['p0','p1','p2','p3','p4'])
            diag.insert(0,'t',ts);diag.insert(0,'forcing_cycle',np.arange(len(ts)))
            diag['successive_return_distance']=np.r_[np.nan,np.linalg.norm(np.diff(pp,axis=0),axis=1)]
            diag.to_csv(out/'protocol_II_omega2_stroboscopic.csv',index=False)

    write_json(out/'extended_protocol_II_omega2.json',extended)
    if endpoint is not None:
        def aug(t,Y):
            ga=.525+.475*np.sin(2*t);p=Y[:5];M=Y[5:].reshape(5,5)
            return np.r_[field(p,ga),(np.asarray(jn(ga,*p))@M).ravel()]
        T=np.pi
        sol=solve_ivp(aug,(1200,1200+T),np.r_[endpoint,np.eye(5).ravel()],
                      method='DOP853',rtol=1e-12,atol=1e-14,max_step=.02,dense_output=True)
        ev=np.linalg.eigvals(sol.y[5:,-1].reshape(5,5));rho=float(max(abs(ev)))
        result={'period':T,'one_period_return_residual':float(np.linalg.norm(sol.y[:5,-1]-endpoint)),
                'stroboscopic_spectral_radius':rho,'logrho_over_period':float(np.log(rho)/T),
                'multipliers':[[float(z.real),float(z.imag)] for z in ev]}
        write_json(out/'protocol_II_omega2_periodic_response.json',result);print(result,flush=True)
        times=np.linspace(1200,1200+4*T,2401)
        late=solve_ivp(lambda t,p:field(p,.525+.475*np.sin(2*t)),
                       (times[0],times[-1]),endpoint,t_eval=times,method='DOP853',
                       rtol=1e-12,atol=1e-14,max_step=.02)
        if not late.success: raise RuntimeError(late.message)
        dd=pd.DataFrame(late.y.T,columns=['p0','p1','p2','p3','p4'])
        dd.insert(0,'gamma',.525+.475*np.sin(2*times));dd.insert(0,'t',times)
        dd.to_csv(out/'protocol_II_omega2_late_response.csv',index=False)


if __name__=='__main__':main()
