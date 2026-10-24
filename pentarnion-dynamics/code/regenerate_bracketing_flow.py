#!/usr/bin/env python3
"""Regenerate all initial-condition-dependent data for the sparse rational
Pentarnion bracketing model using the low-rational initial state

    p(0) = (1 - i - j + k - ell)/2.

The script retains every high-resolution trajectory sample used in the
publication figures. Model-dependent objects (equilibria, refined periodic
orbit, Floquet multipliers and continuation branch) are independent of the
chosen basin initial condition and are retained from the supplied validation
package.
"""
from __future__ import annotations
from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bracketing_flow"
HI = DATA / "high_resolution"
HI.mkdir(parents=True, exist_ok=True)

SQRT2 = math.sqrt(2.0)
A = np.array([-1/2, -2, 1/2, -1, -1/2], dtype=np.float64)
B = np.array([1, 0, 2, 0, -1/2], dtype=np.float64)
P0 = np.array([1/2, -1/2, -1/2, 1/2, -1/2], dtype=np.float64)

DT = 0.005
TRANSIENT = 1000.0
MEASURE = 5000.0
RENORM = 0.25
HISTORY_INTERVAL = 50.0
BLOWUP = 100.0

@njit(cache=True)
def pmul(p, q):
    p0,p1,p2,p3,p4 = p
    q0,q1,q2,q3,q4 = q
    r = np.empty(5, dtype=np.float64)
    r[0] = p0*q0-p1*q1-p2*q2-p3*q3-p4*q4
    r[1] = p0*q1+q0*p1+(p2*q3-p2*q4-p3*q2-p3*q4+p4*q2+p4*q3)/SQRT2
    r[2] = p0*q2+q0*p2+(-p1*q3-p1*q4+p3*q1-p3*q4+p4*q1+p4*q3)/SQRT2
    r[3] = p0*q3+q0*p3+(p1*q2-p1*q4-p2*q1-p2*q4+p4*q1+p4*q2)/SQRT2
    r[4] = p0*q4+q0*p4+(p1*q2-p1*q3-p2*q1+p2*q3+p3*q1-p3*q2)/SQRT2
    return r

@njit(cache=True)
def rhs(q, eta):
    qq = pmul(q,q)
    return pmul(A,q) + (1.0-eta)*pmul(B,qq) + eta*pmul(pmul(B,q),q)

@njit(cache=True)
def tangent_rhs(q,h,eta):
    left = pmul(B, pmul(q,h)+pmul(h,q))
    right = pmul(pmul(B,h),q)+pmul(pmul(B,q),h)
    return pmul(A,h)+(1.0-eta)*left+eta*right

@njit(cache=True)
def divergence(q,eta):
    return -2.5 + 10.0*q[0] - (8.0+4.0*eta)*q[2] + (2.0+3.0*eta)*q[4]

@njit(cache=True)
def rk4_state(q,eta,dt):
    k1=rhs(q,eta); k2=rhs(q+0.5*dt*k1,eta)
    k3=rhs(q+0.5*dt*k2,eta); k4=rhs(q+dt*k3,eta)
    return q+dt*(k1+2*k2+2*k3+k4)/6.0

@njit(cache=True)
def rk4_state_tangent(q,h,eta,dt):
    k1q=rhs(q,eta); k1h=tangent_rhs(q,h,eta)
    q2=q+0.5*dt*k1q; h2=h+0.5*dt*k1h
    k2q=rhs(q2,eta); k2h=tangent_rhs(q2,h2,eta)
    q3=q+0.5*dt*k2q; h3=h+0.5*dt*k2h
    k3q=rhs(q3,eta); k3h=tangent_rhs(q3,h3,eta)
    q4=q+dt*k3q; h4=h+dt*k3h
    k4q=rhs(q4,eta); k4h=tangent_rhs(q4,h4,eta)
    return q+dt*(k1q+2*k2q+2*k3q+k4q)/6.0, h+dt*(k1h+2*k2h+2*k3h+k4h)/6.0

@njit(cache=True)
def _deriv_matrix(q,H,eta):
    dq=rhs(q,eta); dH=np.empty((5,5),dtype=np.float64)
    for j in range(5): dH[:,j]=tangent_rhs(q,H[:,j],eta)
    return dq,dH

@njit(cache=True)
def rk4_matrix(q,H,eta,dt):
    k1q,k1H=_deriv_matrix(q,H,eta)
    k2q,k2H=_deriv_matrix(q+0.5*dt*k1q,H+0.5*dt*k1H,eta)
    k3q,k3H=_deriv_matrix(q+0.5*dt*k2q,H+0.5*dt*k2H,eta)
    k4q,k4H=_deriv_matrix(q+dt*k3q,H+dt*k3H,eta)
    return q+dt*(k1q+2*k2q+2*k3q+k4q)/6.0, H+dt*(k1H+2*k2H+2*k3H+k4H)/6.0

@njit(cache=True)
def modified_gram_schmidt(H):
    Q=np.empty_like(H); logs=np.empty(5)
    for j in range(5):
        v=H[:,j].copy()
        for i in range(j):
            c=np.dot(Q[:,i],v); v-=c*Q[:,i]
        n=np.sqrt(np.dot(v,v)); logs[j]=np.log(n); Q[:,j]=v/n
    return Q,logs

@njit(cache=True)
def integrate(p0,eta,dt,transient,record,stride=1,blowup=100.0):
    q=p0.copy()
    for _ in range(int(round(transient/dt))):
        q=rk4_state(q,eta,dt)
        if np.sqrt(np.dot(q,q))>blowup: return np.empty(0),np.empty((0,5)),True
    n=int(round(record/dt)); m=n//stride+1
    t=np.empty(m); x=np.empty((m,5));t[0]=0.0;x[0]=q;k=1
    for j in range(1,n+1):
        q=rk4_state(q,eta,dt)
        if np.sqrt(np.dot(q,q))>blowup: return t[:k],x[:k],True
        if j%stride==0: t[k]=j*dt;x[k]=q;k+=1
    return t,x,False

@njit(cache=True)
def full_lyapunov(p0,eta,dt,transient,measure,renorm=0.25,history_interval=50.0,blowup=100.0):
    q=p0.copy()
    for _ in range(int(round(transient/dt))):
        q=rk4_state(q,eta,dt)
        if np.sqrt(np.dot(q,q))>blowup:
            return np.full(5,np.nan),np.nan,np.empty(0),np.empty((0,5)),q,np.inf
    H=np.eye(5); nr=max(1,int(round(renorm/dt))); nb=int(round(measure/(nr*dt)))
    sums=np.zeros(5); divsum=0.0; ndiv=0; elapsed=0.0
    he=max(1,int(round(history_interval/(nr*dt)))); nh=nb//he+1
    ht=np.empty(nh); hs=np.empty((nh,5));hi=0
    ht[0]=0.0;hs[0]=0.0;hi=1
    maxnorm=np.sqrt(np.dot(q,q))
    for b in range(nb):
        for _ in range(nr):
            q,H=rk4_matrix(q,H,eta,dt); divsum+=divergence(q,eta); ndiv+=1
            n=np.sqrt(np.dot(q,q)); maxnorm=max(maxnorm,n)
            if n>blowup:
                return np.full(5,np.nan),np.nan,ht[:hi],hs[:hi],q,maxnorm
        H,logs=modified_gram_schmidt(H);sums+=logs;elapsed+=nr*dt
        if (b+1)%he==0:
            ht[hi]=elapsed;hs[hi]=sums/elapsed;hi+=1
    return sums/elapsed,divsum/ndiv,ht[:hi],hs[:hi],q,maxnorm

@njit(cache=True)
def largest_exponent(p0,eta,dt,transient,measure,renorm=0.25,blowup=100.0):
    q=p0.copy()
    for _ in range(int(round(transient/dt))):
        q=rk4_state(q,eta,dt)
        if np.sqrt(np.dot(q,q))>blowup:return np.nan,np.inf
    h=np.array([1.,.3,-.2,.4,.1]);h/=np.sqrt(np.dot(h,h))
    nr=max(1,int(round(renorm/dt))); nb=int(round(measure/(nr*dt)))
    total=0.;elapsed=0.;maxnorm=np.sqrt(np.dot(q,q))
    for _ in range(nb):
        for _ in range(nr):
            q,h=rk4_state_tangent(q,h,eta,dt)
            n=np.sqrt(np.dot(q,q));maxnorm=max(maxnorm,n)
            if n>blowup:return np.nan,maxnorm
        hn=np.sqrt(np.dot(h,h));total+=np.log(hn);h/=hn;elapsed+=nr*dt
    return total/elapsed,maxnorm

@njit(cache=True)
def boundedness_windows(p0,eta,dt,total,window,blowup=100.0):
    q=p0.copy();nwin=int(round(total/window));nstep=int(round(window/dt))
    maxv=np.empty(nwin);meanv=np.empty(nwin)
    for w in range(nwin):
        mx=0.;sm=0.
        for _ in range(nstep):
            q=rk4_state(q,eta,dt);n=np.sqrt(np.dot(q,q));mx=max(mx,n);sm+=n
            if n>blowup:
                maxv[w:]=np.nan;meanv[w:]=np.nan;return maxv,meanv
        maxv[w]=mx;meanv[w]=sm/nstep
    return maxv,meanv


def poincare_section(t,x,index=1,level=0.0,direction=1):
    y=x[:,index]-level
    if direction>0: inds=np.where((y[:-1]<0)&(y[1:]>=0))[0]
    elif direction<0: inds=np.where((y[:-1]>0)&(y[1:]<=0))[0]
    else: inds=np.where(y[:-1]*y[1:]<=0)[0]
    out=[]
    for j in inds:
        den=y[j+1]-y[j]
        a=0.0 if den==0 else -y[j]/den
        out.append(np.r_[t[j]+a*(t[j+1]-t[j]),x[j]+a*(x[j+1]-x[j])])
    return np.asarray(out)


def fft_metrics(t,y):
    yy=y-np.mean(y);n=len(yy);dt=float(np.mean(np.diff(t)))
    spec=np.abs(np.fft.rfft(yy))**2
    if len(spec):spec[0]=0
    total=spec.sum();p=spec/total if total>0 else spec
    freq=np.fft.rfftfreq(n,dt)
    nz=p[p>0]
    entropy=float(-np.sum(nz*np.log(nz))/np.log(len(p))) if len(nz)>1 else 0.0
    sortedp=np.sort(p)[::-1]
    bins90=int(np.searchsorted(np.cumsum(sortedp),.9)+1)
    top10=float(sortedp[:10].sum())
    return freq,p,entropy,bins90,top10


def radial_terms(X,eta):
    vals=np.empty((len(X),4))
    for n,p in enumerate(X):
        lin=2*np.dot(p,pmul(A,p))
        left=2*np.dot(p,pmul(B,pmul(p,p)))
        assoc=2*np.dot(p,pmul(pmul(B,p),p)-pmul(B,pmul(p,p)))
        vals[n]=[lin,left,assoc,lin+left+eta*assoc]
    return vals


def visit_equilibria(X,eq):
    d=np.linalg.norm(X[:,None,:]-eq[None,:,:],axis=2)
    nearest=np.argmin(d,axis=1)
    rows=[]
    for j in range(len(eq)):
        rows.append(dict(equilibrium=f'E{j}',nearest_fraction=float(np.mean(nearest==j)),
                         minimum_distance=float(d[:,j].min()),median_distance=float(np.median(d[:,j]))))
    return pd.DataFrame(rows)


def save_npz(name,t,x):
    np.savez_compressed(HI/name,t=t,x=x)


def main():
    start=time.time()
    # Warm compilation.
    rk4_state(P0,0.0,DT)
    assoc=pmul(pmul(B,P0),P0)-pmul(B,pmul(P0,P0))
    print('P0',P0,'associator',assoc,'norm',np.linalg.norm(assoc))

    pd.DataFrame({
        'component':['scalar','i','j','k','l'],
        'A':A,'B':B,'p0':P0
    }).to_csv(DATA/'parameters.csv',index=False)
    pd.DataFrame([{
        'p0_expression':'(1-i-j+k-l)/2',
        'initial_norm':np.linalg.norm(P0),
        'associator_scalar':assoc[0],
        'associator_i':assoc[1],
        'associator_j':assoc[2],
        'associator_k':assoc[3],
        'associator_l':assoc[4],
        'associator_norm':np.linalg.norm(assoc)
    }]).to_csv(DATA/'initial_condition_validation.csv',index=False)

    # Full Lyapunov spectra.
    spectra=[]
    for eta in (0.0,1.0):
        print('Full spectrum eta',eta,flush=True)
        spec,div,ht,hs,q,m=full_lyapunov(P0,eta,DT,TRANSIENT,MEASURE,RENORM,HISTORY_INTERVAL,BLOWUP)
        np.savez_compressed(HI/f'lyapunov_history_eta{int(eta)}.npz',t=ht,spectrum=hs)
        spectra.append(dict(eta=int(eta),**{f'lambda_{i+1}':spec[i] for i in range(5)},
                            sum_exponents=spec.sum(),mean_divergence=div,maxnorm=m))
        print(spec,div,m)
    pd.DataFrame(spectra).drop(columns='maxnorm').to_csv(DATA/'full_lyapunov_spectra.csv',index=False)

    # High-resolution endpoint trajectories: all samples retained.
    print('High-resolution eta0 trajectory',flush=True)
    t0,x0,esc0=integrate(P0,0.0,DT,TRANSIENT,3000.0,1,BLOWUP)
    print('eta0 points',len(t0),'max',np.linalg.norm(x0,axis=1).max(),'escaped',esc0)
    save_npz('trajectory_eta0_high_resolution.npz',t0,x0)
    print('High-resolution eta1 trajectory',flush=True)
    t1,x1,esc1=integrate(P0,1.0,DT,TRANSIENT,500.0,1,BLOWUP)
    print('eta1 points',len(t1),'max',np.linalg.norm(x1,axis=1).max(),'escaped',esc1)
    save_npz('trajectory_eta1_high_resolution.npz',t1,x1)

    # Compact CSVs retain dense sampling but avoid huge manuscript-package CSVs.
    stride0=max(1,len(t0)//150000);stride1=max(1,len(t1)//100000)
    pd.DataFrame(np.c_[t0[::stride0],x0[::stride0]],columns=['t','x0','x1','x2','x3','x4']).to_csv(DATA/'trajectory_eta0_dense.csv',index=False)
    pd.DataFrame(np.c_[t1[::stride1],x1[::stride1]],columns=['t','x0','x1','x2','x3','x4']).to_csv(DATA/'trajectory_eta1_dense.csv',index=False)

    # Poincare sections.
    pc=poincare_section(t0,x0,1,0,+1);pp=poincare_section(t1,x1,1,0,+1)
    pd.DataFrame(pc[:,1:],columns=['x0','x1','x2','x3','x4']).to_csv(DATA/'poincare_eta0.csv',index=False)
    pd.DataFrame(pp[:,1:],columns=['x0','x1','x2','x3','x4']).to_csv(DATA/'poincare_eta1.csv',index=False)

    # Spectra and spectral diagnostics.
    f0,pw0,e0,b0,top0=fft_metrics(t0,x0[:,2])
    f1,pw1,e1,b1,top1=fft_metrics(t1,x1[:,2])
    pd.DataFrame({'frequency':f0,'normalized_power':pw0}).to_csv(DATA/'power_spectrum_eta0.csv',index=False)
    pd.DataFrame({'frequency':f1,'normalized_power':pw1}).to_csv(DATA/'power_spectrum_eta1.csv',index=False)
    pd.DataFrame([
        dict(eta=0,spectral_entropy=e0,bins_for_90_percent_power=b0,top_10_power_fraction=top0,poincare_crossings=len(pc)),
        dict(eta=1,spectral_entropy=e1,bins_for_90_percent_power=b1,top_10_power_fraction=top1,poincare_crossings=len(pp)),
    ]).to_csv(DATA/'spectral_diagnostics.csv',index=False)

    # Long-time boundedness from the new initial condition.
    print('Long-time boundedness',flush=True)
    m0,a0=boundedness_windows(P0,0.0,DT,10000.0,1000.0,BLOWUP)
    m1,a1=boundedness_windows(P0,1.0,DT,10000.0,1000.0,BLOWUP)
    pd.DataFrame({'window':np.arange(1,11),'eta0_max_norm':m0,'eta0_mean_norm':a0,
                  'eta1_max_norm':m1,'eta1_mean_norm':a1}).to_csv(DATA/'long_time_boundedness.csv',index=False)

    # Radial mean balance and cycle-resolved eta1 data.
    print('Radial balances',flush=True)
    r0=radial_terms(x0,0.0)
    # Use the archived refined cycle for cycle-accurate eta=1 means.
    per=np.load(DATA/'periodic_orbit_refined.npz') if (DATA/'periodic_orbit_refined.npz').exists() else None
    if per is not None:
        # Usually keys t and x; fall back to dense archive.
        keys=per.files
        Xcycle=None;Tcycle=None
        for k in keys:
            arr=per[k]
            if arr.ndim==2 and arr.shape[1]==5:Xcycle=arr
            if arr.ndim==1 and len(arr)>2:Tcycle=arr
        if Xcycle is None:
            dense=np.load(DATA/'periodic_orbit_dense.npz');Xcycle=dense['x'];Tcycle=dense['t']
    else:
        dense=np.load(DATA/'periodic_orbit_dense.npz');Xcycle=dense['x'];Tcycle=dense['t']
    r1=radial_terms(Xcycle,1.0)
    rb=pd.DataFrame([
        dict(eta=0,linear_mean=r0[:,0].mean(),left_quadratic_mean=r0[:,1].mean(),
             associator_mean_raw=r0[:,2].mean(),active_associator_mean=0.0,total_mean=r0[:,3].mean()),
        dict(eta=1,linear_mean=r1[:,0].mean(),left_quadratic_mean=r1[:,1].mean(),
             associator_mean_raw=r1[:,2].mean(),active_associator_mean=r1[:,2].mean(),total_mean=r1[:,3].mean()),
    ])
    rb.to_csv(DATA/'radial_energy_balance.csv',index=False)
    if Tcycle is None or np.ndim(Tcycle)==0:
        Tcycle=np.linspace(0,1,len(Xcycle))
    pd.DataFrame({'t':np.asarray(Tcycle)[:len(Xcycle)],'linear':r1[:,0],
                  'left_quadratic':r1[:,1],'active_associator':r1[:,2],
                  'total':r1[:,3]}).to_csv(DATA/'periodic_cycle_energy_balance.csv',index=False)

    # Timestep refinement with the new initial condition.
    print('Timestep refinement',flush=True)
    dtrows=[]
    for eta in (0.0,1.0):
        for dt in (0.01,0.005,0.0025):
            spec,div,ht,hs,q,m=full_lyapunov(P0,eta,dt,1000.0,1000.0,0.25,100.0,BLOWUP)
            dtrows.append(dict(eta=eta,dt=dt,**{f'l{i+1}':spec[i] for i in range(5)},mean_div=div,maxnorm=m))
            print('dt',eta,dt,spec[0],m)
    pd.DataFrame(dtrows).to_csv(DATA/'timestep_convergence.csv',index=False)

    # Eta scan and bifurcation data (new p0).
    print('Eta scan',flush=True)
    scan=[];bif=[]
    for eta in np.linspace(0,1,41):
        le,m=largest_exponent(P0,float(eta),0.005,500.0,1000.0,0.25,BLOWUP)
        ts,xs,esc=integrate(P0,float(eta),0.005,500.0,400.0,1,BLOWUP)
        if len(xs)>2:
            y=xs[:,2];peaks=np.where((y[1:-1]>y[:-2])&(y[1:-1]>=y[2:]))[0]+1
            if len(peaks)>300:peaks=peaks[-300:]
            for j in peaks:bif.append((eta,y[j]))
        scan.append(dict(eta=eta,largest_Lyapunov=le,maximum_norm=m,escaped=esc))
        print(f'  {eta:.3f} {le:.5f} {m:.3f}',flush=True)
    pd.DataFrame(scan).to_csv(DATA/'eta_scan_largest_lyapunov.csv',index=False)
    pd.DataFrame(bif,columns=['eta','local_maximum_xj']).to_csv(DATA/'eta_bifurcation.csv',index=False)

    # Equilibrium visitation for the new high-resolution chaotic trajectory.
    eq=pd.read_csv(DATA/'equilibria_eta0.csv').to_numpy()
    visit_equilibria(x0,eq).to_csv(DATA/'chaotic_equilibrium_visitation.csv',index=False)

    # Summary.
    ly=pd.DataFrame(spectra)
    br=rb
    summary={
        'p0':P0.tolist(),
        'p0_exact':'(1-i-j+k-l)/2',
        'initial_associator':assoc.tolist(),
        'initial_associator_exact':'(3 sqrt(2)+7 i+13 j-5 k-15 ell)/8',
        'initial_associator_norm':float(np.linalg.norm(assoc)),
        'initial_associator_norm_exact':'9 sqrt(6)/8',
        'eta0_lyapunov':[spectra[0][f'lambda_{i}'] for i in range(1,6)],
        'eta1_lyapunov':[spectra[1][f'lambda_{i}'] for i in range(1,6)],
        'eta0_mean_divergence':spectra[0]['mean_divergence'],
        'eta1_mean_divergence':spectra[1]['mean_divergence'],
        'eta0_max_norm_10000':float(np.nanmax(m0)),
        'eta1_max_norm_10000':float(np.nanmax(m1)),
        'eta0_spectral_entropy':e0,'eta1_spectral_entropy':e1,
        'eta0_bins_for_90_percent_power':b0,'eta1_bins_for_90_percent_power':b1,
        'eta0_top10_power_fraction':top0,'eta1_top10_power_fraction':top1,
        'eta0_trajectory_samples':len(t0),'eta1_trajectory_samples':len(t1),
        'trajectory_dt':DT,
        'runtime_seconds':time.time()-start,
    }
    # Copy model-invariant periodic data into summary.
    pom=pd.read_csv(DATA/'periodic_orbit_metrics.csv').iloc[0]
    ppnt=pd.read_csv(DATA/'periodic_orbit_phase_point.csv').iloc[0]
    fl=pd.read_csv(DATA/'floquet_multipliers.csv')
    summary.update(period=float(pom['period']),periodic_phase_point=ppnt.tolist(),
                   floquet_multipliers=fl[['multiplier_real','multiplier_imag','modulus']].to_dict('records'),
                   saddle_node_eta_estimate=0.50903065)
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
