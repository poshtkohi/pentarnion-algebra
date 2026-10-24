#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
from numba import njit

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'bracketing_flow'; HI=DATA/'high_resolution'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import regenerate_bracketing_flow as m

@njit(cache=True)
def radial_terms_batch(X,eta):
    vals=np.empty((X.shape[0],4),dtype=np.float64)
    for n in range(X.shape[0]):
        p=X[n]
        lin=2*np.dot(p,m.pmul(m.A,p))
        left=2*np.dot(p,m.pmul(m.B,m.pmul(p,p)))
        assoc=2*np.dot(p,m.pmul(m.pmul(m.B,p),p)-m.pmul(m.B,m.pmul(p,p)))
        vals[n,0]=lin;vals[n,1]=left;vals[n,2]=assoc;vals[n,3]=lin+left+eta*assoc
    return vals

def main():
    start=time.time()
    d0=np.load(HI/'trajectory_eta0_high_resolution.npz');t0=d0['t'];x0=d0['x']
    d1=np.load(HI/'trajectory_eta1_high_resolution.npz');t1=d1['t'];x1=d1['x']

    print('Radial balance batch',flush=True)
    r0=radial_terms_batch(x0,0.0)
    dense=np.load(DATA/'periodic_orbit_dense.npz');tc=dense['t'];xc=dense['x']
    r1=radial_terms_batch(xc,1.0)
    rb=pd.DataFrame([
        dict(eta=0,linear_mean=r0[:,0].mean(),left_quadratic_mean=r0[:,1].mean(),associator_mean_raw=r0[:,2].mean(),active_associator_mean=0.0,total_mean=r0[:,3].mean()),
        dict(eta=1,linear_mean=r1[:,0].mean(),left_quadratic_mean=r1[:,1].mean(),associator_mean_raw=r1[:,2].mean(),active_associator_mean=r1[:,2].mean(),total_mean=r1[:,3].mean()),
    ])
    rb.to_csv(DATA/'radial_energy_balance.csv',index=False)
    pd.DataFrame({'t':tc,'linear':r1[:,0],'left_quadratic':r1[:,1],'active_associator':r1[:,2],'total':r1[:,3]}).to_csv(DATA/'periodic_cycle_energy_balance.csv',index=False)
    print(rb)

    print('Timestep refinement',flush=True)
    dtrows=[]
    for eta in (0.0,1.0):
        for dt in (0.01,0.005,0.0025):
            spec,div,ht,hs,q,mx=m.full_lyapunov(m.P0,eta,dt,1000.0,1000.0,0.25,100.0,m.BLOWUP)
            dtrows.append(dict(eta=eta,dt=dt,**{f'l{i+1}':spec[i] for i in range(5)},mean_div=div,maxnorm=mx))
            print('  dt',eta,dt,spec[0],mx,flush=True)
    pd.DataFrame(dtrows).to_csv(DATA/'timestep_convergence.csv',index=False)

    print('Eta scan',flush=True)
    scan=[];bif=[]
    for eta in np.linspace(0,1,41):
        le,mx=m.largest_exponent(m.P0,float(eta),0.005,500.0,1000.0,0.25,m.BLOWUP)
        ts,xs,esc=m.integrate(m.P0,float(eta),0.005,500.0,400.0,1,m.BLOWUP)
        if len(xs)>2:
            y=xs[:,2];peaks=np.where((y[1:-1]>y[:-2])&(y[1:-1]>=y[2:]))[0]+1
            if len(peaks)>300:peaks=peaks[-300:]
            for j in peaks:bif.append((eta,y[j]))
        scan.append(dict(eta=eta,largest_Lyapunov=le,maximum_norm=mx,escaped=esc))
        print(f'  eta={eta:.3f} LE={le:.6f} max={mx:.3f} escape={esc}',flush=True)
    pd.DataFrame(scan).to_csv(DATA/'eta_scan_largest_lyapunov.csv',index=False)
    pd.DataFrame(bif,columns=['eta','local_maximum_xj']).to_csv(DATA/'eta_bifurcation.csv',index=False)

    print('Equilibrium visitation',flush=True)
    eq=pd.read_csv(DATA/'equilibria_eta0.csv').to_numpy()
    m.visit_equilibria(x0,eq).to_csv(DATA/'chaotic_equilibrium_visitation.csv',index=False)

    # Update summary from generated data and model-invariant archives.
    assoc=m.pmul(m.pmul(m.B,m.P0),m.P0)-m.pmul(m.B,m.pmul(m.P0,m.P0))
    ly=pd.read_csv(DATA/'full_lyapunov_spectra.csv')
    long=pd.read_csv(DATA/'long_time_boundedness.csv')
    sp=pd.read_csv(DATA/'spectral_diagnostics.csv')
    pom=pd.read_csv(DATA/'periodic_orbit_metrics.csv').iloc[0]
    ppnt=pd.read_csv(DATA/'periodic_orbit_phase_point.csv').iloc[0]
    fl=pd.read_csv(DATA/'floquet_multipliers.csv')
    summary={
        'p0':m.P0.tolist(),'p0_exact':'(1-i-j+k-l)/2',
        'initial_associator':assoc.tolist(),
        'initial_associator_exact':'(3 sqrt(2)+7 i+13 j-5 k-15 ell)/8',
        'initial_associator_norm':float(np.linalg.norm(assoc)),
        'initial_associator_norm_exact':'9 sqrt(6)/8',
        'eta0_lyapunov':ly.loc[ly.eta==0,[f'lambda_{i}' for i in range(1,6)]].iloc[0].tolist(),
        'eta1_lyapunov':ly.loc[ly.eta==1,[f'lambda_{i}' for i in range(1,6)]].iloc[0].tolist(),
        'eta0_mean_divergence':float(ly.loc[ly.eta==0,'mean_divergence'].iloc[0]),
        'eta1_mean_divergence':float(ly.loc[ly.eta==1,'mean_divergence'].iloc[0]),
        'period':float(pom['period']),'periodic_phase_point':ppnt.tolist(),
        'floquet_multipliers':fl[['multiplier_real','multiplier_imag','modulus']].to_dict('records'),
        'eta0_max_norm_10000':float(long.eta0_max_norm.max()),
        'eta1_max_norm_10000':float(long.eta1_max_norm.max()),
        'eta0_spectral_entropy':float(sp.loc[sp.eta==0,'spectral_entropy'].iloc[0]),
        'eta1_spectral_entropy':float(sp.loc[sp.eta==1,'spectral_entropy'].iloc[0]),
        'eta0_bins_for_90_percent_power':int(sp.loc[sp.eta==0,'bins_for_90_percent_power'].iloc[0]),
        'eta1_bins_for_90_percent_power':int(sp.loc[sp.eta==1,'bins_for_90_percent_power'].iloc[0]),
        'eta0_top10_power_fraction':float(sp.loc[sp.eta==0,'top_10_power_fraction'].iloc[0]),
        'eta1_top10_power_fraction':float(sp.loc[sp.eta==1,'top_10_power_fraction'].iloc[0]),
        'eta0_trajectory_samples':len(t0),'eta1_trajectory_samples':len(t1),'trajectory_dt':0.005,
        'saddle_node_eta_estimate':0.50903065,
        'resume_runtime_seconds':time.time()-start,
    }
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
