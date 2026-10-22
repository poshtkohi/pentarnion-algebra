#!/usr/bin/env python3
"""Validate freshly computed files and record differences from paper reference data.

A numerical execution PASS is not a claim of bitwise reference reproduction.
Long-horizon differences are retained explicitly in numerical_comparison.json.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('--task',required=True);a=p.parse_args();r=a.report.resolve();task=a.task
    pi=r/'generated/part-I/data';pii=r/'generated/part-II/data';ref=ROOT/'part-II/data'
    tests=[];comparison=[]
    def check(name,condition,details=None):
        tests.append({'check':name,'passed':bool(condition),'details':details})
        if not condition:raise AssertionError(name+': '+str(details))
    def selected(t):return task in ('all',t)
    def compare(new,old,columns):
        x=pd.read_csv(new);y=pd.read_csv(old);row={'new_file':str(new.relative_to(r)),'reference_file':str(old.relative_to(ROOT)),'new_rows':len(x),'reference_rows':len(y)}
        if len(x)==len(y):
            for c in columns:
                if c not in x or c not in y:continue
                ok=np.isfinite(x[c])&np.isfinite(y[c]);row[c+'_max_absolute_difference']=float(np.max(abs(x.loc[ok,c]-y.loc[ok,c]))) if ok.any() else None
        else:row['comparison']='Different sample counts; no rowwise comparison attempted.'
        comparison.append(row)
    if selected('scans'):
        for f,folder,n,col in [('static_scan.csv','riccati',201,'gamma'),('time_scan_transition_band.csv','time_switching',15,'omega'),('time_scan_full_range_regularized.csv','time_switching',15,'omega')]:
            d=pd.read_csv(pii/folder/f);check(f+' row count',len(d)==n,len(d));ok=d.blowup==0;check(f+' successful-run exponents finite',np.isfinite(d.loc[ok,'lle']).all());check(f+' stopped runs excluded',d.loc[~ok,'lle'].isna().all());check(f+' increasing grid',(np.diff(d[col])>0).all());compare(pii/folder/f,ref/folder/f,['lle','tail_mean','tail_max'])
        d=pd.read_csv(pii/'riccati/static_scan.csv');check('static stable-side sampled exponents',bool((d.loc[d.gamma<.36,'lle']<0).all()));check('static gamma 0.6 exponent positive',float(d.loc[np.isclose(d.gamma,.6),'lle'].iloc[0])>0)
        d=pd.read_csv(pii/'time_switching/time_scan_full_range_regularized.csv');end=d.loc[np.isclose(d.omega,2)].iloc[0]
        comparison.append({'quantity':'Protocol II omega=2 integration status','reference_stop_flag':1,'new_stop_flag':int(end.blowup),'new_finite_time_exponent':None if not np.isfinite(end.lle) else float(end.lle),'interpretation':'A numerical failure is not required to reproduce across compilers; any completed run is assessed separately.'})
        if not int(end.blowup):check('Protocol II omega=2 completed run agrees with adaptive late exponent',abs(end.lle+.250780896887)<.002,float(end.lle))
    if selected('peaks'):
        f=pii/'riccati/static_bifurcation_peaks.csv';d=pd.read_csv(f);check('551 peak-scan parameter values',d.gamma.nunique()==551);check('peak records finite',np.isfinite(d.to_numpy()).all());check('at most 150 retained peaks',d.groupby('gamma').size().max()<=150);compare(f,ref/'riccati/static_bifurcation_peaks.csv',['z_max'])
    if selected('spectra'):
        f=pii/'riccati/full_lyapunov_spectra.csv';d=pd.read_csv(f);check('12 full-spectrum points',len(d)==12);check('spectra finite',np.isfinite(d.to_numpy()).all());ls=d[[f'lambda{i}' for i in range(1,6)]].to_numpy();check('spectra descending order',bool((np.diff(ls,axis=1)<=1e-12).all()));check('gamma=0 spectrum',np.max(abs(ls[0]+.5))<1e-4);compare(f,ref/'riccati/full_lyapunov_spectra.csv',[f'lambda{i}' for i in range(1,6)])
    if selected('dense'):
        sizes={'static_gamma_0p60':200001,'static_gamma_0p70':60001,'protocol_I_omega_0p2':50001,'protocol_I_omega_0p8':50001,'protocol_II_omega_0p2':50001,'protocol_II_omega_0p8':50001,'protocol_I_omega_0p8_dense':230001,'protocol_II_omega_0p2_dense':230001}
        for stem,n in sizes.items():
            d=pd.read_csv(pii/'riccati_dense'/f'{stem}_timeseries.csv');check(stem+' rows',len(d)==n,len(d));check(stem+' finite',np.isfinite(d.to_numpy()).all());check(stem+' times',(np.diff(d.t)>0).all());u='u' if stem.startswith('static') else 'u_active';radius=np.linalg.norm(d[['w','x','y','z',u]].to_numpy(),axis=1);check(stem+' radius consistency',np.max(abs(radius-d.radius))<1e-10)
            if u=='u_active':check(stem+' coordinate transformation',np.max(abs(d.u_active-d.gamma*d.q4_hidden))<1e-11)
        for tag,n in [('gamma_0p60',803),('gamma_0p70',287)]:
            d=pd.read_csv(pii/'riccati_dense'/f'static_{tag}_poincare.csv');tests.append({'check':'crossing count '+tag,'passed':True,'details':{'computed':len(d),'reference':n,'interpretation':'Count recorded; no assumed identical chaotic path.'}})
    if selected('branches'):
        b=pii/'riccati/periodic_branches';raw=[f for f in b.glob('*.csv') if f.name.startswith(('primary_','cascade_','refine_','window_','broad_','narrow_')) and 'doubling' not in f.name]
        check('33 raw continuation/refinement files',len(raw)==33,len(raw))
        for f in raw:
            d=pd.read_csv(f);check(f.name+' successful shooting',len(d)>0 and np.isfinite(d.to_numpy()).all() and d.res.max()<1e-7 and d['T'].min()>0,{'rows':len(d),'max_residual':float(d.res.max())});compare(f,ref/'riccati/periodic_branches'/f.name,['T','res'])
        d=pd.read_csv(b/'primary_period_doubling_cascade.csv');check('six computed flip estimates',len(d)==6);check('flip ordering',(np.diff(d.gamma)>0).all());old=pd.read_csv(ref/'riccati/periodic_branches/primary_period_doubling_cascade.csv');err=float(np.max(abs(d.gamma-old.gamma)));check('flip estimates within 2e-4 of reference',err<2e-4,err);comparison.append({'quantity':'primary_flip_gamma_max_error','value':err})
    if selected('validation'):
        v=pii/'riccati/validation';d=json.loads((v/'symbolic_and_hopf_checks.json').read_text());check('characteristic residual zero',d['charpoly_residual']=='0');check('Hopf coefficient',abs(d['first_Lyapunov_coefficient']+.1832582814093947)<1e-10,d);check('eigenvectors',max(d['eigenvector_residual'],d['adjoint_residual'])<1e-12)
        d=pd.read_csv(v/'independent_orbit_checks.csv');check('seven periodic seeds reintegrated',len(d)==7);check('adaptive periodic closures',d.adaptive_closure_residual.max()<1e-7,float(d.adaptive_closure_residual.max()));check('Floquet reference radii',np.max(abs(d.transverse_radius-d.archive_transverse_radius))<1e-6)
        for tag in ['dt005_T2000','dt0025_T2000','dt0025_T4000']:
            f=v/f'spectrum_gamma060_{tag}.csv';d=pd.read_csv(f).iloc[0];check(tag+' positive exponent',d.lambda1>0,float(d.lambda1));check(tag+' divergence trace check',abs(d.sum_minus_mean_divergence)<2e-7,float(d.sum_minus_mean_divergence));compare(f,ref/'riccati/validation'/f.name,['lambda1','mean_divergence'])
        d=json.loads((v/'protocol_II_omega2_periodic_response.json').read_text());check('Protocol II periodic closure',d['one_period_return_residual']<1e-9,d['one_period_return_residual']);check('Protocol II spectral radius',abs(d['stroboscopic_spectral_radius']-.454820963936)<1e-6,d['stroboscopic_spectral_radius'])
        for tag,n in [('I',147),('II',36)]:
            d=pd.read_csv(v/f'protocol_{tag}_archived_stroboscopic.csv');check('phase sample count '+tag,len(d)==n);check('phase alignment interval '+tag,d.local_alignment_interval.max()<.0050001)
    if selected('spin-scan') or task=='smoke':
        f=pi/'spin/representative_lyapunov_scan.csv';d=pd.read_csv(f);check('spin scan finite',np.isfinite(d.to_numpy()).all());check('spin projected norm',d.max_norm_error.max()<1e-12);check('spin tangent exponent sum',d.sum_exponents.abs().max()<1e-7,float(d.sum_exponents.abs().max()))
        if task!='smoke':
            check('14 spin frequencies',len(d)==14);check('spin omega=1.2 positive',float(d.loc[np.isclose(d.Omega,1.2),'lambda_max'].iloc[0])>.01);check('spin omega=0.3 small exponent',abs(float(d.loc[np.isclose(d.Omega,.3),'lambda_max'].iloc[0]))<.005);compare(f,ROOT/'part-I/data/spin/representative_lyapunov_scan.csv',['lambda_max'])
    if task=='smoke':
        d=pd.read_csv(pii/'riccati/periodic_branches/smoke_orbit.csv');check('single orbit smoke closure',len(d)==1 and d.res.max()<1e-8)
        d=pd.read_csv(pii/'riccati/validation/smoke_spectrum.csv');check('short spectrum trace',d.sum_minus_mean_divergence.abs().max()<1e-5)
    (r/'numerical_checks.json').write_text(json.dumps({'status':'PASS','checks':tests},indent=2)+'\n')
    (r/'numerical_comparison.json').write_text(json.dumps({'interpretation':'Differences are retained. Passing implementation and numerical checks does not mean every historical finite-time table is replicated. Sensitive long runs and reconstruction-specific spin settings must be assessed separately.','comparisons':comparison},indent=2)+'\n')
    print('PASS',len(tests),'numerical consistency checks. Reference comparisons recorded separately.')

if __name__=='__main__':main()
