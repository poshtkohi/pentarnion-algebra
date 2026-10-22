#!/usr/bin/env python3
"""Compute Floquet tables, crossings and orbit samples from recomputed monodromy data.

The neutral multiplier is selected by alignment with the flow direction. Event
values are calculated from adjacent samples, never copied from reference tables.
A detected crossing on P32 is not a computed P64 orbit. Endpoints are not folds.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from switching_algebra_model import static_rhs
BRANCH_FILES={
 'P1':['primary_down.csv','primary_up.csv','refine_PD1.csv'],
 'P2':['cascade_P2_down.csv','cascade_P2_up.csv'],
 'P4':['cascade_P4_down.csv','cascade_P4_up.csv'],
 'P8':['cascade_P8_down.csv','cascade_P8_up.csv'],
 'P16':['cascade_P16_down.csv','cascade_P16_up.csv'],
 'P32':['refine_P32_plus1.csv','refine_P32_cont1.csv','refine_P32_cont2.csv','refine_P32_cont3.csv'],
 'Broad_P2':['broad_down.csv','broad_up.csv','refine_broad_lower.csv','refine_broad_reversePD.csv','refine_broad_upper.csv'],
 'Window_C1':['window_C1_down.csv','window_C1_up.csv','refine_window_C1_lower.csv'],
 'Window_C2':['window_C2_down.csv','window_C2_up.csv'],
 'Window_C4':['window_C4_down.csv','window_C4_up.csv'],
 'Narrow_P4':['narrow_down.csv','narrow_up.csv','refine_narrow_lower.csv','refine_narrow_PD.csv']}


def enrich(data:pd.DataFrame) -> pd.DataFrame:
    required=['gamma','T','res']+['x'+str(i) for i in range(5)]+['m'+str(i) for i in range(25)]
    if data.empty or not np.isfinite(data[required].to_numpy()).all():raise ValueError('Invalid or empty continuation table')
    if (data['T']<=0).any() or (data.res>1e-6).any():raise ValueError('Continuation residual or period outside accepted bounds')
    rows=[]
    for _,r in data.iterrows():
        M=np.array([r[f'm{i}'] for i in range(25)]).reshape(5,5)
        ev,V=np.linalg.eig(M);flow=static_rhs(0,np.array([r[f'x{i}'] for i in range(5)]),float(r.gamma));nf=np.linalg.norm(flow)
        if nf<1e-12:raise ValueError('Equilibrium is not a nonconstant periodic orbit')
        alignment=np.abs(V.conj().T@flow)/(np.linalg.norm(V,axis=0)*nf);j=int(np.argmax(alignment));trans=np.delete(ev,j);trans=trans[np.argsort(-abs(trans))]
        row={'neutral_real':float(ev[j].real),'neutral_imag':float(ev[j].imag),'neutral_alignment':float(alignment[j]),'max_transverse_modulus':float(max(abs(trans))),'stable':int(max(abs(trans))<1.)}
        for k,z in enumerate(trans,1):row.update({f'mu{k}_real':float(z.real),f'mu{k}_imag':float(z.imag),f'mu{k}_modulus':float(abs(z))})
        rows.append(row)
    return pd.concat([data.reset_index(drop=True),pd.DataFrame(rows)],axis=1)


def crossings(data:pd.DataFrame) -> list[dict]:
    selected=[]
    for _,r in data.iterrows():
        ev=[complex(r[f'mu{i}_real'],r[f'mu{i}_imag']) for i in range(1,5)]
        real=[z.real for z in ev if abs(z.imag)<1e-6]
        selected.append(min(real,key=lambda z:abs(z+1)) if real else np.nan)
    hits=[]
    for i in range(len(data)-1):
        a,b=selected[i]+1,selected[i+1]+1
        if not np.isfinite(a+b) or a*b>0 or a==b:continue
        x,y=data.iloc[i],data.iloc[i+1];u=-a/(b-a)
        hits.append({'gamma':float(x.gamma+u*(y.gamma-x.gamma)),'parent_period':float(x['T']+u*(y['T']-x['T'])),'gamma_left':float(x.gamma),'gamma_right':float(y.gamma),'sample_separation':float(y.gamma-x.gamma),'multiplier_left':float(a-1),'multiplier_right':float(b-1)})
    return hits


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--orbits',action='store_true');a=p.parse_args()
    source=a.data.resolve();out=(a.output or source).resolve();ref=HERE.parents[2]/'data'
    if out==ref or ref in out.parents:p.error('Output must be outside reference data')
    out.mkdir(parents=True,exist_ok=True);branches={};all_events={};rows=[]
    for name,files in BRANCH_FILES.items():
        tables=[pd.read_csv(source/f) for f in files if (source/f).exists()]
        if not tables:raise FileNotFoundError(f'No input tables for {name} in {source}')
        d=pd.concat(tables,ignore_index=True);d['_key']=d.gamma.round(12);d=d.drop_duplicates('_key',keep='last').drop(columns='_key').sort_values('gamma')
        d=enrich(d);d.insert(0,'branch',name);d.to_csv(out/f'{name}_floquet.csv',index=False);branches[name]=d
        ev=crossings(d);all_events[name]=ev
        if name in ['P1','P2','P4','P8','P16','P32']:
            if not ev:raise RuntimeError(f'No -1 crossing detected on {name}')
            selected=max(ev,key=lambda r:r['gamma']);selected.update({'doubling':f'{name}→P{int(name[1:])*2}','parent_branch':name,'method':'adjacent samples; nearest real multiplier to -1; largest-parameter crossing','daughter_continued':name!='P32'});rows.append(selected)
    pdtab=pd.DataFrame(rows);g=pdtab.gamma.to_numpy();d=np.diff(g);pdtab['interval_from_previous']=np.r_[np.nan,d];pdtab['successive_interval_ratio']=np.r_[np.nan,np.nan,d[:-1]/d[1:]];pdtab.to_csv(out/'primary_period_doubling_cascade.csv',index=False)
    (out/'computed_crossings.json').write_text(json.dumps(all_events,indent=2)+'\n')
    summary={'event_values_status':'Recomputed interpolation of the supplied generated monodromy matrices; not interval-certified.','endpoint_interpretation':'Natural-parameter continuation endpoints are recorded only; they do not establish folds or closed isolas.','branch_rows':{k:len(v) for k,v in branches.items()},'maximum_shooting_residual':float(max(d.res.max() for d in branches.values())),'primary_flip_estimates':pdtab[['doubling','gamma','sample_separation']].to_dict('records')}
    if a.orbits:
        for name,target in [('P1',.4),('P2',.48),('P4',.49),('P8',.4923),('P16',.493),('P32',.49318)]:
            df=branches[name];r=df.iloc[np.argmin(abs(df.gamma-target))];y0=np.array([r[f'x{i}'] for i in range(5)]);ts=np.linspace(0,float(r['T']),2401)
            sol=solve_ivp(lambda t,y:static_rhs(t,y,float(r.gamma)),(0,ts[-1]),y0,t_eval=ts,method='DOP853',rtol=1e-11,atol=1e-13,max_step=.03)
            if not sol.success:raise RuntimeError(sol.message)
            dd=pd.DataFrame(sol.y.T,columns=['w','x','y','z','u']);dd.insert(0,'t',ts);dd.to_csv(out/f'orbit_{name}.csv',index=False)
    (out/'recomputed_branch_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(pdtab.to_string(index=False));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
