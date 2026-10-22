#!/usr/bin/env python3
"""Audit the six reported flips against adjacent archived Floquet samples.

This performs secant interpolation of archived multiplier values, not a new
shooting solve, eigenbranch-continuation proof, or interval certificate. Original
CSVs are never overwritten. The last event does not supply a P64 daughter orbit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def nearest_real_multiplier(row: pd.Series) -> float:
    values = [complex(row[f'mu{j}_real'],row[f'mu{j}_imag']) for j in range(1,5)]
    real = [z.real for z in values if abs(z.imag)<1e-6]
    return min(real,key=lambda x:abs(x+1)) if real else np.nan

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    package=Path(__file__).resolve().parents[3]
    parser.add_argument('--data',type=Path,default=package/'data/riccati/periodic_branches')
    parser.add_argument('--output',type=Path,default=package/'outputs/numerical/validation/primary_flip_brackets.csv')
    args=parser.parse_args()
    reference=pd.read_csv(args.data/'primary_period_doubling_cascade.csv')
    records=[]
    for index,name in enumerate(['P1','P2','P4','P8','P16','P32']):
        target=float(reference.iloc[index]['gamma'])
        df=pd.read_csv(args.data/f'{name}_floquet.csv').sort_values('gamma').drop_duplicates('gamma')
        df['selected_real_multiplier']=df.apply(nearest_real_multiplier,axis=1)
        candidates=[]
        for j in range(len(df)-1):
            a,b=df.iloc[j],df.iloc[j+1]
            y0,y1=a.selected_real_multiplier+1,b.selected_real_multiplier+1
            if not np.isfinite(y0+y1) or y0*y1>0 or y1==y0:continue
            est=float(a.gamma-y0*(b.gamma-a.gamma)/(y1-y0))
            candidates.append((abs(est-target),a,b,est))
        if not candidates:raise RuntimeError(f'No archived sign-change bracket found for {name}')
        _,a,b,estimate=min(candidates,key=lambda x:x[0])
        records.append({'parent_branch':name,'reference_gamma':target,
            'gamma_left':a.gamma,'gamma_right':b.gamma,'bracket_width':b.gamma-a.gamma,
            'multiplier_left':a.selected_real_multiplier,'multiplier_right':b.selected_real_multiplier,
            'secant_gamma':estimate,'secant_minus_reference':estimate-target,
            'source_csv':str((args.data/f'{name}_floquet.csv').name),
            'method':'nearest-real-to-minus-one; adjacent archived samples; secant interpolation',
            'daughter_orbit_archived':name!='P32'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(args.output,index=False)
    print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':main()
