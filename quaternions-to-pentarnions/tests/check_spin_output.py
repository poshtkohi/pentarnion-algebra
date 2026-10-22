#!/usr/bin/env python3
"""Sanity-check newly generated spin files; do not demand chaotic pointwise agreement."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    for tag,rows,periods,display in [('regular',24001,900,2293),('chaotic',91001,3000,10429)]:
        data=pd.read_csv(args.directory/f'{tag}_timeseries.csv')
        section=pd.read_csv(args.directory/f'{tag}_poincare.csv')
        view=pd.read_csv(args.directory/f'{tag}_display_window.csv')
        if (len(data),len(section),len(view))!=(rows,periods,display):raise RuntimeError('Unexpected output sizes: '+tag)
        if not np.isfinite(data.to_numpy()).all():raise RuntimeError('Nonfinite values: '+tag)
        if not np.all(np.diff(data.t)>0):raise RuntimeError('Unordered times: '+tag)
        error=float(np.max(np.abs(np.linalg.norm(data[['y_h','y_b','y_c']].to_numpy(),axis=1)-1)))
        if error>1e-12:raise RuntimeError('Spin norm error: '+str(error))
        work=float(np.max(np.abs(data.work_error)))
        if work>1e-3:raise RuntimeError('Unexpectedly large work residual: '+str(work))
        print(f'PASS {tag}: rows={rows}, periods={periods}, norm_error={error:.3g}, max_sampled_work_residual={work:.6g}')
    print('These finite numerical sanity checks are not rigorous invariant-set or Lyapunov certificates.')

if __name__=='__main__':main()
