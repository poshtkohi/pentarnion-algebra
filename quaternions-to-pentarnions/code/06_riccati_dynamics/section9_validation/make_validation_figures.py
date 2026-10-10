#!/usr/bin/env python3
"""Plot the supplied Riccati validation data.

The required CSVs are included. This script does not rerun the numerical
experiments and does not require the original validation generators.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'data/riccati/section9_validation'
BR=ROOT/'data/riccati/periodic_branches'
FIG = Path(os.environ.get('PENTARNION_OUTPUT_DIR', str(ROOT/'outputs/plots'))).expanduser().resolve()
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size':14,'axes.labelsize':14,'axes.titlesize':14,
                     'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':10.5})
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/name,dpi=260,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,4.8))
for name in ['P2','P4','P8','P16','P32']:
    d=pd.read_csv(BR/f'{name}_floquet.csv').sort_values('gamma')
    q=d[(d.stable==1)&(d.gamma>=.472)&(d.gamma<=.49325)]
    ax.plot(q.gamma,q['T'],marker='.',linewidth=1.2,label=fr'$P_{{{name[1:]}}}$')
events=pd.read_csv(BR/'primary_period_doubling_cascade.csv')
ax.scatter(events.gamma,events.parent_period,marker='x',s=38,label='flip estimates')
ax.set_xlim(.472,.49335);ax.set_yscale('log')
ax.set_xlabel(r'$\gamma$');ax.set_ylabel('period')
ax.set_title('Primary-cascade detail from continued periodic branches')
ax.ticklabel_format(style='plain',axis='x',useOffset=False)
ax.legend(ncol=2);ax.grid(True,alpha=.35);save(fig,'riccati_primary_cascade_zoom.png')

for name,omega in [('I',.8),('II',.2)]:
    df=pd.read_csv(DATA/f'protocol_{name}_archived_stroboscopic.csv')
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    ax.scatter(df.p0,df.p3,s=14,alpha=.75)
    ax.set_xlabel(r'$p_0$ at $\omega t=2\pi n$')
    ax.set_ylabel(r'$p_3$')
    ax.set_title(f'Protocol {name}, $\\omega={omega:g}$: {len(df)} phase-aligned samples')
    ax.grid(True,alpha=.35);save(fig,f'protocol_{name}_stroboscopic.png')

late=pd.read_csv(DATA/'protocol_II_omega2_late_response.csv')
fig,ax=plt.subplots(figsize=(8.6,4.8));tau=late.t-late.t.iloc[0]
for j in range(5):ax.plot(tau,late[f'p{j}'],linewidth=1.25,label=fr'$p_{j}$')
ax.set_xlabel('elapsed time from $t=1200$');ax.set_ylabel('state component')
ax.set_title(r'Protocol II, $\omega=2$: late forcing-periodic response')
forcing=ax.twinx();forcing.plot(tau,late.gamma,linestyle='--',linewidth=1.1,label=r'$\gamma(t)$')
forcing.set_ylabel(r'$\gamma(t)$');forcing.set_ylim(0,1.03)
h,l=ax.get_legend_handles_labels();hh,ll=forcing.get_legend_handles_labels()
ax.legend(h+hh,l+ll,ncol=3,loc='upper right');ax.grid(True,alpha=.35)
save(fig,'protocol_II_omega2_periodic_response.png')

returns=pd.read_csv(DATA/'protocol_II_omega2_stroboscopic.csv').dropna()
fig,ax=plt.subplots(figsize=(6.8,4.8))
ax.semilogy(returns.forcing_cycle,np.maximum(returns.successive_return_distance,1e-16),linewidth=1.1)
ax.set_xlabel('forcing cycle $n$');ax.set_ylabel(r'$\|p(n\pi)-p((n-1)\pi)\|_2$')
ax.set_title(r'Protocol II, $\omega=2$: successive stroboscopic returns')
ax.grid(True,alpha=.35);save(fig,'protocol_II_omega2_return_residual.png')
print('Wrote five Section 9 validation figures.')
