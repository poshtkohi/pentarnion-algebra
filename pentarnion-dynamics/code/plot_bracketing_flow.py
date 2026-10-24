#!/usr/bin/env python3
"""Generate compact, publication-quality figures for the sparse rational
Pentarnion model using the revised low-rational initial condition.

The chaotic trajectory panel retains all 600,001 archived samples.  The
periodic panel uses one complete attracting period sampled at dt=0.005; the
full 100,001-sample endpoint archive remains in data/bracketing_flow/high_resolution.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures'; DATA=ROOT/'data'/'bracketing_flow'; HI=DATA/'high_resolution'
FIG.mkdir(exist_ok=True)

D0=np.load(HI/'trajectory_eta0_high_resolution.npz');t0=D0['t'];x0=D0['x']
D1=np.load(HI/'trajectory_eta1_high_resolution.npz');t1=D1['t'];x1=D1['x']
summary=json.loads((DATA/'summary.json').read_text())
T=float(summary['period'])

# ------------------------------------------------------------------
# 1. High-density endpoint trajectories and all five time series.
# ------------------------------------------------------------------
fig=plt.figure(figsize=(13.4,9.4))
gs=GridSpec(2,2,figure=fig,wspace=.22,hspace=.25)

a=fig.add_subplot(gs[0,0],projection='3d')
a.plot(x0[:,0],x0[:,3],x0[:,4],linewidth=.16,alpha=.75,rasterized=True)
a.set_xlabel(r'$x_0$');a.set_ylabel(r'$x_3$');a.set_zlabel(r'$x_4$')
a.set_title(r'(a) Left bracketing, $\eta=0$: 600,001 samples')
a.view_init(25,42)

# One complete asymptotic period with approximately 1159 samples.
period_mask=t1>=t1[-1]-T
b=fig.add_subplot(gs[0,1],projection='3d')
b.plot(x1[period_mask,0],x1[period_mask,3],x1[period_mask,4],linewidth=1.0)
b.set_xlabel(r'$x_0$');b.set_ylabel(r'$x_3$');b.set_zlabel(r'$x_4$')
b.set_title(r'(b) Right bracketing, $\eta=1$: one attracting period')
b.view_init(25,42)

# Last 120 time units at eta=0, all points retained.
mask0=t0>=t0[-1]-120
c=fig.add_subplot(gs[1,0])
for j,label in enumerate([r'$x_0$',r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$']):
    c.plot(t0[mask0]-(t0[-1]-120),x0[mask0,j],linewidth=.55,label=label)
c.set_xlabel('post-transient time');c.set_ylabel('component')
c.set_title('(c) Irregular recurrent endpoint signal')
c.grid(True,alpha=.25);c.legend(ncol=5,fontsize=7,loc='upper right')

# Last four periods at eta=1.
mask1=t1>=t1[-1]-4*T
d=fig.add_subplot(gs[1,1])
for j,label in enumerate([r'$x_0$',r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$']):
    d.plot(t1[mask1]-(t1[-1]-4*T),x1[mask1,j],linewidth=.72,label=label)
d.set_xlabel('time over the final four periods');d.set_ylabel('component')
d.set_title('(d) Converged periodic endpoint signal')
d.grid(True,alpha=.25);d.legend(ncol=5,fontsize=7,loc='upper right')

fig.suptitle(r'Sparse rational Pentarnion dynamics from $p(0)=\frac{1}{2}(1-i-j+k-\ell)$',y=.995)
fig.savefig(FIG/'bracketing_endpoint_dynamics.png',dpi=400,bbox_inches='tight')
fig.savefig(FIG/'bracketing_endpoint_dynamics.pdf',dpi=400,bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# 2. Numerical validation summary.
# ------------------------------------------------------------------
ly=pd.read_csv(DATA/'full_lyapunov_spectra.csv')
dt=pd.read_csv(DATA/'timestep_convergence.csv')
long=pd.read_csv(DATA/'long_time_boundedness.csv')
pc=pd.read_csv(DATA/'poincare_eta0.csv');pp=pd.read_csv(DATA/'poincare_eta1.csv')
f0=pd.read_csv(DATA/'power_spectrum_eta0.csv');f1=pd.read_csv(DATA/'power_spectrum_eta1.csv')
h0=np.load(HI/'lyapunov_history_eta0.npz');h1=np.load(HI/'lyapunov_history_eta1.npz')
sp=pd.read_csv(DATA/'spectral_diagnostics.csv')

fig,axes=plt.subplots(2,3,figsize=(14.8,8.8))
inds=np.arange(1,6);width=.34
v0=ly.loc[ly.eta==0,[f'lambda_{i}' for i in range(1,6)]].to_numpy().ravel()
v1=ly.loc[ly.eta==1,[f'lambda_{i}' for i in range(1,6)]].to_numpy().ravel()
axes[0,0].bar(inds-width/2,v0,width,label=r'$\eta=0$')
axes[0,0].bar(inds+width/2,v1,width,label=r'$\eta=1$')
axes[0,0].axhline(0,linewidth=.8);axes[0,0].set_xlabel('exponent index');axes[0,0].set_ylabel(r'$\lambda_r$')
axes[0,0].set_title('(a) Full Lyapunov spectra');axes[0,0].legend();axes[0,0].grid(True,axis='y',alpha=.25)

axes[0,1].plot(h0['t'],np.sort(h0['spectrum'],axis=1)[:,-1],label=r'$\eta=0$')
axes[0,1].plot(h1['t'],np.sort(h1['spectrum'],axis=1)[:,-1],label=r'$\eta=1$')
axes[0,1].axhline(0,linewidth=.8);axes[0,1].set_xlabel('measurement time');axes[0,1].set_ylabel(r'running $\lambda_1$')
axes[0,1].set_title('(b) Largest-exponent convergence');axes[0,1].legend();axes[0,1].grid(True,alpha=.25)

for eta,marker in [(0,'o'),(1,'s')]:
    g=dt[dt.eta==eta].sort_values('dt')
    axes[0,2].plot(g.dt,g.l1,marker=marker,label=rf'$\eta={eta}$')
axes[0,2].axhline(0,linewidth=.8);axes[0,2].set_xscale('log');axes[0,2].invert_xaxis()
axes[0,2].set_xlabel(r'RK4 step $\Delta t$');axes[0,2].set_ylabel(r'$\lambda_1$')
axes[0,2].set_title('(c) Timestep refinement');axes[0,2].legend();axes[0,2].grid(True,alpha=.25)

axes[1,0].plot(long.window,long.eta0_max_norm,'o-',label=r'$\eta=0$')
axes[1,0].plot(long.window,long.eta1_max_norm,'s--',label=r'$\eta=1$')
axes[1,0].set_xlabel('successive 1000-time-unit window');axes[1,0].set_ylabel(r'maximum $\|p(t)\|$')
axes[1,0].set_title('(d) Long-time boundedness');axes[1,0].legend();axes[1,0].grid(True,alpha=.25)

axes[1,1].scatter(pc.x2,pc.x3,s=4,alpha=.55,label=rf'$\eta=0$ ({len(pc)} crossings)')
axes[1,1].scatter(pp.x2,pp.x3,s=18,alpha=.8,label=rf'$\eta=1$ ({len(pp)} crossings)')
axes[1,1].set_xlabel(r'$x_2$ on $x_1=0$');axes[1,1].set_ylabel(r'$x_3$ on $x_1=0$')
axes[1,1].set_title('(e) Poincare sections');axes[1,1].legend(fontsize=8);axes[1,1].grid(True,alpha=.25)

# Plot only frequencies <= 2 and decimate for rendering; FFT used all samples.
for df,label in [(f0,r'$\eta=0$'),(f1,r'$\eta=1$')]:
    g=df[df.frequency<=2]
    stride=max(1,len(g)//20000)
    axes[1,2].semilogy(g.frequency.iloc[::stride],g.normalized_power.iloc[::stride]+1e-20,label=label,linewidth=.75)
axes[1,2].set_xlabel('frequency');axes[1,2].set_ylabel('normalized power')
axes[1,2].set_title('(f) Spectral concentration');axes[1,2].legend();axes[1,2].grid(True,alpha=.25)
text=(rf'$H_0={sp.loc[sp.eta==0,"spectral_entropy"].iloc[0]:.4f}$, '
      rf'$H_1={sp.loc[sp.eta==1,"spectral_entropy"].iloc[0]:.4f}$')
axes[1,2].text(.98,.95,text,transform=axes[1,2].transAxes,ha='right',va='top',fontsize=8)

fig.tight_layout()
fig.savefig(FIG/'bracketing_numerical_diagnostics.pdf',bbox_inches='tight')
fig.savefig(FIG/'bracketing_numerical_diagnostics.png',dpi=320,bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# 3. Exact local structure and nonlinear mechanism summary.
# ------------------------------------------------------------------
eq=pd.read_csv(DATA/'equilibria_eta0.csv').to_numpy()
eig=pd.read_csv(DATA/'equilibria_eigenvalues_eta0.csv')
rb=pd.read_csv(DATA/'radial_energy_balance.csv')
cycle=pd.read_csv(DATA/'periodic_cycle_energy_balance.csv')
br=pd.read_csv(DATA/'periodic_orbit_continuation.csv')

fig=plt.figure(figsize=(15,9.2))
gs=GridSpec(2,3,figure=fig,wspace=.28,hspace=.30)

a=fig.add_subplot(gs[0,0],projection='3d')
a.plot(x0[:,0],x0[:,3],x0[:,4],linewidth=.10,alpha=.35,rasterized=True)
a.scatter(eq[:,0],eq[:,3],eq[:,4],s=35,marker='o')
for j,e in enumerate(eq):a.text(e[0],e[3],e[4],f' E{j}',fontsize=7)
a.set_xlabel(r'$x_0$');a.set_ylabel(r'$x_3$');a.set_zlabel(r'$x_4$');a.set_title('(a) Chaotic saddle skeleton')
a.view_init(25,42)

b=fig.add_subplot(gs[0,1])
# Origin eigenvalues are the E0 row, five columns.
r=eig.iloc[0];z=[]
for k in range(1,6):z.append(complex(r[f'eig{k}_real'],r[f'eig{k}_imag']))
b.scatter([q.real for q in z],[q.imag for q in z],s=42)
b.axvline(0,linewidth=.8);b.axhline(0,linewidth=.6);b.set_xlabel(r'$\Re\lambda$');b.set_ylabel(r'$\Im\lambda$')
b.set_title('(b) Parameter-independent origin spectrum');b.grid(True,alpha=.25)

c=fig.add_subplot(gs[0,2])
labels=['linear','left quadratic','active associator'];pos=np.arange(3);width=.34
vals0=[rb.loc[rb.eta==0,'linear_mean'].iloc[0],rb.loc[rb.eta==0,'left_quadratic_mean'].iloc[0],0]
vals1=[rb.loc[rb.eta==1,'linear_mean'].iloc[0],rb.loc[rb.eta==1,'left_quadratic_mean'].iloc[0],rb.loc[rb.eta==1,'active_associator_mean'].iloc[0]]
c.bar(pos-width/2,vals0,width,label=r'$\eta=0$');c.bar(pos+width/2,vals1,width,label=r'$\eta=1$')
c.axhline(0,linewidth=.8);c.set_xticks(pos,labels,rotation=15);c.set_ylabel(r'mean contribution to $\dot N$')
c.set_title('(c) Mean radial balance');c.legend();c.grid(True,axis='y',alpha=.25)

d=fig.add_subplot(gs[1,0])
d.plot(cycle.t,cycle.linear,label='linear',linewidth=.8)
d.plot(cycle.t,cycle.left_quadratic,label='left quadratic',linewidth=.8)
d.plot(cycle.t,cycle.active_associator,label='active associator',linewidth=.8)
d.plot(cycle.t,cycle.total,label='total',linewidth=1.0)
d.set_xlabel('phase time');d.set_ylabel(r'contribution to $\dot N$');d.set_title('(d) One attracting period')
d.legend(fontsize=7);d.grid(True,alpha=.25)

e=fig.add_subplot(gs[1,1])
for stability,g in br.groupby('stability'):
    e.plot(g.eta,g.period,marker='o',markersize=3,linewidth=.9,label=stability)
e.axvline(.50903065,linestyle='--',linewidth=.8);e.set_xlabel(r'$\eta$');e.set_ylabel(r'period $T$')
e.set_title('(e) Saddle-node branch of cycles');e.legend();e.grid(True,alpha=.25)

f=fig.add_subplot(gs[1,2])
for stability,g in br.groupby('stability'):
    f.plot(g.eta,g.max_transverse_multiplier,marker='o',markersize=3,linewidth=.9,label=stability)
f.axhline(1,linestyle='--',linewidth=.8);f.axvline(.50903065,linestyle='--',linewidth=.8)
f.set_xlabel(r'$\eta$');f.set_ylabel('maximum transverse multiplier')
f.set_title('(f) Floquet stability');f.legend();f.grid(True,alpha=.25)

fig.subplots_adjust(left=.06,right=.98,bottom=.07,top=.96,wspace=.28,hspace=.30)
fig.savefig(FIG/'bracketing_associator_bifurcation.pdf',bbox_inches='tight')
fig.savefig(FIG/'bracketing_associator_bifurcation.png',dpi=320,bbox_inches='tight')
plt.close(fig)

print('Generated bracketing-family figures in',FIG)
