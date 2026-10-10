from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from matplotlib.ticker import MaxNLocator
import subprocess
import sys

# Section 9 revision: use explicit data provenance and readable individual plots.
plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11})

ROOT=Path(__file__).resolve().parents[2]
FIG = Path(os.environ.get('PENTARNION_OUTPUT_DIR', str(ROOT/'outputs/plots'))).expanduser().resolve()
FIG.mkdir(parents=True, exist_ok=True)
DENSE=ROOT/'data'/'riccati_dense'
BASE=ROOT/'data'/'riccati'
BR=BASE/'periodic_branches'

def save(fig,name):
    w,h=fig.get_size_inches()
    width=8.6 if 'timeseries' in name else (7.0 if 'atlas' in name else 6.6)
    fig.set_size_inches(width, max(4.2,min(5.5,width*h/w)))
    for axis in fig.axes:
        title=axis.get_title()
        if len(title)>58:
            axis.set_title(textwrap.fill(title, width=58, break_long_words=False, break_on_hyphens=False))
        legend=axis.get_legend()
        if legend:
            for item in legend.get_texts(): item.set_fontsize(10.5)
    extra=[]
    has3d=any(hasattr(axis,'zaxis') for axis in fig.axes)
    if has3d:
        fig.subplots_adjust(left=.04,right=.86,bottom=.10,top=.89)
        for axis in fig.axes:
            if hasattr(axis,'zaxis'):
                axis.set_box_aspect((1,1,1),zoom=.88)
                for xyzaxis in (axis.xaxis,axis.yaxis,axis.zaxis):
                    xyzaxis.set_major_locator(MaxNLocator(4))
                    extra.append(xyzaxis.label)
    else:
        fig.tight_layout()
    fig.savefig(FIG/name,dpi=260,bbox_inches='tight',bbox_extra_artists=extra or None,pad_inches=.14)
    plt.close(fig)


def select_full_horizon(df, max_points=120000):
    """Retain the complete post-transient time horizon in temporal order.

    Uniform decimation is used only when required for a manageable rasterised
    line figure.  It preserves the first sample, final sample, and full time
    span without interpolation or geometric smoothing.
    """
    q = df.copy()
    if len(q) > max_points:
        idx = np.linspace(0, len(q) - 1, max_points, dtype=int)
        q = q.iloc[np.unique(idx)].copy()
    return q.reset_index(drop=True)



def add_connected_trajectory(ax, df, cols, linewidth=.08, alpha=.48):
    """Draw one continuous time-ordered trajectory over the full retained horizon.

    The trajectory is rendered as a line, not as a scatter cloud.  No
    interpolation or geometric smoothing is applied.
    """
    xyz = df.loc[:, cols].to_numpy(float)
    ax.plot(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        linewidth=linewidth,
        alpha=alpha,
        solid_capstyle='round',
        solid_joinstyle='round',
        rasterized=True,
    )
    # Retain the full-horizon line while highlighting one contiguous terminal
    # arc, making the time-ordered path explicit without scatter rendering.
    n_highlight = min(5000, max(1200, len(xyz) // 20))
    h = xyz[-n_highlight:]
    ax.plot(
        h[:, 0], h[:, 1], h[:, 2],
        linewidth=max(.34, 4.0 * linewidth),
        alpha=.96,
        solid_capstyle='round',
        solid_joinstyle='round',
        rasterized=True,
        zorder=4,
    )
    for values, setter in zip(xyz.T, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        pad = .04 * max(hi - lo, 1e-9)
        setter(lo - pad, hi + pad)
    return float(df['t'].iloc[-1] - df['t'].iloc[0])

# Distinguish the exact origin-instability condition from archived flip interpolants.
hopf=0.36874617836683965
EVENTS=pd.read_csv(BR/'primary_period_doubling_cascade.csv')
G=EVENTS['gamma'].to_numpy(float)
# Use one categorical row per event.  The late period doublings are extremely
# close in gamma, so a conventional single-line timeline makes their labels
# collide even at full text width.  This layout retains the true horizontal
# parameter positions while separating the labels vertically.
labels=[r'origin Hopf $\gamma_H$',r'$PD_1$',r'$PD_2$',r'$PD_3$',r'$PD_4$',r'$PD_5$',r'$PD_6$']
values=np.r_[hopf,G]
y=np.arange(len(values))[::-1]
fig,ax=plt.subplots(figsize=(11.4,5.0))
ax.hlines(y,0.36,values,linewidth=1.0,alpha=.65)
ax.scatter(values,y,s=[48]+[36]*6,zorder=3)
for x,yy in zip(values,y):
    ax.annotate(f'{x:.8f}',xy=(x,yy),xytext=(7,0),textcoords='offset points',
                ha='left',va='center',fontsize=9.5)
ax.set_xlim(.36,.505); ax.set_ylim(-.65,len(values)-.35)
ax.set_yticks(y,labels); ax.set_xlabel(r'static switching parameter $\gamma$')
ax.set_title('Resolved dynamical milestones of the static Riccati family')
ax.grid(axis='x',alpha=.35)
ax.spines[['right','top']].set_visible(False)
save(fig,'riccati_event_timeline.png')

# Static Lyapunov and bifurcation scans.
scan=pd.read_csv(BASE/'static_scan.csv')
fig,ax=plt.subplots(figsize=(9.4,5.2)); ax.plot(scan['gamma'],scan['lle'],linewidth=1.0); ax.axhline(0,linewidth=.8); ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'largest finite-time estimate $\lambda_{\max}^{\mathrm{FT}}$'); ax.set_title(r'Largest finite-time Lyapunov exponent versus $\gamma$'); ax.grid(True); save(fig,'riccati_static_largest_lyapunov.png')
peaks=pd.read_csv(BASE/'static_bifurcation_peaks.csv')
fig,ax=plt.subplots(figsize=(10.0,5.6)); ax.scatter(peaks['gamma'],peaks['z_max'],s=.35,alpha=.45,rasterized=True); ax.set_xlabel(r'$\gamma$'); ax.set_ylabel(r'successive maxima of $p_3$'); ax.set_title('Static switching bifurcation diagram'); ax.grid(True); save(fig,'riccati_static_bifurcation_diagram.png')

# Static records.  The trajectory panels use the complete post-transient
# records as continuous time-ordered curves.  The component panels keep the
# previously selected late-time intervals so their oscillatory structure stays
# readable.  Long Poincare sections are retained.
for tag, g, kind, linewidth, alpha in [
    ('gamma_0p60', .60, 'chaotic', .15, .38),
    ('gamma_0p70', .70, 'regular', .28, .72),
]:
    d = pd.read_csv(DENSE / f'static_{tag}_timeseries.csv')
    psec = pd.read_csv(DENSE / f'static_{tag}_poincare.csv')
    trajectory = select_full_horizon(d, max_points=120000)
    q = d[d['t'] >= d['t'].iloc[-1] - 60].copy()
    q['tau'] = q['t'] - q['t'].iloc[0]

    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection='3d')
    add_connected_trajectory(ax, trajectory, ['w', 'z', 'u'], linewidth=linewidth, alpha=alpha)
    ax.set_xlabel(r'$p_0$')
    ax.set_ylabel(r'$p_3$')
    ax.set_zlabel(r'$p_4$')
    ax.set_title(fr'Static $\gamma={g:.1f}$: projected trajectory')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)
    save(fig, f'riccati_static_{kind}_trajectory_dense.png')

    fig, ax = plt.subplots(figsize=(7.3, 6.1))
    ax.scatter(psec['y'], psec['z'], s=1.7 if kind == 'chaotic' else 7,
               alpha=.65, rasterized=True)
    ax.set_xlabel(r'$p_2$ at upward $p_1=0$')
    ax.set_ylabel(r'$p_3$')
    ax.set_title(fr'Static $\gamma={g:.1f}$: Poincaré section')
    ax.grid(True)
    save(fig, f'riccati_static_{kind}_poincare_dense.png')

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for col, lab in [('w', r'$p_0$'), ('x', r'$p_1$'), ('y', r'$p_2$'),
                     ('z', r'$p_3$'), ('u', r'$p_4$')]:
        ax.plot(q['tau'], q[col], linewidth=.72, label=lab)
    ax.set_xlabel('elapsed time in displayed interval')
    ax.set_ylabel('component')
    ax.set_title(fr'Static $\gamma={g:.1f}$: representative component signal')
    ax.legend(ncol=5, fontsize=9)
    ax.grid(True)
    save(fig, f'riccati_static_{kind}_timeseries_dense.png')

# Periodic branches and Floquet diagnostics.
names=['P1','P2','P4','P8','P16','P32','Broad_P2','Window_C1','Window_C2','Window_C4','Narrow_P4']
# Scientific figure labels; these do not rename the internal data keys.
DISPLAY_LABELS = {
    'P1': r'$P_1$', 'P2': r'$P_2$', 'P4': r'$P_4$',
    'P8': r'$P_8$', 'P16': r'$P_{16}$', 'P32': r'$P_{32}$',
    'Broad_P2': 'Broad-window branch',
    'Window_C1': 'Secondary branch',
    'Window_C2': 'First doubled branch',
    'Window_C4': 'Second doubled branch',
    'Narrow_P4': 'Narrow-window branch',
}
B={n:pd.read_csv(BR/f'{n}_floquet.csv') for n in names}
periods=EVENTS['parent_period'].to_numpy(float)
fig,ax=plt.subplots(figsize=(9.5,5.8))
for n in ['P1','P2','P4','P8','P16','P32']:
    q=B[n].sort_values('gamma'); q=q[q['stable']==1]; ax.plot(q['gamma'],q['T'],marker='.',label=DISPLAY_LABELS[n])
ax.scatter(G,periods,s=35,label='estimated flip parameters'); ax.set_yscale('log'); ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('period'); ax.set_title('Primary periodic branches and period doublings'); ax.legend(); ax.grid(True); save(fig,'riccati_primary_branch_periods.png')

def critical(df,target=-1):
    vals=[]
    for _,r in df.iterrows():
        rr=[]
        for j in range(1,5):
            z=complex(r[f'mu{j}_real'],r[f'mu{j}_imag'])
            if abs(z.imag)<1e-6: rr.append(z.real)
        vals.append(min(rr,key=lambda x:abs(x-target)) if rr else np.nan)
    return np.array(vals)
birth=np.r_[hopf,G[:-1]]
fig,ax=plt.subplots(figsize=(9.5,5.6))
for k,n in enumerate(['P1','P2','P4','P8','P16','P32']):
    d=B[n].sort_values('gamma'); lo,hi=birth[k],G[k]; q=d[(d.gamma>=lo+.35*(hi-lo))&(d.gamma<=hi+.02*(hi-lo))]; ax.scatter((q.gamma-lo)/(hi-lo),critical(q),s=14,label=DISPLAY_LABELS[n])
ax.axhline(-1,linestyle='--',label='flip threshold'); ax.set_xlabel(r'$(\gamma-\gamma_b)/(\gamma_f-\gamma_b)$'); ax.set_ylabel('real multiplier nearest $-1$'); ax.set_title('Real transverse multipliers near six flips'); ax.legend(); ax.grid(True); save(fig,'riccati_primary_floquet_flip_sequence.png')
intervals=np.diff(G); ratios=intervals[:-1]/intervals[1:]; idx=np.arange(3,3+len(ratios))
fig,ax=plt.subplots(figsize=(8.5,5.1)); ax.plot(idx,ratios,marker='o',label='measured ratios'); ax.axhline(4.66920160910299,linestyle='--',label='Feigenbaum value'); ax.set_xticks(idx); ax.set_xlabel('cascade ratio index'); ax.set_ylabel(r'$\Delta\gamma_{n-1}/\Delta\gamma_n$'); ax.set_title('Scaling of the primary period-doubling cascade'); ax.legend(); ax.grid(True); save(fig,'riccati_primary_cascade_scaling.png')
fig=plt.figure(figsize=(9,7)); ax=fig.add_subplot(111,projection='3d')
for n in ['P1','P2','P4','P8','P16','P32']:
    o=pd.read_csv(BR/f'orbit_{n}.csv'); ax.plot(o['w'],o['z'],o['u'],linewidth=.8,label=DISPLAY_LABELS[n])
ax.set_xlabel(r'$p_0$'); ax.set_ylabel(r'$p_3$'); ax.set_zlabel(r'$p_4$'); ax.set_title('Growth of the primary periodic orbits'); ax.legend(loc='upper left',ncol=2,fontsize=10); save(fig,'riccati_primary_orbit_growth.png')
d=B['Broad_P2'].sort_values('gamma'); fig,ax=plt.subplots(figsize=(9.4,5.4)); ax.semilogy(d.gamma,d.max_transverse_modulus); ax.axhline(1,linestyle='--',label='stability boundary');
for x,l in [(0.48725,'lower endpoint candidate'),(0.4876774177,'flip'),(0.6692803628,'inverse flip'),(0.72955,'upper endpoint candidate')]: ax.axvline(x,linestyle=':',label=l)
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('largest transverse multiplier modulus'); ax.set_title('Transverse stability of the broad-window branch'); ax.legend(fontsize=8); ax.grid(True); save(fig,'riccati_broad_P2_stability.png')
WG=np.array([0.7788714237819837,0.7930853213196591,0.795397739979263]); WP=np.array([5.67013081656135,11.363451119305696,22.733081298347777])
fig,ax=plt.subplots(figsize=(9.2,5.5))
for n in ['Window_C1','Window_C2','Window_C4']:
    q=B[n].sort_values('gamma');q=q[q.stable==1];ax.plot(q.gamma,q['T'],marker='.',label=DISPLAY_LABELS[n])
ax.scatter(WG,WP,s=35,label='estimated flip parameters'); ax.set_yscale('log'); ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('period'); ax.set_title('Secondary cascade inside the broad regular window'); ax.legend(); ax.grid(True); save(fig,'riccati_regular_window_cascade.png')
d=B['Narrow_P4'].sort_values('gamma'); fig,ax=plt.subplots(figsize=(9.2,5.3)); ax.plot(d.gamma,d.max_transverse_modulus); ax.axhline(1,linestyle='--',label='stability boundary'); ax.axvline(.8531575,linestyle=':',label='fold candidate'); ax.axvline(.8555959746,linestyle=':',label='flip'); ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('largest transverse multiplier modulus'); ax.set_title(r'Narrow periodic window near $\gamma=0.855$'); ax.legend(); ax.grid(True); save(fig,'riccati_narrow_window_stability.png')
atlas=[(r'Primary $P_1$',hopf,G[0]),(r'Primary $P_2$',G[0],G[1]),(r'Primary $P_4$',G[1],G[2]),(r'Primary $P_8$',G[2],G[3]),(r'Primary $P_{16}$',G[3],G[4]),(r'Primary $P_{32}$',G[4],G[5]),('Broad window: lower',.48726,.4876774177),('Broad window: upper',.6692803628,.72954),('Secondary branch',.72733,WG[0]),('Secondary: first doubling',WG[0],WG[1]),('Secondary: second doubling',WG[1],WG[2]),('Narrow window',.853158,.8555959746)]
fig,ax=plt.subplots(figsize=(10,6.1))
for i,(lab,lo,hi) in enumerate(atlas): ax.plot([lo,hi],[i,i],linewidth=5,solid_capstyle='round'); ax.scatter([lo,hi],[i,i],s=18)
ax.set_yticks(np.arange(len(atlas)),[a[0] for a in atlas]); ax.invert_yaxis(); ax.set_xlabel(r'$\gamma$'); ax.set_title('Stability intervals of periodic branches'); ax.grid(True,axis='x'); save(fig,'riccati_periodic_branch_atlas.png')

# Both time-dependent switching protocols.
I=pd.read_csv(ROOT/'data'/'time_switching'/'time_scan_transition_band.csv'); II=pd.read_csv(ROOT/'data'/'time_switching'/'time_scan_full_range_regularized.csv')
fig,ax=plt.subplots(figsize=(9.5,5.4)); ax.plot(I.omega,I.lle,marker='o',label=r'Protocol I: $0.68+0.08\sin(\omega t)$'); valid=II.blowup==0; ax.plot(II.loc[valid,'omega'],II.loc[valid,'lle'],marker='o',label=r'Protocol II: $0.525+0.475\sin(\omega t)$'); esc=II[II.blowup==1];
# Failed entries are deliberately omitted, not plotted at zero.
# The caption identifies the omitted row; no placeholder is placed in data space.
ax.axhline(0,linewidth=.8); ax.set_xlabel(r'$\omega$'); ax.set_ylabel(r'largest finite-time estimate $\lambda_{\max}^{\mathrm{FT}}$'); ax.set_title('Finite-time Lyapunov exponents versus forcing frequency'); ax.legend(fontsize=9); ax.grid(True); save(fig,'both_protocols_frequency_scan.png')

cases = [
    ('protocol_I_omega_0p2', 'protocol_band_omega_0p2_timeseries.png', r'Protocol I, $\omega=0.2$'),
    ('protocol_I_omega_0p8', 'protocol_band_omega_0p8_timeseries.png', r'Protocol I, $\omega=0.8$'),
    ('protocol_II_omega_0p2', 'protocol_full_omega_0p2_timeseries.png', r'Protocol II, $\omega=0.2$'),
    ('protocol_II_omega_0p8', 'protocol_full_omega_0p8_timeseries.png', r'Protocol II, $\omega=0.8$'),
]
for stem, name, title in cases:
    d = pd.read_csv(DENSE / f'{stem}_timeseries.csv')
    q = d[d.t >= d.t.iloc[-1] - 60].copy()
    q['tau'] = q.t - q.t.iloc[0]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    for col, lab in [('w', r'$p_0$'), ('x', r'$p_1$'), ('y', r'$p_2$'),
                     ('z', r'$p_3$'), ('u_active', r'$p_4$')]:
        ax.plot(q.tau, q[col], linewidth=.72, label=lab)
    ax.set_xlabel('elapsed time in displayed interval')
    ax.set_ylabel('state component')
    ax.set_title(title)
    forcing=ax.twinx()
    forcing.plot(q.tau,q['gamma'],linestyle='--',linewidth=1.1,label=r'$\gamma(t)$')
    forcing.set_ylabel(r'$\gamma(t)$')
    forcing.set_ylim(0,1.03)
    handles,labels=ax.get_legend_handles_labels()
    fh,fl=forcing.get_legend_handles_labels()
    ax.legend(handles+fh,labels+fl,ncol=3,fontsize=10,loc='upper right')
    ax.grid(True)
    save(fig, name)

# The representative protocol trajectories use the complete long retained
# records and are drawn as single connected lines.  No point-cloud rendering,
# interpolation, or geometric smoothing is used.
for stem, name, title in [
    ('protocol_I_omega_0p8_dense', 'protocol_transition_band_dense.png',
     r'Protocol I, $\omega=0.8$: projected trajectory'),
    ('protocol_II_omega_0p2_dense', 'protocol_near_full_dense.png',
     r'Protocol II, $\omega=0.2$: intermittent bursts and returns'),
]:
    d = pd.read_csv(DENSE / f'{stem}_timeseries.csv')
    trajectory = select_full_horizon(d, max_points=120000)
    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection='3d')
    add_connected_trajectory(ax, trajectory, ['w', 'z', 'u_active'], linewidth=.15, alpha=.38)
    ax.set_xlabel(r'$p_0$')
    ax.set_ylabel(r'$p_3$')
    ax.set_zlabel(r'$p_4$')
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)
    save(fig, name)

print('wrote full-horizon connected Riccati/protocol trajectories and representative time-series panels')

# Additional figures use only the accompanying, independently generated checks.
subprocess.run([sys.executable, str(Path(__file__).parent/'section9_validation/make_validation_figures.py')], check=True)
