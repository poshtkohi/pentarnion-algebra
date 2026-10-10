from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT=Path(__file__).resolve().parents[2]
FIG=Path(os.environ.get('PENTARNION_OUTPUT_DIR', str(ROOT/'outputs/plots'))).expanduser().resolve()
DATA=ROOT/'data/spin'
FIG.mkdir(parents=True, exist_ok=True)
SQ3=math.sqrt(3.0)

def frame(g):
    r=math.sqrt(3+g*g); C=SQ3/r; S=g/r
    h=np.array([C,0,0,S]); n=np.array([-S,0,0,C])
    return C,S,h,n

def moving_P(g):
    C,S,_,_=frame(g)
    return np.array([[C,0,0,-S],[0,1,0,0],[0,0,1,0],[S,0,0,C]],float)

def skew3(u):
    x,y,z=u
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)

def rod(u,th):
    u=np.asarray(u,float); u=u/np.linalg.norm(u); K=skew3(u)
    return math.cos(th)*np.eye(3)+(1-math.cos(th))*np.outer(u,u)+math.sin(th)*K

def R(g,axis,th):
    axes={'h':np.array([1.,0,0]),'b':np.array([0,1.,0]),'c':np.array([0,0,1.])}
    B=np.eye(4); B[:3,:3]=rod(axes[axis],th); P=moving_P(g)
    return P@B@P.T

# 1 Conceptual chain
fig,ax=plt.subplots(figsize=(11,3.2)); ax.axis('off')
labels=['switching\nmultiplication','pure-product\ntensor','alternating\n3-form','tilted plane\n$H_\\gamma$','$SO(3)_\\gamma$\nrotors','$SO(4)$\nswitching','bounded nonlinear\ndynamics']
xs=np.linspace(0.06,0.94,len(labels))
for x,l in zip(xs,labels):
    patch=FancyBboxPatch((x-0.065,0.34),0.13,0.32,boxstyle='round,pad=0.02',linewidth=1.2,facecolor='white')
    ax.add_patch(patch); ax.text(x,0.5,l,ha='center',va='center',fontsize=10)
for x1,x2 in zip(xs[:-1],xs[1:]):
    ax.add_patch(FancyArrowPatch((x1+0.065,0.5),(x2-0.065,0.5),arrowstyle='-|>',mutation_scale=13,linewidth=1.2))
ax.set_title('One multiplication law, one geometric chain, one dynamical mechanism',fontsize=15,pad=10)
fig.tight_layout(); fig.savefig(FIG/'conceptual_chain.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# 2 Quaternion geometry schematic
fig=plt.figure(figsize=(7.2,6.5)); ax=fig.add_subplot(111,projection='3d')
u=np.array([0,0,1.]); v=np.array([1.,0,0]); th=math.radians(60); vr=np.array([math.cos(th),math.sin(th),0])
ax.quiver(0,0,0,*u,length=1.0,arrow_length_ratio=.12,label='axis $u$')
ax.quiver(0,0,0,*v,length=1.0,arrow_length_ratio=.12,label='initial $v$')
ax.quiver(0,0,0,*vr,length=1.0,arrow_length_ratio=.12,label="rotated $v'$")
phi=np.linspace(0,th,100); ax.plot(np.cos(phi),np.sin(phi),np.zeros_like(phi),linewidth=2)
ax.set_xlim(-.2,1.1); ax.set_ylim(-.2,1.1); ax.set_zlim(0,1.1); ax.set_box_aspect((1,1,1)); ax.view_init(24,-55)
ax.set_xlabel('$i$'); ax.set_ylabel('$j$'); ax.set_zlabel('$k$'); ax.set_title('Quaternion rotation: fixed axis, preserved length, rotated perpendicular plane'); ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig(FIG/'quaternion_rotation_geometry.png',dpi=220); plt.close(fig)

# 3 Tilted circles on sphere in (a,c,d)
fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
# sphere wireframe
uu=np.linspace(0,2*np.pi,40); vv=np.linspace(0,np.pi,22)
X=np.outer(np.cos(uu),np.sin(vv)); Y=np.outer(np.sin(uu),np.sin(vv)); Z=np.outer(np.ones_like(uu),np.cos(vv))
ax.plot_wireframe(X,Y,Z,rstride=4,cstride=3,linewidth=.35,alpha=.28)
th=np.linspace(0,2*np.pi,500)
for g in [0.,.5,1.]:
    C,S,_,_=frame(g)
    a=C*np.sin(th); c=np.cos(th); d=S*np.sin(th)
    ax.plot(a,c,d,linewidth=2,label=f'$\\gamma={g:g}$, $\\beta={math.degrees(math.atan(g/SQ3)):.1f}^\\circ$')
# h vectors
for g in [0.,1.]:
    C,S,_,_=frame(g); ax.quiver(0,0,0,C,0,S,length=1,arrow_length_ratio=.09)
ax.set_xlabel('$a$'); ax.set_ylabel('$c$'); ax.set_zlabel('$d=\\ell$'); ax.set_title('The tilted rotation shape: unit rotation circles leave the quaternionic plane')
ax.set_box_aspect((1,1,1)); ax.view_init(23,-52); ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig(FIG/'tilted_rotation_circles_on_sphere.png',dpi=240); plt.close(fig)

# 4 plane disks cross-section
fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
r=np.linspace(0,1,35); th=np.linspace(0,2*np.pi,100); rr,tt=np.meshgrid(r,th)
for g in [0.,.5,1.]:
    C,S,_,_=frame(g); A=rr*C*np.sin(tt); Cc=rr*np.cos(tt); D=rr*S*np.sin(tt)
    ax.plot_surface(A,Cc,D,alpha=.18,linewidth=0)
    ax.plot(C*np.sin(th),np.cos(th),S*np.sin(th),linewidth=2,label=f'$\\gamma={g:g}$')
ax.set_xlabel('$a$'); ax.set_ylabel('$c$'); ax.set_zlabel('$d=\\ell$'); ax.set_title('Cross-sections of the moving rotational three-plane about the fixed axis $b$')
ax.set_box_aspect((1,1,1)); ax.view_init(22,-52); ax.legend()
fig.tight_layout(); fig.savefig(FIG/'tilted_rotation_plane_disks.png',dpi=240); plt.close(fig)

# 5 raw sandwich defect for example gamma=1,u=i,v=j
th=np.linspace(0,2*np.pi,500)
# derive pure norm numerically by exact formula for this example using basis multiplication is unnecessary: use direct symbolic expression at pi/2 and numerical table via small implementation
# Implement original-basis pure cross matrix using multiplication table.
def mul(g,p,q):
    # p,q in [scalar,i,j,k,l]
    k=1/math.sqrt(1+g*g); s=g/math.sqrt(1+g*g); rt2=math.sqrt(2)
    table={}
    e=[np.eye(5)[i] for i in range(5)]
    # bilinear basis products
    for j in range(5): table[(0,j)]=e[j]; table[(j,0)]=e[j]
    for j in range(1,5): table[(j,j)]=-e[0]
    def V(i=0,j=0,kc=0,l=0): return np.array([0.,i,j,kc,l])
    table[(1,2)]=V(kc=k,l=s); table[(2,1)]=-table[(1,2)]
    table[(2,3)]=V(i=k,l=s); table[(3,2)]=-table[(2,3)]
    table[(3,1)]=V(j=k,l=s); table[(1,3)]=-table[(3,1)]
    table[(1,4)]=V(j=-1/rt2,kc=-1/rt2); table[(4,1)]=-table[(1,4)]
    table[(2,4)]=V(i=-1/rt2,kc=-1/rt2); table[(4,2)]=-table[(2,4)]
    table[(3,4)]=V(i=-1/rt2,j=-1/rt2); table[(4,3)]=-table[(3,4)]
    out=np.zeros(5)
    for a in range(5):
        for b in range(5): out+=p[a]*q[b]*table[(a,b)]
    return out
norms=[]; scalars=[]
v=np.array([0,0,1,0,0.],float)
for angle in th:
    r0=np.array([math.cos(angle/2),math.sin(angle/2),0,0,0.]); rb=np.array([math.cos(angle/2),-math.sin(angle/2),0,0,0.])
    out=mul(1.0,mul(1.0,r0,v),rb); norms.append(np.linalg.norm(out[1:])); scalars.append(out[0])
fig,ax=plt.subplots(figsize=(8.2,4.8)); ax.plot(np.degrees(th),norms,label='pure-part norm'); ax.axhline(1,linestyle='--',label='unit norm'); ax.set_xlabel('nominal angle $\\theta$ (degrees)'); ax.set_ylabel('norm'); ax.set_title('Raw unit Pentarnion sandwich is not a pure rotation ($u=i$, $v=j$, $\\gamma=1$)'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(FIG/'raw_sandwich_norm_defect.png',dpi=220); plt.close(fig)

# 6 tensor rotational/deformation norms
g=np.linspace(0,1,400); rot=2*(g*g+3)/(1+g*g); deform=2*(5*g*g+3)/(1+g*g)
fig,ax=plt.subplots(figsize=(8.2,4.8)); ax.plot(g,rot,label='$\\|\\operatorname{Alt}T_\\gamma\\|_F^2$'); ax.plot(g,deform,label='$\\|T_\\gamma-\\operatorname{Alt}T_\\gamma\\|_F^2$'); ax.plot(g,rot+deform,linestyle='--',label='$\\|T_\\gamma\\|_F^2=12$'); ax.set_xlabel('$\\gamma$'); ax.set_ylabel('squared Frobenius norm'); ax.set_title('Rotational and nonalternating parts of the pure multiplication tensor'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(FIG/'tensor_rotation_deformation_norms.png',dpi=220); plt.close(fig)

# 7 holonomy phase
eps=np.linspace(0,np.pi/2,300); phi=4*np.arcsin(.5*np.sin(eps/2)**2); approx=.5*eps**2
fig,ax=plt.subplots(figsize=(8.2,4.8)); ax.plot(np.degrees(eps),np.degrees(phi),label='exact switching holonomy'); ax.plot(np.degrees(eps),np.degrees(approx),linestyle='--',label='small-loop approximation'); ax.set_xlabel('elementary angle $\\varepsilon$ (degrees)'); ax.set_ylabel('residual angle $\\Phi$ (degrees)'); ax.set_title('Exact holonomy for the full switch $\\gamma:0\\leftrightarrow1$'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(FIG/'switching_holonomy_phase.png',dpi=220); plt.close(fig)

# 8 holonomy control-loop path in (a,b,d)
def cycle_path(eps=math.radians(45),n=80):
    x=np.array([1.,0,0,0]); pts=[]
    controls=[(0,'c',eps),(1,'c',eps),(0,'c',-eps),(1,'c',-eps)]
    for g,axis,ang in controls:
        for tau in np.linspace(0,1,n,endpoint=False): pts.append(R(g,axis,tau*ang)@x)
        x=R(g,axis,ang)@x
    pts.append(x); return np.array(pts),x
pts,xf=cycle_path()
fig=plt.figure(figsize=(7.5,6.7)); ax=fig.add_subplot(111,projection='3d'); ax.plot(pts[:,0],pts[:,1],pts[:,3],linewidth=2); ax.scatter([1],[0],[0],s=45,label='start'); ax.scatter([xf[0]],[xf[1]],[xf[3]],s=45,label='end'); ax.set_xlabel('$a$'); ax.set_ylabel('$b$'); ax.set_zlabel('$d=\\ell$'); ax.set_title('Four-step algebra-switching loop leaves a residual orientation'); ax.legend(); ax.set_box_aspect((1,1,.7)); ax.view_init(25,-55)
fig.tight_layout(); fig.savefig(FIG/'switching_holonomy_loop.png',dpi=230); plt.close(fig)

# 9 double rotation torus projection
Q=R(1,'c',math.pi/2)@R(0,'b',math.pi/2); x=np.array([.5,.5,.5,.5]); orb=[]
for _ in range(1600): orb.append(x.copy()); x=Q@x
orb=np.array(orb)
fig=plt.figure(figsize=(7.4,6.7)); ax=fig.add_subplot(111,projection='3d'); ax.scatter(orb[:,0],orb[:,1],orb[:,2],s=4); ax.set_xlabel('$a$'); ax.set_ylabel('$b$'); ax.set_zlabel('$c$'); ax.set_title('Projection of a switching-induced double-rotation orbit on $S^3$'); ax.set_box_aspect((1,1,1)); ax.view_init(24,-55)
fig.tight_layout(); fig.savefig(FIG/'double_rotation_torus_projection.png',dpi=230); plt.close(fig)

# 10 anisotropy
g=np.linspace(0,1,400); A=3*(g**4+2*g*g+3)/((1+g*g)*(3+g*g)); B=(4*g*g+3)/(1+g*g)
fig,ax=plt.subplots(figsize=(8.2,4.8)); ax.plot(g,A,label='$A_\\gamma$ along $h_\\gamma$'); ax.plot(g,B,label='$B_\\gamma$ along $b,c$'); ax.set_xlabel('$\\gamma$'); ax.set_ylabel('multiplication-induced coefficient'); ax.set_title('Quaternionic isotropy becomes Pentarnion uniaxial anisotropy'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(FIG/'multiplication_induced_anisotropy.png',dpi=220); plt.close(fig)

# Principal numerical figures from supplied CSVs
for regime,om in [('regular',.30),('chaotic',1.20)]:
    d=pd.read_csv(DATA/f'{regime}_timeseries.csv')
    # discard early transient for clearer figure
    cut=60 if regime=='regular' else 100
    ds=d[d.t>=cut]
    fig,ax=plt.subplots(figsize=(10,5.2)); ax.plot(ds.t,ds.y_h,label='$y_h$'); ax.plot(ds.t,ds.y_b,label='$y_b$'); ax.plot(ds.t,ds.y_c,label='$y_c$'); ax.plot(ds.t,ds.energy,label='$H_{\\gamma(t)}$',linewidth=1.6); ax.set_xlabel('time'); ax.set_ylabel('state component / energy'); ax.set_title(f'{regime.capitalize()} algebra-native rotation: time series ($\\Omega={om:.2f}$)'); ax.legend(ncol=4,loc='upper center'); ax.grid(True)
    fig.tight_layout(); fig.savefig(FIG/f'{regime}_timeseries.png',dpi=240); plt.close(fig)
    fig=plt.figure(figsize=(7.5,6.8)); ax=fig.add_subplot(111,projection='3d'); ax.plot(ds.y_h,ds.y_b,ds.y_c,linewidth=.7); ax.set_xlabel('$y_h$'); ax.set_ylabel('$y_b$'); ax.set_zlabel('$y_c$'); ax.set_title(f'{regime.capitalize()} moving-frame trajectory ($\\Omega={om:.2f}$)'); ax.set_box_aspect((1,1,1)); ax.view_init(25,-58)
    fig.tight_layout(); fig.savefig(FIG/f'{regime}_trajectory.png',dpi=240); plt.close(fig)
    p=pd.read_csv(DATA/f'{regime}_poincare.csv')
    fig,ax=plt.subplots(figsize=(6.5,6.2)); ax.scatter(p.y_h,p.y_b,s=7); ax.set_xlabel('stroboscopic $y_h$'); ax.set_ylabel('stroboscopic $y_b$'); ax.set_title(f'{regime.capitalize()} Poincare section ($\\Omega={om:.2f}$)'); ax.grid(True)
    fig.tight_layout(); fig.savefig(FIG/f'{regime}_poincare.png',dpi=230); plt.close(fig)

# 14 Lyapunov scan from high-accuracy prior verified values supplemented by scan
scan=pd.read_csv(DATA/'representative_lyapunov_scan.csv')
fig,ax=plt.subplots(figsize=(8.6,4.9)); ax.plot(scan.Omega,scan.lambda_max,marker='o'); ax.axhline(.005,linestyle='--',label='regular threshold'); ax.axhline(.015,linestyle='--',label='chaotic threshold'); ax.set_xlabel('switching frequency $\\Omega$'); ax.set_ylabel('largest Lyapunov exponent'); ax.set_title('Regular-chaotic-regular response of the multiplication-induced spin'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(FIG/'lyapunov_frequency_scan.png',dpi=230); plt.close(fig)

# 15 energy balance
b=pd.read_csv(DATA/'energy_balance_chaotic.csv')
fig,ax=plt.subplots(figsize=(8.5,4.9)); ax.plot(b.t,b.energy_change,label='$H(t)-H(0)$'); ax.plot(b.t,b.integrated_power,label='$\\int_0^t \\partial_\\gamma H\\,\\dot\\gamma\\,ds$'); ax.set_xlabel('time'); ax.set_ylabel('energy change'); ax.set_title('The dynamic energy change equals the accumulated algebra-switching work'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(FIG/'energy_balance_verification.png',dpi=230); plt.close(fig)

# 16 lab-coordinate chaotic trajectory derived from moving coords
c=pd.read_csv(DATA/'chaotic_timeseries.csv'); c=c[c.t>=100].copy(); gam=c.gamma.to_numpy(); C=SQ3/np.sqrt(3+gam*gam); S=gam/np.sqrt(3+gam*gam); a=C*c.y_h.to_numpy(); ell=S*c.y_h.to_numpy()
fig=plt.figure(figsize=(7.4,6.7)); ax=fig.add_subplot(111,projection='3d'); ax.plot(a,c.y_b,ell,linewidth=.65); ax.set_xlabel('$a$'); ax.set_ylabel('$b$'); ax.set_zlabel('$\\ell$'); ax.set_title('Chaotic trajectory in the fixed four-dimensional pure-space projection'); ax.set_box_aspect((1,1,1)); ax.view_init(25,-58)
fig.tight_layout(); fig.savefig(FIG/'chaotic_lab_projection.png',dpi=230); plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,4.7)); ax.plot(c.t,ell); ax.set_xlabel('time'); ax.set_ylabel('$\\ell$ coordinate'); ax.set_title('Dynamic activation of the fourth pure direction in the chaotic regime'); ax.grid(True)
fig.tight_layout(); fig.savefig(FIG/'chaotic_ell_timeseries.png',dpi=230); plt.close(fig)

print('generated',len(list(FIG.glob('*.png'))),'figures')
