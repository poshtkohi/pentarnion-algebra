#!/usr/bin/env python3
"""Reconstruct the three Part I bracket-geometry illustrations from the equations.

No reference image is used as input. Numerical assertions test the parametrised
ellipse, its semi-axes and the left/right determinant zero sets before plotting.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
STUDY=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(STUDY))
from shared.coordinate_product import product


def maps(p: np.ndarray, gamma: float) -> list[np.ndarray]:
    i=np.array([0.,1.,0.,0.,0.])
    ip=product(i,p,gamma);pi=product(p,i,gamma);pp=product(p,p,gamma)
    return [product(i,pp,gamma),product(pp,i,gamma),product(ip,p,gamma),
            product(p,ip,gamma),product(pi,p,gamma),product(p,pi,gamma)]


def ellipse(theta: np.ndarray, gamma: float) -> np.ndarray:
    c=1/np.sqrt(1+gamma**2);s=gamma*c;r=np.sqrt(2.)
    z=np.array([-c*c,-s/(2*r),s/(2*r),-s*c])
    a=np.array([s/r,s/(2*r),s/(2*r),0.])
    b=np.array([0.,s/(2*r),-s/(2*r),0.])
    return z+np.cos(2*theta)[:,None]*a+np.sin(2*theta)[:,None]*b


def zero_curve(theta: np.ndarray, side: str='left') -> np.ndarray:
    sign=1 if side=='left' else -1
    y=np.cos(theta);z=np.sin(theta)
    w=np.array([brentq(lambda w:2*w**3+w+sign*np.sqrt(2.)*(yy**3+zz**3),-2,2,xtol=1e-14) for yy,zz in zip(y,z)])
    p=np.column_stack((w,y,z));p/=np.linalg.norm(p,axis=1)[:,None]
    return p


def checks() -> dict:
    theta=np.linspace(0,2*np.pi,65);basis=np.eye(5)
    error=0.
    for g in (0.,.25,.5,.75,1.):
        pp=np.array([[0.,0.,np.cos(t),np.sin(t),0.] for t in theta])
        calculated=np.array([maps(p,g)[2][1:] for p in pp])
        error=max(error,float(np.max(np.abs(calculated-ellipse(theta,g)))))
        for p in pp:
            q=maps(p,g)
            for a,b in [(q[0],-basis[1]),(q[1],-basis[1]),(q[2],q[5]),(q[3],q[4]),(q[3],-q[2])]:
                np.testing.assert_allclose(a,b,atol=3e-14,rtol=0)
    residual=0.
    for side in ('left','right'):
        curve=zero_curve(theta,side)
        np.testing.assert_allclose(np.linalg.norm(curve,axis=1),1,atol=1e-14)
        for w,y,z in curve:
            p=np.array([w,0.,y,z,0.])
            M=np.column_stack([product(p,e,1.) if side=='left' else product(e,p,1.) for e in basis])
            residual=max(residual,abs(float(np.linalg.det(M))))
    assert error<1e-13 and residual<1e-12,(error,residual)
    return {'ellipse_max_coordinate_residual':error,'determinant_max_residual':residual,'checks':'all six maps; ellipse; unit-sphere left/right zero curves'}


def axes3(title: str, labels: tuple[str,str,str]):
    fig=plt.figure(figsize=(7.0,5.8));ax=fig.add_subplot(111,projection='3d')
    ax.set_title(title,pad=18);ax.set_xlabel(labels[0]);ax.set_ylabel(labels[1]);ax.set_zlabel(labels[2]);ax.view_init(25,125)
    return fig,ax


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(os.environ.get('PENTARNION_OUTPUT_DIR',str(STUDY/'part-I/outputs/plots'))))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();result=checks();print(json.dumps(result,indent=2))
    if args.check_only:return
    out=args.output.resolve();ref=STUDY/'part-I/data'
    if out==ref or ref in out.parents:parser.error('Output must be outside reference data')
    out.mkdir(parents=True,exist_ok=True);theta=np.linspace(0,np.pi,721)
    fig,ax=axes3('Images of the unit circle under $M_{2,\\gamma}$',('$i$','$j$','$k$'))
    for g in (.25,.5,.75,1.):
        q=ellipse(theta,g);ax.plot(q[:,0],q[:,1],q[:,2],label=rf'$\gamma={g:g}$')
    ax.scatter([-1],[0],[0],marker='x',s=55,label=r'$\gamma=0$: $-i$');ax.legend(loc='upper right',fontsize=9)
    fig.tight_layout();fig.savefig(out/'gamma_bracketing_ellipse_3d.png',dpi=220);plt.close(fig)
    fig,ax=axes3('The six bracketings at $\\gamma=1$',('$i$','$j$','$k$'));q=ellipse(theta,1.)
    ax.plot(q[:,0],q[:,1],q[:,2],label=r'$M_2=M_5$');ax.plot(-q[:,0],-q[:,1],-q[:,2],label=r'$M_3=M_4$')
    ax.scatter([-1],[0],[0],marker='x',s=55,label=r'$M_0=M_1=-i$');ax.legend(loc='upper right',fontsize=9)
    fig.tight_layout();fig.savefig(out/'six_bracketings_gamma1_circle.png',dpi=220);plt.close(fig)
    fig,ax=axes3('Left and right zero divisors at $\\gamma=1$',('$w$','$y$','$z$'))
    for side,style,label in [('left','-',r'$\det L_p=0$'),('right','--',r'$\det R_p=0$')]:
        q=zero_curve(np.linspace(0,2*np.pi,721),side);ax.plot(q[:,0],q[:,1],q[:,2],style,label=label,linewidth=2)
    # Three great circles provide the unit-sphere reference without hiding the loci.
    t=np.linspace(0,2*np.pi,181)
    for j in range(3):
        q=np.zeros((len(t),3));q[:,(j+1)%3]=np.cos(t);q[:,(j+2)%3]=np.sin(t);ax.plot(*q.T,linewidth=.5,alpha=.2)
    ax.set_box_aspect((1,1,1));ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1);ax.legend(loc='upper right',fontsize=9)
    fig.tight_layout();fig.savefig(out/'gamma1_left_right_determinant_geometry.png',dpi=220);plt.close(fig)
    (out/'bracket_geometry_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Generated three equation-based bracket geometry figures in',out)

if __name__=='__main__': main()
