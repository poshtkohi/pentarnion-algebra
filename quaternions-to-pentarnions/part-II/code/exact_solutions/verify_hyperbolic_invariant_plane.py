#!/usr/bin/env python3
"""Exact substitution checks for prop:hyperbolic-invariant-plane in Part II.

Checks the invariant plane, its translated linear system, the matrix-exponential
formula (gamma>0), the gamma=0 formula, and the stated endpoint example.
This is a new supporting verifier for an existing manuscript proposition, not
a global integrability or arbitrary-initial-condition claim.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sympy as s


def require_zero(expr, name: str) -> None:
    entries = list(expr) if isinstance(expr, s.MatrixBase) else [expr]
    residual = [s.simplify(s.expand(e)) for e in entries]
    if any(e != 0 for e in residual):
        raise RuntimeError(f'{name}: nonzero residual {residual}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'outputs/exact_solutions')
    args = parser.parse_args()
    mu, gamma = s.symbols('mu gamma', positive=True, real=True)
    m = mu*gamma
    p0,p1,p2,p3,p4 = p = s.symbols('p0:5', real=True)
    F = s.Matrix([
        p1-p3+p0*p1+2*(1+m)*p0*p3-2*(1+m)*p1*p2,
        p2-(1+m)*p0-p0**2+2*m*p0*p2,
        2*p0*p1,
        1-2*(1+m)*p0**2+3*mu*p4*p0,
        3*gamma*p0*p3,
    ])
    xi=1/(2*(1+m))
    zeta=(2*m**2+4*m+3)/(4*(1+m)*(1+2*m))
    shift=(1+2*m)/(3*mu)
    U,W=s.symbols('U W', real=True)
    restricted=s.simplify(F.subs({p0:xi,p1:0,p2:zeta,p3:U,p4:W-shift}))
    a=3*mu/(2*(1+m)); b=3*gamma/(2*(1+m))
    require_zero(restricted-s.Matrix([0,0,0,a*W,b*U]),'invariant plane and reduction')
    L=s.Matrix([[0,a],[b,0]])
    lam=3*s.sqrt(mu*gamma)/(2*(1+m))
    require_zero(L*L-lam**2*s.eye(2),'coefficient matrix square')
    t,U0,W0=s.symbols('t U0 W0',real=True)
    z=s.Matrix([U0*s.cosh(lam*t)+s.sqrt(mu/gamma)*W0*s.sinh(lam*t),
                W0*s.cosh(lam*t)+s.sqrt(gamma/mu)*U0*s.sinh(lam*t)])
    require_zero(z.diff(t)-L*z,'hyperbolic solution')
    require_zero(z.subs(t,0)-s.Matrix([U0,W0]),'initial condition')
    L0=s.simplify(L.subs(gamma,0)); z0=s.Matrix([U0+s.Rational(3,2)*mu*W0*t,W0])
    require_zero(z0.diff(t)-L0*z0,'gamma=0 endpoint')
    example=s.Matrix([s.Rational(1,3),0,s.Rational(11,24),
                      2*s.sqrt(2)/3*s.sinh(t/s.sqrt(2)),
                      s.Rational(4,3)*(s.cosh(t/s.sqrt(2))-1)])
    require_zero(example.diff(t)-F.subs({gamma:1,mu:s.Rational(1,2)}).subs(dict(zip(p,example))), 'endpoint example in all five equations')
    lines=[
        'Exact check: Part II, prop:hyperbolic-invariant-plane',
        'Coordinates: (p0,p1,p2,p3,p4)=(scalar,i,j,k,ell); m=mu*gamma.',
        'PASS invariant plane and reduced equations (symbolic).',
        'PASS squared reduced matrix equals lambda^2 I.',
        'PASS hyperbolic solution and initial data (symbolic).',
        'PASS gamma=0 linear-in-time formula (symbolic).',
        'PASS mu=1/2, gamma=1 example in all five equations.',
        'No numerical sampling or interval certification is used.',
        'Scope: these are solutions on the specified invariant affine plane.',
        'The calculation does not classify other initial conditions.',
    ]
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'hyperbolic_invariant_plane_checks.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':
    main()
