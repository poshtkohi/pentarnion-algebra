#!/usr/bin/env python3
"""Exact SymPy certificate for the invariant ideal and switching path."""
from __future__ import annotations

from pathlib import Path
import sympy as s

ROOT = Path(__file__).resolve().parents[2] / 'outputs/exact_integrable'
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

w, x, y, z, u = s.symbols("w x y z u", real=True)
gamma, mu = s.symbols("gamma mu", real=True)
m = mu*gamma

F = s.Matrix([
    x-z+w*x+2*(1+m)*w*z-2*(1+m)*x*y,
    y-(1+m)*w-w**2+2*m*w*y,
    2*w*x,
    1-2*(1+m)*w**2+3*mu*u*w,
    3*gamma*w*z,
])

G1 = y-w**2
G2 = z-w*x
G3 = u-gamma*w*y
G4 = x**2-m*y**2+(1+m)*y-1
G = [G1, G2, G3, G4]
variables = (w, x, y, z, u)

LF = [s.expand(sum(s.diff(gj, variable)*F[index]
                   for index, variable in enumerate(variables))) for gj in G]
claimed = [
    4*(1+m)*w*x*G1 + 2*w*(1-2*(1+m)*w)*G2,
    (2*(1+m)*x**2-m*y-w+1+m)*G1
    +x*(1-2*(1+m)*w)*G2 + 3*mu*w*G3-G4,
    gamma*x*(2*(1+m)*y-1)*G1
    +gamma*(y+3*w-2*(1+m)*w*y)*G2,
    2*x*G1,
]

checks = [s.expand(left-right) for left, right in zip(LF, claimed)]
assert checks == [0, 0, 0, 0]

# Restriction to the invariant curve.
subs_curve = {y:w**2, z:w*x, u:gamma*w**3}
restricted = [s.expand(component.subs(subs_curve)) for component in F]
curve_relation = s.Eq(x**2, (1-w**2)*(1-m*w**2))
expected = [
    x,
    -(1+m)*w+2*m*w**3,
    2*w*x,
    1-2*(1+m)*w**2+3*m*w**4,
    3*gamma*w**2*x,
]
# The z derivative agrees after reducing x^2 by the quartic relation.
restricted_reduced = restricted.copy()
restricted_reduced[3] = s.expand(restricted_reduced[3])
assert s.expand(restricted[0]-expected[0]) == 0
assert s.expand(restricted[1]-expected[1]) == 0
assert s.expand(restricted[2]-expected[2]) == 0
assert s.expand(restricted[3]-expected[3]) == 0
assert s.expand(restricted[4]-expected[4]) == 0

# Check that d(wx)/dt reduces to the same z equation on the quartic curve.
zdot_from_product = s.expand(x**2 + w*expected[1])
quartic_sub = {x**2:(1-w**2)*(1-m*w**2)}
zdot_reduced = s.expand(zdot_from_product.subs(quartic_sub))
assert s.expand(zdot_reduced-expected[3]) == 0

# Algebraic reduction of the ideal.
reduced_G4 = s.expand(G4.subs({y:w**2}))
assert s.expand(reduced_G4 - (x**2-(1-w**2)*(1-m*w**2))) == 0

certificate = f"""PASS: all four invariant-ideal Lie derivative identities vanish exactly.
PASS: the restricted vector field agrees with the Jacobi reduction.
PASS: the eliminated curve is x^2=(1-w^2)(1-m w^2), m=mu*gamma.

Lie derivative identities:
L_F G1 = {s.sstr(claimed[0])}
L_F G2 = {s.sstr(claimed[1])}
L_F G3 = {s.sstr(claimed[2])}
L_F G4 = {s.sstr(claimed[3])}
"""
(DATA / "symbolic_certificate.txt").write_text(certificate, encoding="utf-8")
print(certificate)
