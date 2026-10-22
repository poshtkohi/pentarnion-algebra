"""SymPy evaluation and verification of selected rotational formulas."""
from __future__ import annotations
import sympy as s
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "outputs/rotations/symbolic_verification.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)
g = s.symbols('gamma', nonnegative=True, real=True)
A, B, C, D = s.symbols('A B C D', real=True)
k = 1 / s.sqrt(1 + g**2)
sig = g / s.sqrt(1 + g**2)
rt2, rt3 = s.sqrt(2), s.sqrt(3)

Cu = s.Matrix([
    [rt2*D, -k*C, k*B, -rt2*A],
    [k*C, -D/rt2, -k*A, B/rt2],
    [-k*B, k*A, -D/rt2, C/rt2],
    [0, -rt3*sig*C, rt3*sig*B, 0],
])

sympart = s.simplify(Cu + Cu.T)
u = s.Matrix([A, B, C, D])
pure_condition = s.simplify(Cu.T * u)

Cgamma = rt3 / s.sqrt(3 + g**2)
Sgamma = g / s.sqrt(3 + g**2)
mu = s.sqrt((3 + g**2) / (3 * (1 + g**2)))
threeform_checks = [s.simplify(mu*Cgamma-k), s.simplify(mu*Sgamma-sig/rt3)]

Ag = 3 * (g**4 + 2*g**2 + 3) / ((1+g**2)*(3+g**2))
Bg = (4*g**2 + 3) / (1+g**2)
Ader = s.factor(s.diff(Ag, g))
Bder = s.factor(s.diff(Bg, g))

lam = s.symbols('lambda')
C1, S1 = rt3/2, s.Rational(1,2)
Rb0 = s.Matrix([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
Rc1 = s.Matrix([
    [S1**2, -C1, 0, -C1*S1],
    [C1, 0, 0, S1],
    [0,0,1,0],
    [-C1*S1, -S1, 0, C1**2],
])
Q = s.simplify(Rc1 * Rb0)
charpoly = s.factor(Q.charpoly(lam).as_expr())

# Fail on incorrect identities instead of merely writing residuals to a report.
if any(value != 0 for value in threeform_checks):
    raise RuntimeError("Three-form factorisation failed")
if s.simplify((Q-s.eye(4)).det()-1) != 0:
    raise RuntimeError("The stated det(Q-I)=1 check failed")
if s.simplify(Q.T*Q-s.eye(4)) != s.zeros(4):
    raise RuntimeError("The rotation matrix is not orthogonal")

lines = []
lines.append('Symmetric part of C_u:')
lines.append(str(sympart))
lines.append('\nC_u^T u (pure-preserving condition):')
lines.append(str(pure_condition))
lines.append('\nThree-form factorization residuals (must be zero):')
lines.append(str(threeform_checks))
lines.append('\nA_gamma derivative: ' + str(Ader))
lines.append('B_gamma derivative: ' + str(Bder))
lines.append('\nExact double-rotation matrix:')
lines.append(str(Q))
lines.append('Characteristic polynomial: ' + str(charpoly))
lines.append('\nAll reported residual identities simplify exactly to zero where indicated.')
OUT.write_text('\n'.join(lines), encoding='utf-8')
print(OUT)
