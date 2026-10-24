#!/usr/bin/env python3
"""Exact symbolic verification for the dense genus-one Pentarnion flow.

Authors: Alireza Poshtkohi and Susanne Pumplün

This script verifies:
  1. invariance of the genus-one ideal for the dense all-nonlinear extension;
  2. the elliptic normal sextic quadrics;
  3. the compact matrix--Pentarnion representation;
  4. linear irreducibility of the ambient Jacobian family;
  5. exclusion of projective and conformal Riccati quadratic forms;
  6. the one-parameter Jacobi family and the exact modulus-selection
     condition m=1/2 for the two-product Pentarnion ansatz.
"""
from __future__ import annotations

from pathlib import Path
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = ROOT / "notes" / "genus_one_symbolic_verification_report.txt"
REPORT.parent.mkdir(parents=True, exist_ok=True)

w, x, y, z, u, m = sp.symbols("w x y z u m")
state = sp.Matrix([w, x, y, z, u])

# ---------------------------------------------------------------------------
# Dense all-nonlinear extension at m=1/2.
# ---------------------------------------------------------------------------
F = sp.Matrix([
    x - z + w*x + 3*w*z - 3*x*y,
    y - sp.Rational(3, 2)*w - w**2 + w*y,
    2*w*x,
    1 - 3*w**2 + sp.Rational(3, 2)*u*w,
    3*w*z,
])

G1 = y - w**2
G2 = z - w*x
G3 = u - w*y
G4 = x**2 - sp.Rational(1, 2)*y**2 + sp.Rational(3, 2)*y - 1
generators = [G1, G2, G3, G4]

def lie(field: sp.Matrix, polynomial: sp.Expr) -> sp.Expr:
    grad = sp.Matrix([sp.diff(polynomial, q) for q in state])
    return sp.expand(grad.dot(field))

ideal_claims = [
    6*w*x*G1 + 2*w*(1 - 3*w)*G2,
    (3*x**2 - w - sp.Rational(1, 2)*y + sp.Rational(3, 2))*G1
        + x*(1 - 3*w)*G2 + sp.Rational(3, 2)*w*G3 - G4,
    x*(3*y - 1)*G1 + (y + 3*w - 3*w*y)*G2,
    2*x*G1,
]
for idx, (g, claim) in enumerate(zip(generators, ideal_claims), start=1):
    assert sp.expand(lie(F, g) - claim) == 0, f"Ideal identity {idx} failed"

curve_subs = {y: w**2, z: w*x, u: w**3}
restricted = [sp.factor(component.subs(curve_subs)) for component in F]
expected_restricted = [
    x,
    w**3 - sp.Rational(3, 2)*w,
    2*w*x,
    1 - 3*w**2 + sp.Rational(3, 2)*w**4,
    3*w**2*x,
]
assert all(sp.expand(a-b) == 0 for a, b in zip(restricted, expected_restricted))

# ---------------------------------------------------------------------------
# Elliptic normal sextic quadrics.
# ---------------------------------------------------------------------------
X0, X1, X2, X3, X4, X5 = sp.symbols("X0:6")
quadrics = [
    X1**2 - X0*X3,
    X1*X2 - X0*X4,
    X1*X3 - X0*X5,
    -2*X0**2 + 3*X0*X3 - X1*X5 + 2*X2**2,
    -X1*X4 + X2*X3,
    -X1*X5 + X3**2,
    -X2*X5 + X3*X4,
    2*X0*X1 - 3*X0*X5 - 2*X2*X4 + X3*X5,
    2*X0*X3 - 3*X1*X5 - 2*X4**2 + X5**2,
]
projective_subs = {X0: 1, X1: w, X2: x, X3: w**2, X4: w*x, X5: w**3}
quartic_relation = x**2 - sp.Rational(1, 2)*w**4 + sp.Rational(3, 2)*w**2 - 1
quartic_poly = sp.Poly(quartic_relation, x, w)
for idx, quadric in enumerate(quadrics, start=1):
    remainder = sp.rem(sp.Poly(sp.expand(quadric.subs(projective_subs)), x, w), quartic_poly)
    assert remainder.as_expr() == 0, f"Projective quadric {idx} failed"

# Linear independence of the nine quadrics in the 21-dimensional vector space
# of quadrics in six homogeneous variables.
hom_vars = [X0, X1, X2, X3, X4, X5]
hom_monomials = [hom_vars[i]*hom_vars[j] for i in range(6) for j in range(i, 6)]
quadric_coeffs = []
for quadric in quadrics:
    poly = sp.Poly(quadric, *hom_vars)
    quadric_coeffs.append([
        poly.coeff_monomial(sp.Poly(mon, *hom_vars).monoms()[0])
        for mon in hom_monomials
    ])
assert sp.Matrix(quadric_coeffs).rank() == 9

# ---------------------------------------------------------------------------
# Canonical Pentarnion multiplication and compact matrix realization.
# ---------------------------------------------------------------------------
sqrt2 = sp.sqrt(2)
T = sp.MutableDenseNDimArray.zeros(5, 5, 5)
for a in range(5):
    T[0, a, a] = 1
    T[a, 0, a] = 1
for a in range(1, 5):
    T[a, a, 0] = -1

def set_product(a: int, b: int, vector: list[sp.Expr]) -> None:
    for c, coefficient in enumerate(vector):
        T[a, b, c] = sp.sympify(coefficient)

set_product(1, 2, [0, 0, 0, 1/sqrt2, 1/sqrt2])
set_product(2, 1, [0, 0, 0, -1/sqrt2, -1/sqrt2])
set_product(2, 3, [0, 1/sqrt2, 0, 0, 1/sqrt2])
set_product(3, 2, [0, -1/sqrt2, 0, 0, -1/sqrt2])
set_product(3, 1, [0, 0, 1/sqrt2, 0, 1/sqrt2])
set_product(1, 3, [0, 0, -1/sqrt2, 0, -1/sqrt2])
set_product(4, 1, [0, 0, 1/sqrt2, 1/sqrt2, 0])
set_product(1, 4, [0, 0, -1/sqrt2, -1/sqrt2, 0])
set_product(4, 2, [0, 1/sqrt2, 0, 1/sqrt2, 0])
set_product(2, 4, [0, -1/sqrt2, 0, -1/sqrt2, 0])
set_product(4, 3, [0, 1/sqrt2, 1/sqrt2, 0, 0])
set_product(3, 4, [0, -1/sqrt2, -1/sqrt2, 0, 0])

def pent_mul(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([
        sum(a[r]*b[s]*T[r, s, c] for r in range(5) for s in range(5))
        for c in range(5)
    ])

p = state
N = sp.Matrix([
    [0, 1, 0, -1, 0],
    [-sp.Rational(3, 2), 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])
M = sp.Matrix([
    [0, 0, 0, 0, 0],
    [-sp.Rational(1, 2), 0, sp.Rational(1, 2), 0, 0],
    [0, 1, 0, -sp.Rational(9, 8), -sp.Rational(9, 8)],
    [-sp.Rational(3, 2), 0, sp.Rational(9, 8), 0, -sp.Rational(3, 8)],
    [0, 0, sp.Rational(9, 8), sp.Rational(3, 8), 0],
])
K = sp.Matrix([
    [0, 0, 0, 0, 0],
    [sp.Rational(9, 8)*sqrt2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])
k_basis = sp.Matrix([0, 0, 0, 1, 0])
compact = k_basis + N*p + pent_mul(p, (M+K)*p) + pent_mul((M-K)*p, p)
assert all(sp.expand(compact[q] - F[q]) == 0 for q in range(5))

# ---------------------------------------------------------------------------
# Linear irreducibility via the exact Jacobian coefficient algebra.
# ---------------------------------------------------------------------------
Jac = F.jacobian(state)
zero_subs = {w: 0, x: 0, y: 0, z: 0, u: 0}
J0 = Jac.subs(zero_subs)
Jw = Jac.diff(w)
Jx = Jac.diff(x)
Jy = Jac.diff(y)
Jz = Jac.diff(z)
Ju = Jac.diff(u)
gens = {"J0": J0, "Jw": Jw, "Jx": Jx, "Jy": Jy, "Jz": Jz, "Ju": Ju}
word_names = [
    "I", "J0", "Jw", "Jx", "Jy", "Jz", "Ju",
    "J0J0", "J0Jw", "J0Jx", "J0Jy", "J0Jz", "J0Ju",
    "JwJ0", "JwJw", "JwJx", "JwJy",
    "JxJ0", "JxJw", "JxJx",
    "JzJ0", "JzJw", "JzJx",
    "J0J0Jw", "JxJ0Jw",
]

def evaluate_word(name: str) -> sp.Matrix:
    if name == "I":
        return sp.eye(5)
    tokens: list[str] = []
    q = 0
    while q < len(name):
        assert name[q] == "J"
        token = name[q:q+2]
        tokens.append(token)
        q += 2
    result = sp.eye(5)
    for token in tokens:
        result = result*gens[token]
    return result

word_matrices = [evaluate_word(name) for name in word_names]
word_vector_matrix = sp.Matrix.hstack(*[sp.Matrix(list(A)) for A in word_matrices])
word_determinant = sp.factor(word_vector_matrix.det())
assert word_determinant == 3**26
assert word_vector_matrix.rank() == 25

# ---------------------------------------------------------------------------
# Exact exclusion of projective and conformal Riccati quadratic fields.
# ---------------------------------------------------------------------------
# Projective Riccati: Q(p)=lambda(p)p must be collinear with p. At e_w the
# quadratic part is (0,-1,0,-3,0), not collinear with e_w.
quadratic_part = sp.Matrix([
    w*x + 3*w*z - 3*x*y,
    -w**2 + w*y,
    2*w*x,
    -3*w**2 + sp.Rational(3, 2)*u*w,
    3*w*z,
])
q_at_ew = quadratic_part.subs({w: 1, x: 0, y: 0, z: 0, u: 0})
assert q_at_ew == sp.Matrix([0, -1, 0, -3, 0])

# Conformal Riccati after a linear coordinate change would require a nonzero
# symmetric metric G satisfying the conformal Killing identities for the
# homogeneous quadratic field. Solve the exact linear system.
g_symbols = sp.symbols("g0:15")
G = sp.zeros(5)
cursor = 0
for r in range(5):
    for s in range(r, 5):
        G[r, s] = G[s, r] = g_symbols[cursor]
        cursor += 1
conformal_equations = []
for A in [Jw, Jx, Jy, Jz, Ju]:
    lam = sp.Rational(2, 5)*sp.trace(A)
    E = A.T*G + G*A - lam*G
    for r in range(5):
        for s in range(r, 5):
            conformal_equations.append(sp.expand(E[r, s]))
conformal_matrix, conformal_rhs = sp.linear_eq_to_matrix(conformal_equations, g_symbols)
assert conformal_matrix.rank() == 15
assert sp.linsolve((conformal_matrix, conformal_rhs), g_symbols) == {tuple([0]*15)}

# ---------------------------------------------------------------------------
# One-parameter Jacobi family and modulus selection.
# ---------------------------------------------------------------------------
Fm = sp.Matrix([
    x - z + w*x + 3*w*z - 3*x*y,
    y - (1+m)*w - w**2 + 2*m*w*y,
    2*w*x,
    1 - 2*(1+m)*w**2 + 3*m*u*w,
    3*w*z,
])
G4m = x**2 - m*y**2 + (1+m)*y - 1
general_generators = [G1, G2, G3, G4m]
general_claims = [
    6*w*x*G1 + 2*w*(1-3*w)*G2,
    (3*x**2 - w - m*y + m + 1)*G1
        + x*(1-3*w)*G2 + 3*m*w*G3 - G4m,
    x*(3*y-1)*G1 + (y+3*w-3*w*y)*G2,
    2*x*G1,
]
for idx, (g, claim) in enumerate(zip(general_generators, general_claims), start=1):
    assert sp.expand(lie(Fm, g) - claim) == 0, f"General ideal identity {idx} failed"

# Build the linear coefficient map for Q(p)=p A(p)+B(p) p, where A and B
# are arbitrary real-linear maps. There are 75 output/monomial coefficients
# and 50 map coefficients.
coords = [w, x, y, z, u]
monomials = [coords[r]*coords[s] for r in range(5) for s in range(r, 5)]
A_symbols = sp.symbols("a0:25")
B_symbols = sp.symbols("b0:25")
Amap = sp.Matrix(5, 5, A_symbols)
Bmap = sp.Matrix(5, 5, B_symbols)
ansatz = pent_mul(p, Amap*p) + pent_mul(Bmap*p, p)
ansatz_coefficients = []
for output in range(5):
    poly = sp.Poly(sp.expand(ansatz[output]), *coords)
    for monomial in monomials:
        exponent = sp.Poly(monomial, *coords).monoms()[0]
        ansatz_coefficients.append(poly.coeff_monomial(exponent))
coefficient_map, _ = sp.linear_eq_to_matrix(
    ansatz_coefficients, list(A_symbols)+list(B_symbols)
)
assert coefficient_map.shape == (75, 50)
assert coefficient_map.rank() == 44

# Extract the quadratic target from F_m.
target_quadratic = []
for component in Fm:
    poly = sp.Poly(component, *coords)
    expression = 0
    for exponent, coefficient in poly.terms():
        if sum(exponent) == 2:
            term = coefficient
            for variable, power in zip(coords, exponent):
                term *= variable**power
            expression += term
    target_quadratic.append(sp.expand(expression))

target_coefficients = []
for output in range(5):
    poly = sp.Poly(target_quadratic[output], *coords)
    for monomial in monomials:
        exponent = sp.Poly(monomial, *coords).monoms()[0]
        target_coefficients.append(poly.coeff_monomial(exponent))
target_vector = sp.Matrix(target_coefficients)

left_nullspace = coefficient_map.T.nullspace()
constraints = [sp.factor(vector.dot(target_vector)) for vector in left_nullspace]
nonzero_constraints = [constraint for constraint in constraints if constraint != 0]
assert nonzero_constraints == [1-2*m, 1-2*m, 1-2*m]

# Verify solvability at m=1/2 and failure generically through augmented rank.
target_half = target_vector.subs(m, sp.Rational(1, 2))
assert coefficient_map.row_join(target_half).rank() == 44
target_zero = target_vector.subs(m, 0)
assert coefficient_map.row_join(target_zero).rank() == 45

# Weierstrass invariants of the Jacobi quartic. With
# X=(x+1)/w^2 and Y=w(X^2-m), then after scaling Y by sqrt(2):
# y^2=X^3-(1+m)X^2/2-mX+m(1+m)/2.
a1 = a3 = 0
a2 = -(1+m)/2
a4 = -m
a6 = m*(1+m)/2
b2 = a1**2 + 4*a2
b4 = 2*a4 + a1*a3
b6 = a3**2 + 4*a6
b8 = 4*a2*a6 - a4**2
c4 = sp.factor(b2**2 - 24*b4)
c6 = sp.factor(-b2**3 + 36*b2*b4 - 216*b6)
discriminant = sp.factor(-b2**2*b8 - 8*b4**3 - 27*b6**2 + 9*b2*b4*b6)
j_invariant = sp.factor(c4**3/discriminant)
assert sp.simplify(c4 - 4*(m**2 + 14*m + 1)) == 0
assert sp.simplify(c6 - 8*(m+1)*(m**2 - 34*m + 1)) == 0
assert sp.simplify(discriminant - 4*m*(m-1)**4) == 0
assert sp.simplify(j_invariant - 16*(m**2 + 14*m + 1)**3/(m*(m-1)**4)) == 0
assert j_invariant.subs(m, sp.Rational(1, 2)) == 287496

lines = []
lines.append("Dense genus-one Pentarnion flow: exact verification")
lines.append("="*61)
lines.append("")
lines.append("1. Invariant-ideal identities: PASSED (4/4).")
lines.append("2. Restriction to the Jacobi quartic: PASSED.")
lines.append("3. Elliptic-normal-sextic quadrics: PASSED (9/9); rank=9.")
lines.append("4. Matrix--Pentarnion expansion: PASSED exactly.")
lines.append(f"5. Jacobian coefficient algebra determinant: {word_determinant} = 3^26.")
lines.append("   Hence the generated algebra is M_5(Q), so the ambient flow is")
lines.append("   linearly irreducible (no common nontrivial invariant subspace).")
lines.append("6. Projective Riccati test: FAILED by the field, hence excluded.")
lines.append("7. Conformal Riccati metric test: only G=0; rank=15, hence excluded.")
lines.append(f"8. Two-product coefficient map: shape={coefficient_map.shape}, rank={coefficient_map.rank()}.")
lines.append("   Nonzero compatibility constraints: " + ", ".join(map(str, nonzero_constraints)))
lines.append("   Therefore the Jacobi family is representable in this compact class iff m=1/2.")
lines.append("9. Jacobi-quartic j-invariant:")
lines.append(f"   j(m) = {sp.sstr(j_invariant)}")
lines.append("   j(1/2) = 287496 = 66^3.")
lines.append("")
lines.append("All assertions passed under exact SymPy arithmetic.")
REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
print(REPORT.read_text(encoding="utf-8"))
