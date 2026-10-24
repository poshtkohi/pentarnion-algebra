#!/usr/bin/env python3
"""Exact coordinate verification of the Pentarnion Jacobiator-Hodge identity.

This script deliberately works with three completely general Pentarnions
A, B, C (15 independent real coordinates) and expands the canonical
multiplication table over Q(sqrt(2)).
"""

import sympy as sp

SQRT2 = sp.sqrt(2)
DIM = 5

# Structure tensor M[a,b,c]: e_a e_b = sum_c M[a,b,c] e_c
# Basis: e_0=1, e_1=i, e_2=j, e_3=k, e_4=l.
M = sp.MutableDenseNDimArray.zeros(DIM, DIM, DIM)
for a in range(DIM):
    M[0, a, a] = 1
    M[a, 0, a] = 1
for a in range(1, DIM):
    M[a, a, 0] = -1


def put(a, b, terms):
    for c, value in terms.items():
        M[a, b, c] = value


put(1, 2, {3: 1 / SQRT2, 4: 1 / SQRT2})
put(2, 1, {3: -1 / SQRT2, 4: -1 / SQRT2})
put(1, 3, {2: -1 / SQRT2, 4: -1 / SQRT2})
put(3, 1, {2: 1 / SQRT2, 4: 1 / SQRT2})
put(1, 4, {2: -1 / SQRT2, 3: -1 / SQRT2})
put(4, 1, {2: 1 / SQRT2, 3: 1 / SQRT2})
put(2, 3, {1: 1 / SQRT2, 4: 1 / SQRT2})
put(3, 2, {1: -1 / SQRT2, 4: -1 / SQRT2})
put(2, 4, {1: -1 / SQRT2, 3: -1 / SQRT2})
put(4, 2, {1: 1 / SQRT2, 3: 1 / SQRT2})
put(3, 4, {1: -1 / SQRT2, 2: -1 / SQRT2})
put(4, 3, {1: 1 / SQRT2, 2: 1 / SQRT2})


def mul(x, y):
    out = [sp.Integer(0)] * DIM
    for a in range(DIM):
        for b in range(DIM):
            if x[a] == 0 or y[b] == 0:
                continue
            for c in range(DIM):
                coeff = M[a, b, c]
                if coeff != 0:
                    out[c] += x[a] * y[b] * coeff
    return [sp.expand(v) for v in out]


def add(*vectors):
    return [sp.expand(sum(entries)) for entries in zip(*vectors)]


def scale(scalar, vector):
    return [sp.expand(scalar * value) for value in vector]


def comm(x, y):
    return add(mul(x, y), scale(-1, mul(y, x)))


a = sp.symbols("a0:5", real=True)
b = sp.symbols("b0:5", real=True)
c = sp.symbols("c0:5", real=True)
A, B, C = list(a), list(b), list(c)

# Standard Akivis/Jacobi convention:
# J(A,B,C) = [[A,B],C] + [[B,C],A] + [[C,A],B].
J = add(comm(comm(A, B), C), comm(comm(B, C), A), comm(comm(C, A), B))

u = sp.Matrix(a[1:])
v = sp.Matrix(b[1:])
w = sp.Matrix(c[1:])

# q = *(u wedge v wedge w) for orientation i wedge j wedge k wedge l > 0.
q1 = -sp.Matrix([[u[1], u[2], u[3]], [v[1], v[2], v[3]], [w[1], w[2], w[3]]]).det()
q2 = sp.Matrix([[u[0], u[2], u[3]], [v[0], v[2], v[3]], [w[0], w[2], w[3]]]).det()
q3 = -sp.Matrix([[u[0], u[1], u[3]], [v[0], v[1], v[3]], [w[0], w[1], w[3]]]).det()
q4 = sp.Matrix([[u[0], u[1], u[2]], [v[0], v[1], v[2]], [w[0], w[1], w[2]]]).det()
q = sp.Matrix([sp.expand(q1), sp.expand(q2), sp.expand(q3), sp.expand(q4)])
sigma = sp.Matrix([1, 1, 1, 1])
delta = sp.expand((sigma.T * q)[0])

# Standard left-nested Jacobiator is the negative of the right-nested
# convention used in some exploratory notes.
expected = [sp.Integer(0)] + [sp.expand(-4 * (q[r] - delta)) for r in range(4)]
assert all(sp.expand(J[r] - expected[r]) == 0 for r in range(DIM))

# Verify that scalar coordinates of A,B,C cancel from J.
for scalar_symbol in (a[0], b[0], c[0]):
    assert all(sp.diff(component, scalar_symbol) == 0 for component in J)

# Exact norm-volume identity.
gram = sp.Matrix(
    [
        [u.dot(u), u.dot(v), u.dot(w)],
        [v.dot(u), v.dot(v), v.dot(w)],
        [w.dot(u), w.dot(v), w.dot(w)],
    ]
)
left_norm = sp.expand(sum(component**2 for component in J))
right_norm = sp.expand(16 * (gram.det() + 2 * delta**2))
assert sp.expand(left_norm - right_norm) == 0

# The distortion operator T = I - sigma sigma^T is invertible.
T = sp.eye(4) - sigma * sigma.T
assert T.det() == -3
assert T.inv() == sp.eye(4) - sp.Rational(1, 3) * sigma * sigma.T

# Spot-check the canonical triple i,j,k.
e1 = [0, 1, 0, 0, 0]
e2 = [0, 0, 1, 0, 0]
e3 = [0, 0, 0, 1, 0]
J_ijk = add(comm(comm(e1, e2), e3), comm(comm(e2, e3), e1), comm(comm(e3, e1), e2))
assert J_ijk == [0, 4, 4, 4, 0]

print("PASS: full 15-variable coordinate identity")
print("J_std(A,B,C) = -4 (I - sigma sigma^T) *(u wedge v wedge w)")
print("PASS: scalar parts cancel")
print("PASS: norm-volume identity")
print("PASS: det(I - sigma sigma^T) = -3")
print("PASS: J_std(i,j,k) = 4(i+j+k)")
