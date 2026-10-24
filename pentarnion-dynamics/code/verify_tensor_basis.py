#!/usr/bin/env python3
"""Independent tensor/basis verification of the Jacobiator-Hodge theorem.

Unlike verify_direct_coordinates.py, this script never expands three general
Pentarnions. It works only with the alternating tensor Pi on the pure space V,
constructs the cross-Jacobiator on a basis of Lambda^3 V, and identifies the
resulting linear map through the Hodge star.
"""

import itertools
import sympy as sp

R2 = sp.sqrt(2)
E = [sp.eye(4).col(i) for i in range(4)]

# Canonical pure product Pi(e_a wedge e_b).
POS = {
    (0, 1): {2: 1 / R2, 3: 1 / R2},
    (1, 2): {0: 1 / R2, 3: 1 / R2},
    (2, 0): {1: 1 / R2, 3: 1 / R2},
    (3, 0): {1: 1 / R2, 2: 1 / R2},
    (3, 1): {2: 1 / R2, 0: 1 / R2},
    (3, 2): {0: 1 / R2, 1: 1 / R2},
}


def basis_cross(a, b):
    out = sp.zeros(4, 1)
    if a == b:
        return out
    if (a, b) in POS:
        sign = 1
        terms = POS[(a, b)]
    elif (b, a) in POS:
        sign = -1
        terms = POS[(b, a)]
    else:
        raise RuntimeError("missing product")
    for index, coeff in terms.items():
        out[index] = sign * coeff
    return out


def cross(x, y):
    out = sp.zeros(4, 1)
    for a in range(4):
        for b in range(4):
            out += x[a] * y[b] * basis_cross(a, b)
    return sp.simplify(out)


def cross_jacobi(x, y, z):
    return sp.simplify(
        cross(x, cross(y, z))
        + cross(y, cross(z, x))
        + cross(z, cross(x, y))
    )


# Oriented Lambda^3 basis: e123, e124, e134, e234.
triples = list(itertools.combinations(range(4), 3))
K = sp.Matrix.hstack(*[cross_jacobi(E[a], E[b], E[c]) for a, b, c in triples])

# Hodge star matrix in the same bases:
# *e123=e4, *e124=-e3, *e134=e2, *e234=-e1.
H = sp.Matrix.hstack(E[3], -E[2], E[1], -E[0])
sigma = sp.ones(4, 1)
T = sp.eye(4) - sigma * sigma.T

assert K == T * H
assert K.rank() == 4
assert T.eigenvals() == {sp.Integer(1): 3, sp.Integer(-3): 1}

# The standard commutator Jacobiator is -4 times the cross-Jacobiator.
J_std_matrix = -4 * K
assert J_std_matrix.det() != 0

print("PASS: basis-level tensor identity K = (I - sigma sigma^T) Hodge")
print("Lambda^3 basis:", triples)
print("Cross-Jacobiator matrix:")
sp.print_latex(K)
print("eigenvalues of distortion operator:", T.eigenvals())
print("rank:", K.rank())
