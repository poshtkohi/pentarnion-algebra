"""Exact/numerical core for the Quaternion--Pentarnion switching algebra.

Coordinate order in the original basis is (1,i,j,k,ell).
Coordinate order in the adapted pure basis is (a,b,c,d), with d=ell.
"""
from __future__ import annotations
import math
import numpy as np


def coefficients(gamma: float) -> tuple[float, float]:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0,1]")
    r = math.sqrt(1.0 + gamma * gamma)
    return 1.0 / r, gamma / r


def product(gamma: float, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Bilinear product in the original basis (1,i,j,k,ell)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != (5,) or q.shape != (5,):
        raise ValueError("p and q must have shape (5,)")
    kap, sig = coefficients(gamma)
    rt2 = math.sqrt(2.0)
    e = np.eye(5)
    table: dict[tuple[int, int], np.ndarray] = {}
    for j in range(5):
        table[(0, j)] = e[j]
        table[(j, 0)] = e[j]
    for j in range(1, 5):
        table[(j, j)] = -e[0]

    def pure(i=0.0, j=0.0, k=0.0, ell=0.0):
        return np.array([0.0, i, j, k, ell])

    table[(1, 2)] = pure(k=kap, ell=sig)
    table[(2, 1)] = -table[(1, 2)]
    table[(2, 3)] = pure(i=kap, ell=sig)
    table[(3, 2)] = -table[(2, 3)]
    table[(3, 1)] = pure(j=kap, ell=sig)
    table[(1, 3)] = -table[(3, 1)]
    table[(1, 4)] = pure(j=-1 / rt2, k=-1 / rt2)
    table[(4, 1)] = -table[(1, 4)]
    table[(2, 4)] = pure(i=-1 / rt2, k=-1 / rt2)
    table[(4, 2)] = -table[(2, 4)]
    table[(3, 4)] = pure(i=-1 / rt2, j=-1 / rt2)
    table[(4, 3)] = -table[(3, 4)]

    out = np.zeros(5)
    for a in range(5):
        for b in range(5):
            out += p[a] * q[b] * table[(a, b)]
    return out


def conjugate(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy()
    p[1:] *= -1.0
    return p


def associator(gamma: float, p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return product(gamma, product(gamma, p, q), r) - product(gamma, p, product(gamma, q, r))


def adapted_to_original_matrix() -> np.ndarray:
    return np.array(
        [
            [1 / math.sqrt(3), 1 / math.sqrt(2), 1 / math.sqrt(6), 0],
            [1 / math.sqrt(3), -1 / math.sqrt(2), 1 / math.sqrt(6), 0],
            [1 / math.sqrt(3), 0, -2 / math.sqrt(6), 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def moving_frame(gamma: float) -> np.ndarray:
    """Columns are (h_gamma,b,c,n_gamma) in fixed (a,b,c,d) coordinates."""
    r = math.sqrt(3 + gamma * gamma)
    C = math.sqrt(3) / r
    S = gamma / r
    return np.array([[C, 0, 0, -S], [0, 1, 0, 0], [0, 0, 1, 0], [S, 0, 0, C]])
