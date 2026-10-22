"""Component-wise numerical product in (1,i,j,k,ell).

Moved unchanged from the original Riccati model. Argument order is
product(p, q, gamma), unlike the independent table implementation
algebra_core.product(gamma, p, q). The test suite compares both.
"""
from __future__ import annotations
from math import sqrt
import numpy as np
from numpy.typing import NDArray
Array = NDArray[np.float64]

def product(a: Array, b: Array, gamma: float) -> Array:
    """Return ``a star_gamma b`` in coordinates (1,i,j,k,ell)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != (5,) or b.shape != (5,):
        raise ValueError("a and b must be length-5 coordinate vectors")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must lie in [0,1]")

    kappa = 1.0 / sqrt(1.0 + gamma * gamma)
    sigma = gamma * kappa
    inv_sqrt2 = 1.0 / sqrt(2.0)

    q = np.zeros(5, dtype=float)
    q[0] = a[0] * b[0] - float(np.dot(a[1:], b[1:]))
    q[1:] = a[0] * b[1:] + b[0] * a[1:]

    # ij, jk, ki and reversed products.
    s12 = a[1] * b[2] - a[2] * b[1]
    q[3] += kappa * s12
    q[4] += sigma * s12

    s23 = a[2] * b[3] - a[3] * b[2]
    q[1] += kappa * s23
    q[4] += sigma * s23

    s31 = a[3] * b[1] - a[1] * b[3]
    q[2] += kappa * s31
    q[4] += sigma * s31

    # ell-input products.
    s41 = a[4] * b[1] - a[1] * b[4]
    q[2] += inv_sqrt2 * s41
    q[3] += inv_sqrt2 * s41

    s42 = a[4] * b[2] - a[2] * b[4]
    q[3] += inv_sqrt2 * s42
    q[1] += inv_sqrt2 * s42

    s43 = a[4] * b[3] - a[3] * b[4]
    q[1] += inv_sqrt2 * s43
    q[2] += inv_sqrt2 * s43

    return q

