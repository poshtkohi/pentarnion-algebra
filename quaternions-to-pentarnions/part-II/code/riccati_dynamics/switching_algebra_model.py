"""Core numerical model for the quaternion--Pentarnion switching algebra.

This module contains the final multiplication law used in the report and the
Riccati-type vector fields used by the static and time-dependent simulations.
Basis order: (1, i, j, k, ell).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


# Keep the original product(p,q,gamma) API for callers of this model.
import sys
from pathlib import Path
_STUDY = Path(__file__).resolve().parents[3]
if str(_STUDY) not in sys.path:
    sys.path.insert(0, str(_STUDY))
from shared.coordinate_product import product


def weighted_coefficients(gamma: float) -> tuple[Array, Array, Array]:
    """Return A_gamma, B_gamma and p_gamma(0) used in the report."""
    A = np.array([-0.5, -2.0, 0.5, -1.0, -0.5 * gamma], dtype=float)
    B = np.array([1.0, 0.0, 2.0, 0.0, -0.5 * gamma], dtype=float)
    p0 = 0.5 * np.array([1.0, -1.0, -1.0, 1.0, -gamma], dtype=float)
    return A, B, p0


def static_rhs(_t: float, p: Array, gamma: float) -> Array:
    """Final static left-bracketed Riccati switching model."""
    A, B, _ = weighted_coefficients(gamma)
    pp = product(p, p, gamma)
    return product(A, p, gamma) + product(B, pp, gamma)


def tangent_action(p: Array, h: Array, gamma: float) -> Array:
    """Exact Jacobian action D F_gamma(p) h by bilinearity."""
    A, B, _ = weighted_coefficients(gamma)
    ph_hp = product(p, h, gamma) + product(h, p, gamma)
    return product(A, h, gamma) + product(B, ph_hp, gamma)


@dataclass(frozen=True)
class SwitchingProtocol:
    center: float
    amplitude: float

    def gamma(self, t: float, omega: float) -> float:
        return self.center + self.amplitude * np.sin(omega * t)

    def gamma_dot(self, t: float, omega: float) -> float:
        return self.amplitude * omega * np.cos(omega * t)

    @property
    def minimum(self) -> float:
        return self.center - self.amplitude

    @property
    def maximum(self) -> float:
        return self.center + self.amplitude


def hidden_rhs(
    t: float,
    q: Array,
    omega: float,
    protocol: SwitchingProtocol,
) -> Array:
    """Hidden-coordinate model p=(w,x,y,z,gamma*v).

    This formulation requires gamma(t)>0. The report uses protocols bounded
    away from zero.
    """
    gamma = float(protocol.gamma(t, omega))
    gamma_dot = float(protocol.gamma_dot(t, omega))
    if gamma <= 0.0:
        raise ZeroDivisionError("hidden-coordinate model requires gamma(t)>0")

    p = np.array([q[0], q[1], q[2], q[3], gamma * q[4]], dtype=float)
    fp = static_rhs(t, p, gamma)
    return np.array(
        [fp[0], fp[1], fp[2], fp[3], (fp[4] - gamma_dot * q[4]) / gamma],
        dtype=float,
    )


TRANSITION_BAND = SwitchingProtocol(center=0.68, amplitude=0.08)
REGULARIZED_FULL_RANGE = SwitchingProtocol(center=0.525, amplitude=0.475)
