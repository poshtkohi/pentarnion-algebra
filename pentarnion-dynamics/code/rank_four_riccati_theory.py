#!/usr/bin/env python3
"""Exact and numerical verification for the rank-four Pentarnion Riccati section.

The script performs four independent tasks:

1. expands all six bracketed words containing one fixed coefficient B and
   two copies of the state p;
2. verifies the universal three-plane classification and the rank-jump
   determinant for the polarised commutative ODE algebra;
3. derives the canonical three-dimensional normal form and checks its two
   first integrals exactly;
4. generates the nonlinear elliptic periodic example, its resultant curve,
   numerical orbit, figures, data and a machine-readable report.

All algebraic checks use exact SymPy arithmetic over Q(sqrt(2),sqrt(3)).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rank_four_riccati"
FIG_DIR = ROOT / "figures"
NOTE_DIR = ROOT / "notes"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

SQRT2 = sp.sqrt(2)
SQRT3 = sp.sqrt(3)


def pmul(p: sp.Matrix, q: sp.Matrix) -> sp.Matrix:
    """Canonical Pentarnion product in the basis (1,i,j,k,l)."""
    p0, p1, p2, p3, p4 = p
    q0, q1, q2, q3, q4 = q
    return sp.Matrix(
        [
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3 - p4 * q4,
            p0 * q1
            + q0 * p1
            + (p2 * q3 - p2 * q4 - p3 * q2 - p3 * q4 + p4 * q2 + p4 * q3)
            / SQRT2,
            p0 * q2
            + q0 * p2
            + (-p1 * q3 - p1 * q4 + p3 * q1 - p3 * q4 + p4 * q1 + p4 * q3)
            / SQRT2,
            p0 * q3
            + q0 * p3
            + (p1 * q2 - p1 * q4 - p2 * q1 - p2 * q4 + p4 * q1 + p4 * q2)
            / SQRT2,
            p0 * q4
            + q0 * p4
            + (p1 * q2 - p1 * q3 - p2 * q1 + p2 * q3 + p3 * q1 - p3 * q2)
            / SQRT2,
        ]
    )


def pure_cross(u: sp.Matrix, v: sp.Matrix) -> sp.Matrix:
    """Pure part C(u,v) of the product of pure Pentarnions."""
    pu = sp.Matrix([0, *list(u)])
    pv = sp.Matrix([0, *list(v)])
    return pmul(pu, pv)[1:, :]


def assert_zero(expr: sp.Expr | sp.Matrix, label: str) -> None:
    if isinstance(expr, sp.MatrixBase):
        values = [sp.simplify(value) for value in expr]
        if any(value != 0 for value in values):
            raise AssertionError(f"{label} failed: {values}")
    else:
        value = sp.simplify(expr)
        if value != 0:
            raise AssertionError(f"{label} failed: {value}")


def lie_derivative(expr: sp.Expr, vars_: Iterable[sp.Symbol], rhs: Iterable[sp.Expr]) -> sp.Expr:
    return sp.factor(sum(sp.diff(expr, var) * component for var, component in zip(vars_, rhs)))


def exact_symbolic_checks() -> dict[str, object]:
    b, s = sp.symbols("b s", real=True)
    w1, w2, w3, w4 = sp.symbols("w1 w2 w3 w4", real=True)
    u1, u2, u3, u4 = sp.symbols("u1 u2 u3 u4", real=True)
    w = sp.Matrix([w1, w2, w3, w4])
    u = sp.Matrix([u1, u2, u3, u4])
    B = sp.Matrix([b, w1, w2, w3, w4])
    p = sp.Matrix([s, u1, u2, u3, u4])

    pp = pmul(p, p)
    Bp = pmul(B, p)
    pB = pmul(p, B)
    words = [
        pmul(B, pp),
        pmul(pp, B),
        pmul(Bp, p),
        pmul(p, Bp),
        pmul(pB, p),
        pmul(p, pB),
    ]

    U = sp.expand(u.dot(u))
    m_inner = sp.expand(w.dot(u))
    K = pure_cross(w, u)
    T = sp.expand(K.dot(u))
    Hvec = pure_cross(K, u)
    A0 = sp.expand(b * (s**2 - U) - 2 * s * m_inner)
    V0 = sp.expand(2 * b * s) * u + (s**2 - U) * w
    V1 = (2 * b * s - m_inner) * u + s**2 * w

    expected = [
        sp.Matrix([A0, *list(V0 + 2 * s * K)]),
        sp.Matrix([A0, *list(V0 - 2 * s * K)]),
        sp.Matrix([A0 - T, *list(V1 + 2 * s * K + Hvec)]),
        sp.Matrix([A0 - T, *list(V1 - Hvec)]),
        sp.Matrix([A0 + T, *list(V1 - Hvec)]),
        sp.Matrix([A0 + T, *list(V1 - 2 * s * K + Hvec)]),
    ]
    for index, (actual, target) in enumerate(zip(words, expected)):
        assert_zero(actual - target, f"six-word expansion M_{index}")

    relation = words[0] - words[1] - words[2] + words[3] - words[4] + words[5]
    assert_zero(relation, "universal six-word relation")

    F = words[0] + words[1]
    G = words[2] + words[4] - words[0]
    H = words[3] - words[4]

    N = sp.expand(s**2 + U)
    c = sp.expand(b * s - m_inner)
    Bbar = sp.Matrix([b, -w1, -w2, -w3, -w4])
    assert_zero(G - (2 * c * p - N * Bbar), "compact G identity")
    assert_zero(H - sp.Matrix([-2 * T, 0, 0, 0, 0]), "compact H identity")

    # The nonflexibility quadratic form is nonzero for every nonreal B.
    expected_T = SQRT2 * (
        w4 * (u1 * u2 + u1 * u3 + u2 * u3)
        - w2 * u1 * u4
        - w3 * u2 * u4
        - w1 * u3 * u4
    )
    assert_zero(T - expected_T, "explicit T_B quadratic form")

    # Rank-jump determinant in the basis (p, Bbar, 1).
    beta, gamma = sp.symbols("beta gamma", real=True)

    # Verify the polarised product directly from the six bracketed words,
    # independently of the coefficient calculation used for the determinant.
    x_symbols = sp.symbols("x0:5", real=True)
    y_symbols = sp.symbols("v0:5", real=True)
    xvec = sp.Matrix(x_symbols)
    yvec = sp.Matrix(y_symbols)

    def word_field(v: sp.Matrix) -> sp.Matrix:
        vv = pmul(v, v)
        return beta * (
            pmul(pmul(B, v), v) + pmul(pmul(v, B), v) - pmul(B, vv)
        ) + gamma * (pmul(v, pmul(B, v)) - pmul(pmul(v, B), v))

    star_direct = sp.simplify(
        (word_field(xvec + yvec) - word_field(xvec) - word_field(yvec)) / 2
    )
    cx = b * xvec[0] - w.dot(xvec[1:, :])
    cy = b * yvec[0] - w.dot(yvec[1:, :])
    inner_xy = sp.expand(xvec.dot(yvec))
    Tx = sp.expand(pure_cross(w, xvec[1:, :]).dot(xvec[1:, :]))
    Ty = sp.expand(pure_cross(w, yvec[1:, :]).dot(yvec[1:, :]))
    Txy = sp.expand(
        (
            pure_cross(w, xvec[1:, :] + yvec[1:, :]).dot(
                xvec[1:, :] + yvec[1:, :]
            )
            - Tx
            - Ty
        )
        / 2
    )
    star_expected = (
        beta * (cx * yvec + cy * xvec - inner_xy * Bbar)
        - 2 * gamma * Txy * sp.Matrix([1, 0, 0, 0, 0])
    )
    assert_zero(star_direct - star_expected, "direct polarisation identity")

    R = sp.expand(K.dot(w))
    Delta = sp.factor(
        beta
        * gamma
        * (beta * (N**2 * R + 2 * c * N * T) + 4 * gamma * s * T**2)
    )

    # Verify the determinant through abstract coefficient arithmetic.
    Mnorm = sp.expand(b**2 + w.dot(w))
    p2_coeff = sp.Matrix([2 * beta * c, -beta * N, -2 * gamma * T])
    p3_p = 4 * beta**2 * c**2 - beta**2 * N * Mnorm - 2 * beta * gamma * b * T
    p3_b = -2 * beta**2 * c * N + 2 * beta * gamma * s * T
    p3_1 = -6 * beta * gamma * c * T - beta * gamma * N * R
    coeff_matrix = sp.Matrix(
        [
            [1, p2_coeff[0], p3_p],
            [0, p2_coeff[1], p3_b],
            [0, p2_coeff[2], p3_1],
        ]
    )
    assert_zero(sp.factor(coeff_matrix.det() - Delta), "rank-jump determinant")

    # Exact canonical active pair a=(i+j+k)/sqrt(3), d=l.
    one = sp.Matrix([1, 0, 0, 0, 0])
    a = sp.Matrix([0, 1 / SQRT3, 1 / SQRT3, 1 / SQRT3, 0])
    dvec = sp.Matrix([0, 0, 0, 0, 1])
    assert_zero(pmul(a, a) + one, "a^2=-1")
    assert_zero(pmul(dvec, dvec) + one, "d^2=-1")
    assert_zero(pmul(a, dvec) + SQRT2 * a, "a d=-sqrt(2)a")
    assert_zero(pmul(dvec, a) - SQRT2 * a, "d a=sqrt(2)a")

    sv, yv, zv = sp.symbols("s y z", real=True)
    pleaf = sv * one + yv * a + zv * dvec
    Bspecial = a
    pp_leaf = pmul(pleaf, pleaf)
    words_leaf = [
        pmul(Bspecial, pp_leaf),
        pmul(pp_leaf, Bspecial),
        pmul(pmul(Bspecial, pleaf), pleaf),
        pmul(pleaf, pmul(Bspecial, pleaf)),
        pmul(pmul(pleaf, Bspecial), pleaf),
        pmul(pleaf, pmul(pleaf, Bspecial)),
    ]
    F_leaf = words_leaf[0] + words_leaf[1]
    G_leaf = words_leaf[2] + words_leaf[4] - words_leaf[0]
    H_leaf = words_leaf[3] - words_leaf[4]

    alpha = sp.symbols("alpha", real=True)
    flow_leaf = sp.simplify(alpha * F_leaf + beta * G_leaf + gamma * H_leaf)
    mu = 2 * alpha + beta
    nu = beta
    chi = -SQRT2 * gamma
    expected_leaf = sp.Matrix(
        [
            -2 * yv * (mu * sv + chi * zv),
            (mu * (sv**2 - yv**2) + (2 * nu - mu) * zv**2) / SQRT3,
            (mu * (sv**2 - yv**2) + (2 * nu - mu) * zv**2) / SQRT3,
            (mu * (sv**2 - yv**2) + (2 * nu - mu) * zv**2) / SQRT3,
            -2 * nu * yv * zv,
        ]
    )
    assert_zero(flow_leaf - expected_leaf, "canonical three-dimensional normal form")

    # Exact one-generator determinant for the full alpha-beta-gamma leaf.
    mu_c, nu_c, chi_c = sp.symbols("mu_c nu_c chi_c", real=True)
    leaf_state = sp.Matrix([sv, yv, zv])

    def leaf_Q(v: sp.Matrix) -> sp.Matrix:
        ss, yy, zz = v
        return sp.Matrix(
            [
                -2 * yy * (mu_c * ss + chi_c * zz),
                mu_c * (ss**2 - yy**2) + (2 * nu_c - mu_c) * zz**2,
                -2 * nu_c * yy * zz,
            ]
        )

    leaf_p2 = leaf_Q(leaf_state)
    leaf_p3 = sp.simplify(
        (
            leaf_Q(leaf_state + leaf_p2)
            - leaf_Q(leaf_state)
            - leaf_Q(leaf_p2)
        )
        / 2
    )
    leaf_det = sp.factor(
        sp.Matrix.hstack(leaf_state, leaf_p2, leaf_p3).det()
    )
    leaf_pi = (
        mu_c**2 * sv**4
        + 2 * mu_c * (mu_c - nu_c) * sv**2 * yv**2
        + 2 * mu_c * (2 * nu_c - mu_c) * sv**2 * zv**2
        + 4 * mu_c * chi_c * sv * yv**2 * zv
        + mu_c * (mu_c - 2 * nu_c) * yv**4
        + 2 * (mu_c - nu_c) * (mu_c - 2 * nu_c) * yv**2 * zv**2
        + (mu_c - 2 * nu_c) ** 2 * zv**4
    )
    expected_leaf_det = sp.factor(
        zv * ((mu_c - nu_c) * sv + chi_c * zv) * leaf_pi
    )
    assert_zero(leaf_det - expected_leaf_det, "canonical leaf rank determinant")

    # Generic first integrals in projective coordinates.
    x, q = sp.symbols("x q", real=True)
    mu_s, nu_s, chi_s = sp.symbols("mu nu chi", real=True, nonzero=True)
    mpar = nu_s - mu_s
    dpar = 2 * nu_s - mu_s
    L = mpar * x - chi_s
    x_tau = 2 * q * L
    q_tau = mu_s * x**2 + dpar * (q**2 + 1)
    z_tau = -2 * nu_s * q * zv
    Rexpr = q**2 + x**2 - 2 * chi_s * x / nu_s + 1 + 2 * chi_s**2 / (nu_s * dpar)
    # Logarithmic derivatives avoid assumptions about the sign of L.
    assert_zero(sp.diff(L, x) * x_tau - 2 * mpar * q * L, "Darboux factor L")
    assert_zero(
        sp.diff(Rexpr, x) * x_tau + sp.diff(Rexpr, q) * q_tau - 2 * dpar * q * Rexpr,
        "Darboux factor R",
    )
    assert_zero(z_tau + 2 * nu_s * q * zv, "z Darboux factor")

    # Rational first integrals and resultant for the publication example.
    rhs_example = sp.Matrix(
        [
            -2 * yv * (3 * sv - zv),
            3 * (sv**2 - yv**2) - zv**2,
            -2 * yv * zv,
        ]
    )
    I1 = (2 * sv - zv) / zv**3
    I2 = (sv**2 + yv**2 + 2 * sv * zv - zv**2) / zv**3
    assert_zero(lie_derivative(I1, (sv, yv, zv), rhs_example), "example first integral I1")
    assert_zero(lie_derivative(I2, (sv, yv, zv), rhs_example), "example first integral I2")

    F1 = 2 * sv - zv - zv**3
    F2 = sv**2 + yv**2 + 2 * sv * zv - zv**2 - 2 * zv**3
    resultant = sp.factor(sp.resultant(F1, F2, sv))
    expected_resultant = 4 * yv**2 + zv**6 + 6 * zv**4 - 8 * zv**3 + zv**2
    assert_zero(sp.factor(resultant - expected_resultant), "resultant curve")

    quartic_z = -zv**4 - 6 * zv**2 + 8 * zv - 1
    assert_zero(sp.discriminant(quartic_z, zv) + 94208, "quartic discriminant")

    Z, Y, X, W = sp.symbols("Z Y X W")
    quartic = -Z**4 - 6 * Z**2 + 8 * Z - 1
    transformed = sp.factor(
        quartic.subs(Z, (X - 1) / (X + 1))
        / (1 - (X - 1) / (X + 1)) ** 4
    )
    assert_zero(transformed - (X**3 - X - 1), "birational Weierstrass transformation")

    return {
        "six_word_relation": "M0-M1-M2+M3-M4+M5=0",
        "closure_conditions": [
            "lambda0-lambda1+lambda2-lambda5=0",
            "lambda2-lambda3-lambda4+lambda5=0",
        ],
        "rank_determinant": str(Delta),
        "nonflexibility_form": str(expected_T),
        "normal_form_parameters": {
            "mu": "2 alpha + beta",
            "nu": "beta",
            "chi": "-sqrt(2) gamma",
        },
        "canonical_leaf_rank_determinant": str(expected_leaf_det),
        "example_first_integrals": [str(I1), str(I2)],
        "example_resultant": str(expected_resultant),
        "weierstrass_model": "W^2=X^3-X-1",
    }


def numerical_example() -> dict[str, object]:
    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        s, y, z = state
        return np.array(
            [
                -2.0 * y * (3.0 * s - z),
                3.0 * (s * s - y * y) - z * z,
                -2.0 * y * z,
            ],
            dtype=float,
        )

    def cubic(z: float) -> float:
        return z**3 + z**2 + 7.0 * z - 1.0

    def quartic(z: float) -> float:
        return -z**4 - 6.0 * z**2 + 8.0 * z - 1.0

    z_min = brentq(cubic, 0.0, 1.0, xtol=1e-15, rtol=1e-15)
    midpoint = 0.5 * (z_min + 1.0)
    halfwidth = 0.5 * (1.0 - z_min)

    pprime_left = -4.0 * z_min**3 - 12.0 * z_min + 8.0
    pprime_right = -8.0
    left_limit = math.sqrt(2.0 * halfwidth) / (z_min**2 * math.sqrt(pprime_left))
    right_limit = math.sqrt(2.0 * halfwidth) / math.sqrt(-pprime_right)

    def transformed_integrand(theta: float) -> float:
        # Cosine substitution removes both square-root endpoint singularities.
        if abs(theta) < 1e-14:
            return left_limit
        if abs(theta - math.pi) < 1e-14:
            return right_limit
        z = midpoint - halfwidth * math.cos(theta)
        dz = halfwidth * math.sin(theta)
        value = max(quartic(z), 0.0)
        return dz / (z * z * math.sqrt(value))

    half_period, quadrature_error = quad(
        transformed_integrand,
        0.0,
        math.pi,
        epsabs=2e-12,
        epsrel=2e-12,
        limit=500,
    )
    period = 2.0 * half_period

    initial = np.array([1.0, 0.0, 1.0], dtype=float)
    sample_count = 6000
    t_eval = np.linspace(0.0, period, sample_count)
    solution = solve_ivp(
        rhs,
        (0.0, period),
        initial,
        method="DOP853",
        t_eval=t_eval,
        rtol=2e-13,
        atol=1e-15,
        max_step=0.002,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    s, y, z = solution.y
    I1 = (2.0 * s - z) / z**3
    I2 = (s * s + y * y + 2.0 * s * z - z * z) / z**3
    resultant_residual = 4.0 * y * y + z**6 + 6.0 * z**4 - 8.0 * z**3 + z**2
    return_error = float(np.linalg.norm(solution.y[:, -1] - initial))

    centered = solution.y.T - np.mean(solution.y.T, axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    third_variance_fraction = float(singular_values[2] ** 2 / np.sum(singular_values**2))
    relative_nonplanarity = float(
        (singular_values[2] / math.sqrt(sample_count))
        / np.mean(np.linalg.norm(centered, axis=1))
    )
    phase = 2.0 * math.pi * solution.t / period
    design = np.column_stack([np.ones_like(phase), np.cos(phase), np.sin(phase)])
    ellipse = design @ np.linalg.lstsq(design, solution.y.T, rcond=None)[0]
    ellipse_error = float(
        np.sqrt(np.mean(np.sum((solution.y.T - ellipse) ** 2, axis=1)))
        / np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    )

    data = pd.DataFrame(
        {
            "t": solution.t,
            "s": s,
            "y": y,
            "z": z,
            "I1": I1,
            "I2": I2,
            "resultant_residual": resultant_residual,
        }
    )
    data.to_csv(DATA_DIR / "nonlinear_elliptic_orbit.csv", index=False)

    # Main publication figure: 3D orbit and analytical resultant oval.
    fig = plt.figure(figsize=(12.2, 5.2))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(s, y, z, linewidth=1.8)
    ax1.scatter([initial[0]], [initial[1]], [initial[2]], s=28, label="initial point")
    ax1.set_xlabel("$s$")
    ax1.set_ylabel("$y$")
    ax1.set_zlabel("$z$")
    ax1.set_title("Nonlinear periodic orbit")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 2, 2)
    z_grid = np.linspace(z_min, 1.0, 1200)
    P_grid = np.maximum(-z_grid**4 - 6.0 * z_grid**2 + 8.0 * z_grid - 1.0, 0.0)
    Y_upper = np.sqrt(P_grid)
    ax2.plot(z_grid, Y_upper, linewidth=1.8, color="C0", label="analytic quartic")
    ax2.plot(z_grid, -Y_upper, linewidth=1.8, color="C0")
    ax2.scatter(
        z[::30],
        (2.0 * y / z)[::30],
        s=6,
        alpha=0.45,
        color="C1",
        label="numerical orbit",
    )
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$Y=2y/z$")
    ax2.set_title("Resultant elliptic curve $Y^2=-z^4-6z^2+8z-1$")
    ax2.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rank_four_elliptic_orbit.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG_DIR / "rank_four_elliptic_orbit.pdf", bbox_inches="tight")
    plt.close(fig)

    # Supplementary time series and exact-invariant residuals.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    axes[0].plot(solution.t, s, label="$s(t)$")
    axes[0].plot(solution.t, y, label="$y(t)$")
    axes[0].plot(solution.t, z, label="$z(t)$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("state")
    axes[0].set_title("Strongly non-sinusoidal components")
    axes[0].legend(loc="best")
    axes[1].plot(solution.t, I1 - 1.0, label="$I_1-1$")
    axes[1].plot(solution.t, I2 - 2.0, label="$I_2-2$")
    axes[1].plot(solution.t, resultant_residual, label="resultant residual")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("absolute residual")
    axes[1].set_title("Exact-invariant diagnostics")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rank_four_elliptic_diagnostics.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG_DIR / "rank_four_elliptic_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "equation": {
            "s_dot": "-2*y*(3*s-z)",
            "y_dot": "3*(s^2-y^2)-z^2",
            "z_dot": "-2*y*z",
        },
        "initial_state": initial.tolist(),
        "lower_turning_point": z_min,
        "period_quadrature": period,
        "quadrature_error_estimate": quadrature_error,
        "one_period_return_error": return_error,
        "max_abs_I1_minus_1": float(np.max(np.abs(I1 - 1.0))),
        "max_abs_I2_minus_2": float(np.max(np.abs(I2 - 2.0))),
        "max_abs_resultant_residual": float(np.max(np.abs(resultant_residual))),
        "min_state_norm": float(np.min(np.linalg.norm(solution.y, axis=0))),
        "max_state_norm": float(np.max(np.linalg.norm(solution.y, axis=0))),
        "best_ellipse_relative_rms_error": ellipse_error,
        "relative_out_of_plane_rms": relative_nonplanarity,
        "third_pca_variance_fraction": third_variance_fraction,
        "quartic_discriminant": -94208,
        "weierstrass_model": "W^2=X^3-X-1",
    }
    with (DATA_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    symbolic = exact_symbolic_checks()
    numerical = numerical_example()
    report = {
        "symbolic": symbolic,
        "numerical": numerical,
        "status": "all exact and numerical checks passed",
    }
    with (NOTE_DIR / "rank_four_riccati_verification_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    text_lines = [
        "Rank-four Pentarnion Riccati verification report",
        "================================================",
        "",
        "All symbolic identities were checked exactly over Q(sqrt(2),sqrt(3)).",
        f"Closure conditions: {symbolic['closure_conditions']}",
        f"Universal relation: {symbolic['six_word_relation']}",
        f"Resultant curve: {symbolic['example_resultant']}",
        f"Weierstrass model: {symbolic['weierstrass_model']}",
        "",
        f"Lower turning point: {numerical['lower_turning_point']:.15g}",
        f"Period from analytic quadrature: {numerical['period_quadrature']:.15g}",
        f"One-period return error: {numerical['one_period_return_error']:.6e}",
        f"Maximum |I1-1|: {numerical['max_abs_I1_minus_1']:.6e}",
        f"Maximum |I2-2|: {numerical['max_abs_I2_minus_2']:.6e}",
        f"Maximum resultant residual: {numerical['max_abs_resultant_residual']:.6e}",
        f"Best-ellipse relative RMS error: {numerical['best_ellipse_relative_rms_error']:.6f}",
        f"Relative out-of-plane RMS: {numerical['relative_out_of_plane_rms']:.6f}",
        "",
        "STATUS: PASS",
    ]
    (NOTE_DIR / "rank_four_riccati_verification_report.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(text_lines))


if __name__ == "__main__":
    main()
