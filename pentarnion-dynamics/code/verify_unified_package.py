#!/usr/bin/env python3
"""Independent numerical/algebraic/data checks for the unified ODE package.

The rank-four and genus-one theorem certificates are checked separately by
``rank_four_riccati_theory.py`` and ``genus_one_symbolic_verification.py``.
This script independently checks the regenerated numerical assets, canonical
Pentarnion products, appendix associator deformation, and bracketing-family data.

Authors: Alireza Poshtkohi and Susanne Pumplün
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RANK4 = DATA / "rank_four_riccati"
GENUS = DATA / "elliptic_flow"
APPENDIX = DATA / "appendix_associator_235"
BRACKETING = DATA / "bracketing_flow"
HIGH = BRACKETING / "high_resolution"
NOTES = ROOT / "notes"
NOTES.mkdir(parents=True, exist_ok=True)

SQRT2 = math.sqrt(2.0)
R2 = 1.0 / SQRT2

# ---------------------------------------------------------------------------
# Canonical Pentarnion multiplication in B=(1,i,j,k,ell).
# ---------------------------------------------------------------------------
TENSOR = np.zeros((5, 5, 5), dtype=float)
for a in range(5):
    TENSOR[0, a, a] = 1.0
    TENSOR[a, 0, a] = 1.0
for a in range(1, 5):
    TENSOR[a, a, 0] = -1.0


def set_product(a: int, b: int, values: list[float]) -> None:
    TENSOR[a, b, :] = np.asarray(values, dtype=float)


set_product(1, 2, [0, 0, 0, R2, R2]); set_product(2, 1, [0, 0, 0, -R2, -R2])
set_product(2, 3, [0, R2, 0, 0, R2]); set_product(3, 2, [0, -R2, 0, 0, -R2])
set_product(3, 1, [0, 0, R2, 0, R2]); set_product(1, 3, [0, 0, -R2, 0, -R2])
set_product(4, 1, [0, 0, R2, R2, 0]); set_product(1, 4, [0, 0, -R2, -R2, 0])
set_product(4, 2, [0, R2, 0, R2, 0]); set_product(2, 4, [0, -R2, 0, -R2, 0])
set_product(4, 3, [0, R2, R2, 0, 0]); set_product(3, 4, [0, -R2, -R2, 0, 0])

ONE = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
I = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
J = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
K = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
ELL = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
CONJUGATION = np.diag([1.0, -1.0, -1.0, -1.0, -1.0])


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("a,b,abc->c", a, b, TENSOR)


def conj(p: np.ndarray) -> np.ndarray:
    return CONJUGATION @ p


def assoc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return mul(mul(a, b), c) - mul(a, mul(b, c))


def left_matrix(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3, q4 = np.asarray(q, dtype=float)
    return np.array([
        [q0, -q1, -q2, -q3, -q4],
        [q1, q0, (-q3 + q4) / SQRT2, (q2 + q4) / SQRT2, (-q2 - q3) / SQRT2],
        [q2, (q3 + q4) / SQRT2, q0, (-q1 + q4) / SQRT2, (-q1 - q3) / SQRT2],
        [q3, (-q2 + q4) / SQRT2, (q1 + q4) / SQRT2, q0, (-q1 - q2) / SQRT2],
        [q4, (-q2 + q3) / SQRT2, (q1 - q3) / SQRT2, (-q1 + q2) / SQRT2, q0],
    ])


def right_matrix(q: np.ndarray) -> np.ndarray:
    return CONJUGATION @ left_matrix(conj(q)) @ CONJUGATION


def square_map(x: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3, x4 = np.asarray(x, dtype=float)
    return np.array([
        x0*x0 - x1*x1 - x2*x2 - x3*x3 - x4*x4,
        2*x0*x1,
        2*x0*x2,
        2*x0*x3,
        2*x0*x4,
    ])


# ---------------------------------------------------------------------------
# Elliptic flow: irreducible genus-one Pentarnion flow.
# ---------------------------------------------------------------------------
N_MAP = np.array([
    [0, 1, 0, -1, 0],
    [-1.5, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=float)
M_MAP = np.array([
    [0, 0, 0, 0, 0],
    [-0.5, 0, 0.5, 0, 0],
    [0, 1, 0, -9/8, -9/8],
    [-1.5, 0, 9/8, 0, -3/8],
    [0, 0, 9/8, 3/8, 0],
], dtype=float)
K_MAP = np.array([
    [0, 0, 0, 0, 0],
    [9*SQRT2/8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=float)


def genus_components(p: np.ndarray) -> np.ndarray:
    w, x, y, z, u = p
    return np.array([
        x - z + w*x + 3*w*z - 3*x*y,
        y - 1.5*w - w*w + w*y,
        2*w*x,
        1 - 3*w*w + 1.5*u*w,
        3*w*z,
    ])


def genus_compact(p: np.ndarray) -> np.ndarray:
    return (
        K
        + N_MAP @ p
        + mul(p, (M_MAP + K_MAP) @ p)
        + mul((M_MAP - K_MAP) @ p, p)
    )


def genus_exact_derivative_from_state(p: np.ndarray) -> np.ndarray:
    """Derivative on the invariant Jacobi curve, expressed using its state."""
    w, x, _y, _z, _u = p
    return np.array([
        x,
        w**3 - 1.5*w,
        2*w*x,
        1 - 3*w*w + 1.5*w**4,
        3*w*w*x,
    ])


# ---------------------------------------------------------------------------
# Appendix A: associator compensation and destruction of a 2:3:5 orbit.
# ---------------------------------------------------------------------------
def compensated_components(p: np.ndarray) -> np.ndarray:
    w, x, y, z, u = p
    return np.array([
        -2*x + 4*w*x,
        2*w - 2*w*w + 2*x*x,
        -3*z + 5*u*y,
        3*y + 5*u*z,
        2.5*(1 - y*y - z*z + u*u),
    ])


def split_q_r(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = -0.5 * mul(I, mul(I, p) + mul(p, I))
    return q, p - q


def associator_family_rhs(p: np.ndarray, eta: float) -> np.ndarray:
    q, r = split_q_r(p)
    term_q = 2.0 * mul(I, mul(q, ONE - q))
    term_rotation = 1.5 * (
        mul(J, mul(K, r) + mul(r, K))
        - mul(mul(J, r) + mul(r, J), K)
    )
    term_nonassociative = 2.5 * (
        ELL - mul(mul(r, ELL), r) - eta * assoc(ELL, r, r)
    )
    return term_q + term_rotation + term_nonassociative


def compensated_compact(p: np.ndarray) -> np.ndarray:
    return associator_family_rhs(p, 1.0)


def exact_compensated_state_and_derivative(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta2 = 2*t
    theta3 = 3*t
    theta5 = 5*t
    D2 = 5 + 4*np.cos(theta2)
    D5 = 2 + np.cos(theta5)
    dD2 = -8*np.sin(theta2)
    dD5 = -5*np.sin(theta5)

    n0 = 1 + 2*np.cos(theta2)
    n1 = 2*np.sin(theta2)
    n2 = math.sqrt(3)*np.cos(theta3)
    n3 = math.sqrt(3)*np.sin(theta3)
    n4 = np.sin(theta5)
    dn0 = -4*np.sin(theta2)
    dn1 = 4*np.cos(theta2)
    dn2 = -3*math.sqrt(3)*np.sin(theta3)
    dn3 = 3*math.sqrt(3)*np.cos(theta3)
    dn4 = 5*np.cos(theta5)

    state = np.column_stack((n0/D2, n1/D2, n2/D5, n3/D5, n4/D5))
    derivative = np.column_stack((
        (dn0*D2 - n0*dD2)/(D2*D2),
        (dn1*D2 - n1*dD2)/(D2*D2),
        (dn2*D5 - n2*dD5)/(D5*D5),
        (dn3*D5 - n3*dD5)/(D5*D5),
        (dn4*D5 - n4*dD5)/(D5*D5),
    ))
    return state, derivative


# ---------------------------------------------------------------------------
# Bracketing family: sparse rational Pentarnion dynamics.
# ---------------------------------------------------------------------------
A2 = np.array([-0.5, -2.0, 0.5, -1.0, -0.5])
B2 = np.array([1.0, 0.0, 2.0, 0.0, -0.5])
P02 = np.array([0.5, -0.5, -0.5, 0.5, -0.5])


def bracketing_direct(p: np.ndarray, eta: float) -> np.ndarray:
    return (
        mul(A2, p)
        + (1.0-eta)*mul(B2, mul(p, p))
        + eta*mul(mul(B2, p), p)
    )


def bracketing_matrix(p: np.ndarray, eta: float) -> np.ndarray:
    return (
        left_matrix(A2) @ p
        + (1.0-eta)*(left_matrix(B2) @ square_map(p))
        + eta*(right_matrix(p) @ (left_matrix(B2) @ p))
    )


def bracketing_analytic_jacobian(p: np.ndarray, eta: float) -> np.ndarray:
    return (
        left_matrix(A2)
        + (1.0-eta)*left_matrix(B2) @ (left_matrix(p) + right_matrix(p))
        + eta*(right_matrix(p) @ left_matrix(B2) + left_matrix(mul(B2, p)))
    )


def finite_difference_jacobian(fun, p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    result = np.empty((5, 5))
    for j in range(5):
        h = np.zeros(5)
        h[j] = eps
        result[:, j] = (fun(p+h) - fun(p-h))/(2*eps)
    return result


# ---------------------------------------------------------------------------
# Random algebraic checks.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260803)
errors: dict[str, float] = {
    "left_matrix": 0.0,
    "right_matrix": 0.0,
    "square_map": 0.0,
    "genus_compact_vs_components": 0.0,
    "appendix_q_projection": 0.0,
    "compensated_compact_vs_components": 0.0,
    "bracketing_direct_vs_matrix": 0.0,
    "bracketing_jacobian": 0.0,
}

for _ in range(1000):
    p = rng.normal(size=5)
    q = rng.normal(size=5)
    eta = float(rng.random())

    errors["left_matrix"] = max(
        errors["left_matrix"],
        float(np.max(np.abs(left_matrix(q) @ p - mul(q, p)))),
    )
    errors["right_matrix"] = max(
        errors["right_matrix"],
        float(np.max(np.abs(right_matrix(q) @ p - mul(p, q)))),
    )
    errors["square_map"] = max(
        errors["square_map"],
        float(np.max(np.abs(square_map(p) - mul(p, p)))),
    )
    errors["genus_compact_vs_components"] = max(
        errors["genus_compact_vs_components"],
        float(np.max(np.abs(genus_compact(p) - genus_components(p)))),
    )

    projected, _ = split_q_r(p)
    expected_projection = np.array([p[0], p[1], 0.0, 0.0, 0.0])
    errors["appendix_q_projection"] = max(
        errors["appendix_q_projection"],
        float(np.max(np.abs(projected - expected_projection))),
    )
    errors["compensated_compact_vs_components"] = max(
        errors["compensated_compact_vs_components"],
        float(np.max(np.abs(compensated_compact(p) - compensated_components(p)))),
    )

    errors["bracketing_direct_vs_matrix"] = max(
        errors["bracketing_direct_vs_matrix"],
        float(np.max(np.abs(bracketing_direct(p, eta) - bracketing_matrix(p, eta)))),
    )
    J_exact = bracketing_analytic_jacobian(p, eta)
    J_fd = finite_difference_jacobian(lambda state: bracketing_direct(state, eta), p)
    errors["bracketing_jacobian"] = max(
        errors["bracketing_jacobian"],
        float(np.max(np.abs(J_exact - J_fd))),
    )


# ---------------------------------------------------------------------------
# Rank-four Riccati exact-curve and numerical-orbit data checks.
# ---------------------------------------------------------------------------
rank4_summary = json.loads((RANK4 / "summary.json").read_text(encoding="utf-8"))
rank4_data = pd.read_csv(RANK4 / "nonlinear_elliptic_orbit.csv")
r4_t = rank4_data["t"].to_numpy(dtype=float)
r4_s = rank4_data["s"].to_numpy(dtype=float)
r4_y = rank4_data["y"].to_numpy(dtype=float)
r4_z = rank4_data["z"].to_numpy(dtype=float)
r4_I1 = (2.0*r4_s-r4_z)/r4_z**3
r4_I2 = (r4_s**2+r4_y**2+2.0*r4_s*r4_z-r4_z**2)/r4_z**3
r4_resultant = 4.0*r4_y**2+r4_z**6+6.0*r4_z**4-8.0*r4_z**3+r4_z**2
r4_return_error = float(np.linalg.norm(
    np.array([r4_s[-1], r4_y[-1], r4_z[-1]])
    - np.array(rank4_summary["initial_state"], dtype=float)
))
r4_lower = float(rank4_summary["lower_turning_point"])
rank4_data_error = max(
    abs(int(r4_t.size)-6000),
    abs(float(r4_t[0])),
    abs(float(r4_t[-1])-float(rank4_summary["period_quadrature"])),
    float(np.max(np.abs(r4_I1-1.0))),
    float(np.max(np.abs(r4_I2-2.0))),
    float(np.max(np.abs(r4_resultant))),
    abs(r4_return_error-float(rank4_summary["one_period_return_error"])),
    abs(r4_lower**3+r4_lower**2+7.0*r4_lower-1.0),
    abs(float(rank4_summary["quartic_discriminant"])+94208.0),
)

# ---------------------------------------------------------------------------
# High-resolution Elliptic flow data checks.
# ---------------------------------------------------------------------------
genus_archive = np.load(GENUS / "exact_orbit_high_resolution.npz")
tg = np.asarray(genus_archive["t"], dtype=float)
Xg = np.asarray(genus_archive["x"], dtype=float)
genus_summary = json.loads((GENUS / "summary.json").read_text(encoding="utf-8"))

genus_invariant_error = float(np.max(np.abs(np.column_stack((
    Xg[:, 2] - Xg[:, 0]**2,
    Xg[:, 3] - Xg[:, 0]*Xg[:, 1],
    Xg[:, 4] - Xg[:, 0]*Xg[:, 2],
    Xg[:, 1]**2 - 0.5*Xg[:, 2]**2 + 1.5*Xg[:, 2] - 1,
)))))
genus_rhs_error = float(np.max(np.abs(
    np.vstack([genus_components(row) for row in Xg])
    - np.vstack([genus_exact_derivative_from_state(row) for row in Xg])
)))
genus_closure_error = float(np.max(np.abs(Xg[0] - Xg[-1])))
genus_sample_error = max(
    abs(int(tg.size) - 200001),
    abs(int(Xg.shape[0]) - 200001),
    abs(float(tg[0])),
    abs(float(tg[-1]) - float(genus_summary["period"])),
    abs(float(genus_archive["period"]) - float(genus_summary["period"])),
    abs(float(genus_archive["modulus"]) - 0.5),
)

# ---------------------------------------------------------------------------
# High-resolution appendix associator deformation checks.
# ---------------------------------------------------------------------------
app_archive = np.load(APPENDIX / "exact_orbit_high_resolution.npz")
tr = np.asarray(app_archive["t"], dtype=float)
Xr = np.asarray(app_archive["x"], dtype=float)
app_summary = json.loads((APPENDIX / "summary.json").read_text(encoding="utf-8"))
Xr_expected, dXr_expected = exact_compensated_state_and_derivative(tr)
app_state_error = float(np.max(np.abs(Xr - Xr_expected)))
app_rhs_error = float(np.max(np.abs(
    np.vstack([compensated_components(row) for row in Xr]) - dXr_expected
)))
app_closure_error = float(np.max(np.abs(Xr[0] - Xr[-1])))
app_sample_error = max(
    abs(int(tr.size) - 200001),
    abs(int(Xr.shape[0]) - 200001),
    abs(float(tr[0])),
    abs(float(tr[-1]) - 2*math.pi),
    abs(float(app_archive["period"]) - 2*math.pi),
    abs(float(app_summary["period"]) - 2*math.pi),
)

app_deformed = np.load(APPENDIX / "deformed_eta_0p9_high_resolution.npz")
td = np.asarray(app_deformed["t"], dtype=float)
Xd = np.asarray(app_deformed["x"], dtype=float)
eta_d = float(app_deformed["eta"])
p0_appendix = np.array([1/3, 0.0, 1/math.sqrt(3), 0.0, 0.0])
r0_appendix = np.array([0.0, 0.0, 1/math.sqrt(3), 0.0, 0.0])
expected_app_assoc = np.array([0.0, -1/6, 0.0, 1/6, 1/3])
app_assoc = assoc(ELL, r0_appendix, r0_appendix)
app_assoc_error = float(np.max(np.abs(app_assoc - expected_app_assoc)))
app_tangent_identity_error = float(np.max(np.abs(
    associator_family_rhs(p0_appendix, eta_d)
    - associator_family_rhs(p0_appendix, 1.0)
    - 2.5*(1.0-eta_d)*app_assoc
)))
app_return_defect = float(np.linalg.norm(Xd[-1] - p0_appendix))
app_deformed_data_error = max(
    abs(int(td.size) - 200001),
    abs(int(Xd.shape[0]) - 200001),
    float(np.max(np.abs(td - tr))),
    float(np.max(np.abs(Xd[0] - p0_appendix))),
    abs(eta_d - 0.9),
    abs(app_return_defect - float(app_summary["deformed_one_period_return_defect"])),
    abs(float(np.max(np.asarray(app_deformed["deviation_from_exact"])))
        - float(app_summary["deformed_maximum_deviation_from_exact"])),
)

# ---------------------------------------------------------------------------
# Preserved Bracketing family exact and data-consistency checks.
# ---------------------------------------------------------------------------
expected_bracketing_assoc = np.array([3*SQRT2, 7.0, 13.0, -5.0, -15.0])/8.0
bracketing_assoc = assoc(B2, P02, P02)
bracketing_assoc_error = float(np.max(np.abs(bracketing_assoc - expected_bracketing_assoc)))
bracketing_assoc_norm_error = abs(float(np.linalg.norm(bracketing_assoc)) - 9*math.sqrt(6)/8)

lyapunov_table = pd.read_csv(BRACKETING / "full_lyapunov_spectra.csv")
bracketing_summary = json.loads((BRACKETING / "summary.json").read_text(encoding="utf-8"))
bracketing_data_error = max(
    abs(float(lyapunov_table.loc[lyapunov_table.eta == 0, "lambda_1"].iloc[0])
        - float(bracketing_summary["eta0_lyapunov"][0])),
    abs(float(lyapunov_table.loc[lyapunov_table.eta == 1, "lambda_1"].iloc[0])
        - float(bracketing_summary["eta1_lyapunov"][0])),
    float(np.max(np.abs(np.asarray(bracketing_summary["p0"], dtype=float) - P02))),
)

traj0 = np.load(HIGH / "trajectory_eta0_high_resolution.npz")
traj1 = np.load(HIGH / "trajectory_eta1_high_resolution.npz")
bracketing_sample_error = max(
    abs(int(traj0["t"].size) - 600001),
    abs(int(traj1["t"].size) - 100001),
    abs(float(np.median(np.diff(traj0["t"]))) - 0.005),
    abs(float(np.median(np.diff(traj1["t"]))) - 0.005),
    abs(int(np.asarray(traj0["x"]).shape[0]) - 600001),
    abs(int(np.asarray(traj1["x"]).shape[0]) - 100001),
)

# ---------------------------------------------------------------------------
# Report and thresholds.
# ---------------------------------------------------------------------------
exact_threshold = 2e-11
jacobian_threshold = 2e-7
exact_checks = [
    errors["left_matrix"],
    errors["right_matrix"],
    errors["square_map"],
    errors["genus_compact_vs_components"],
    errors["appendix_q_projection"],
    errors["compensated_compact_vs_components"],
    rank4_data_error,
    r4_return_error,
    genus_invariant_error,
    genus_rhs_error,
    genus_closure_error,
    genus_sample_error,
    app_state_error,
    app_rhs_error,
    app_closure_error,
    app_sample_error,
    app_assoc_error,
    app_tangent_identity_error,
    app_deformed_data_error,
    errors["bracketing_direct_vs_matrix"],
    bracketing_assoc_error,
    bracketing_assoc_norm_error,
    bracketing_data_error,
    bracketing_sample_error,
]

report = f"""Unified Pentarnion ODE package verification
===========================================

Canonical multiplication
------------------------
Left-multiplication matrix error:                         {errors['left_matrix']:.3e}
Right-multiplication matrix error:                        {errors['right_matrix']:.3e}
Square-map error:                                        {errors['square_map']:.3e}

Rank-four homogeneous Riccati theory
--------------------------------------
Archived curve/invariant/data consistency:                   {rank4_data_error:.3e}
One-period numerical return error:                           {r4_return_error:.3e}
Quadrature period:                                           {float(rank4_summary['period_quadrature']):.12f}
Best first-harmonic ellipse relative RMS error:              {float(rank4_summary['best_ellipse_relative_rms_error']):.6f}

Elliptic flow: irreducible genus-one Pentarnion flow
-------------------------------------------
Compact-Pentarnion/component error:                       {errors['genus_compact_vs_components']:.3e}
Invariant-curve data error:                               {genus_invariant_error:.3e}
Exact Jacobi orbit/component residual:                    {genus_rhs_error:.3e}
One-period closure error:                                 {genus_closure_error:.3e}
Sample-count/period/modulus consistency:                  {genus_sample_error:.3e}

Appendix A: associator-controlled 2:3:5 flow
------------------------------------------------------
Canonical q-projection error:                             {errors['appendix_q_projection']:.3e}
Compact-Pentarnion/component error:                       {errors['compensated_compact_vs_components']:.3e}
Archived-state/closed-form error:                         {app_state_error:.3e}
Exact orbit/component residual:                           {app_rhs_error:.3e}
One-period closure error:                                 {app_closure_error:.3e}
Sample-count/period consistency:                          {app_sample_error:.3e}
Initial-associator exact-value error:                      {app_assoc_error:.3e}
Associator tangent-identity error:                         {app_tangent_identity_error:.3e}
Deformed-orbit archive consistency:                        {app_deformed_data_error:.3e}
One-period return defect at eta=0.9:                       {app_return_defect:.12f}

Bracketing family: Sparse Rational Pentarnion Dynamics
----------------------------------------------
Matrix/direct vector-field error:                         {errors['bracketing_direct_vs_matrix']:.3e}
Analytic/finite-difference Jacobian error:                 {errors['bracketing_jacobian']:.3e}
Initial-associator exact-value error:                      {bracketing_assoc_error:.3e}
Initial-associator exact-norm error:                       {bracketing_assoc_norm_error:.3e}
Regenerated-data consistency error:                       {bracketing_data_error:.3e}
High-resolution sample-count/time-step consistency:       {bracketing_sample_error:.3e}

Exact/data threshold: {exact_threshold:.1e}
Finite-difference Jacobian threshold: {jacobian_threshold:.1e}
"""
(ROOT / "verification_report.txt").write_text(report, encoding="utf-8")
print(report)

if max(exact_checks) > exact_threshold:
    raise SystemExit("Verification failed: an exact or data check exceeded tolerance")
if app_return_defect < 0.1:
    raise SystemExit("Verification failed: the appendix deformation did not destroy one-period recurrence")
if errors["bracketing_jacobian"] > jacobian_threshold:
    raise SystemExit("Verification failed: the Bracketing family Jacobian check exceeded tolerance")
