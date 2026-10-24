#!/usr/bin/env python3
"""Generate and verify the exact analytical Pentarnion flows.

The main text contains the irreducible genus-one equation.
The appendix contains an associator-compensated 2:3:5 product flow and a
high-resolution deformation from the same initial value.

Authors: Alireza Poshtkohi and Susanne Pumpluen
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.special import ellipj, ellipk
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
GENUS_DATA = ROOT / "data" / "elliptic_flow"
APP_DATA = ROOT / "data" / "appendix_associator_235"
for folder in (FIG, GENUS_DATA, APP_DATA):
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SQRT2 = math.sqrt(2.0)
TENSOR = np.zeros((5, 5, 5), dtype=float)
for a in range(5):
    TENSOR[0, a, a] = 1.0
    TENSOR[a, 0, a] = 1.0
for a in range(1, 5):
    TENSOR[a, a, 0] = -1.0

def set_product(a: int, b: int, values: list[float]) -> None:
    TENSOR[a, b, :] = np.asarray(values, dtype=float)

r2 = 1.0 / SQRT2
set_product(1, 2, [0, 0, 0, r2, r2]); set_product(2, 1, [0, 0, 0, -r2, -r2])
set_product(2, 3, [0, r2, 0, 0, r2]); set_product(3, 2, [0, -r2, 0, 0, -r2])
set_product(3, 1, [0, 0, r2, 0, r2]); set_product(1, 3, [0, 0, -r2, 0, -r2])
set_product(4, 1, [0, 0, r2, r2, 0]); set_product(1, 4, [0, 0, -r2, -r2, 0])
set_product(4, 2, [0, r2, 0, r2, 0]); set_product(2, 4, [0, -r2, 0, -r2, 0])
set_product(4, 3, [0, r2, r2, 0, 0]); set_product(3, 4, [0, -r2, -r2, 0, 0])

ONE = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
I = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
J = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
K_BASIS = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
ELL = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("a,b,abc->c", a, b, TENSOR)

def assoc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return mul(mul(a, b), c) - mul(a, mul(b, c))

# -------------------------------------------------------------------------
# Irreducible genus-one Pentarnion flow.
# -------------------------------------------------------------------------
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
    ], dtype=float)

def genus_compact(p: np.ndarray) -> np.ndarray:
    return (
        K_BASIS
        + N_MAP @ p
        + mul(p, (M_MAP + K_MAP) @ p)
        + mul((M_MAP - K_MAP) @ p, p)
    )

m = 0.5
genus_period = float(4.0 * ellipk(m))
ng = 200_001
tg = np.linspace(0.0, genus_period, ng)
sn, cn, dn, _ = ellipj(tg, m)
vg = cn * dn
Xg = np.column_stack((sn, vg, sn**2, sn*vg, sn**3))
dXg = np.column_stack((
    vg,
    sn**3 - 1.5*sn,
    2*sn*vg,
    1 - 3*sn**2 + 1.5*sn**4,
    3*sn**2*vg,
))
RHSg = np.vstack([genus_components(row) for row in Xg])
residual_genus = float(np.max(np.abs(RHSg - dXg)))

rng = np.random.default_rng(20260803)
compact_genus_error = 0.0
for _ in range(1000):
    p = rng.normal(size=5)
    compact_genus_error = max(
        compact_genus_error,
        float(np.max(np.abs(genus_compact(p) - genus_components(p)))),
    )

np.savez_compressed(
    GENUS_DATA / "exact_orbit_high_resolution.npz",
    t=tg,
    x=Xg,
    period=np.array(genus_period),
    modulus=np.array(m),
)
stride = 40
pd.DataFrame(
    np.column_stack((tg[::stride], Xg[::stride])),
    columns=["t", "w", "x_i", "y_j", "z_k", "u_ell"],
).to_csv(GENUS_DATA / "exact_orbit_sample.csv", index=False)

genus_summary = {
    "initial_condition": [0.0, 1.0, 0.0, 0.0, 0.0],
    "jacobi_modulus": m,
    "period": genus_period,
    "high_resolution_samples": int(ng),
    "maximum_exact_orbit_residual": residual_genus,
    "maximum_compact_component_error": compact_genus_error,
}
(GENUS_DATA / "summary.json").write_text(json.dumps(genus_summary, indent=2), encoding="utf-8")

fig = plt.figure(figsize=(13.2, 5.2))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot(Xg[:, 0], Xg[:, 3], Xg[:, 4], linewidth=0.8)
ax1.scatter([Xg[0, 0]], [Xg[0, 3]], [Xg[0, 4]], s=30, marker="o")
ax1.set_xlabel(r"$w$")
ax1.set_ylabel(r"$z$")
ax1.set_zlabel(r"$u$")
ax1.set_title(r"(a) Full-span Jacobi orbit, $m=1/2$")
ax1.view_init(24, 42)

ax2 = fig.add_subplot(1, 2, 2)
labels = [r"$w$", r"$x$ ($i$)", r"$y$ ($j$)", r"$z$ ($k$)", r"$u$ ($\ell$)"]
for j, label in enumerate(labels):
    ax2.plot(tg, Xg[:, j], linewidth=0.78, label=label)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel("canonical component")
ax2.set_title(r"(b) One exact period, $T=4K(1/2)$")
ax2.grid(True, alpha=0.25)
ax2.legend(ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "elliptic_genus_one_orbit.pdf", bbox_inches="tight")
fig.savefig(FIG / "elliptic_genus_one_orbit.png", dpi=360, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------------
# Appendix A: associator compensation and the 2:3:5 periodic orbit.
# -------------------------------------------------------------------------
def compensated_components(p: np.ndarray) -> np.ndarray:
    """Canonical component system at eta=1."""
    w, x, y, z, u = p
    return np.array([
        -2*x + 4*w*x,
        2*w - 2*w*w + 2*x*x,
        -3*z + 5*u*y,
        3*y + 5*u*z,
        2.5*(1 - y*y - z*z + u*u),
    ], dtype=float)


def split_q_r(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = -0.5 * mul(I, mul(I, p) + mul(p, I))
    return q, p - q


def associator_family_rhs(p: np.ndarray, eta: float) -> np.ndarray:
    q, r = split_q_r(p)
    term_q = 2.0 * mul(I, mul(q, ONE - q))
    term_rot = 1.5 * (
        mul(J, mul(K_BASIS, r) + mul(r, K_BASIS))
        - mul(mul(J, r) + mul(r, J), K_BASIS)
    )
    term_nonassoc = 2.5 * (
        ELL - mul(mul(r, ELL), r) - eta*assoc(ELL, r, r)
    )
    return term_q + term_rot + term_nonassoc


def compensated_compact(p: np.ndarray) -> np.ndarray:
    return associator_family_rhs(p, 1.0)


nr = 200_001
tr = np.linspace(0.0, 2.0*math.pi, nr)
theta2 = 2.0*tr
theta3 = 3.0*tr
theta5 = 5.0*tr
D2 = 5.0 + 4.0*np.cos(theta2)
D5 = 2.0 + np.cos(theta5)
Xr = np.column_stack((
    (1.0 + 2.0*np.cos(theta2))/D2,
    2.0*np.sin(theta2)/D2,
    math.sqrt(3.0)*np.cos(theta3)/D5,
    math.sqrt(3.0)*np.sin(theta3)/D5,
    np.sin(theta5)/D5,
))
dXr = np.vstack([compensated_components(row) for row in Xr])
dD2 = -8.0*np.sin(theta2)
dD5 = -5.0*np.sin(theta5)
Nr0 = 1.0 + 2.0*np.cos(theta2)
Nr1 = 2.0*np.sin(theta2)
Nr2 = math.sqrt(3.0)*np.cos(theta3)
Nr3 = math.sqrt(3.0)*np.sin(theta3)
Nr4 = np.sin(theta5)
dNr0 = -4.0*np.sin(theta2)
dNr1 = 4.0*np.cos(theta2)
dNr2 = -3.0*math.sqrt(3.0)*np.sin(theta3)
dNr3 = 3.0*math.sqrt(3.0)*np.cos(theta3)
dNr4 = 5.0*np.cos(theta5)
dXr_closed = np.column_stack((
    (dNr0*D2 - Nr0*dD2)/(D2*D2),
    (dNr1*D2 - Nr1*dD2)/(D2*D2),
    (dNr2*D5 - Nr2*dD5)/(D5*D5),
    (dNr3*D5 - Nr3*dD5)/(D5*D5),
    (dNr4*D5 - Nr4*dD5)/(D5*D5),
))
residual_compensated = float(np.max(np.abs(dXr - dXr_closed)))

compact_compensated_error = 0.0
split_error = 0.0
for _ in range(1000):
    p_test = rng.normal(size=5)
    q_test, _ = split_q_r(p_test)
    split_error = max(
        split_error,
        float(np.max(np.abs(q_test - np.array([p_test[0], p_test[1], 0, 0, 0])))),
    )
    compact_compensated_error = max(
        compact_compensated_error,
        float(np.max(np.abs(compensated_compact(p_test) - compensated_components(p_test)))),
    )

p0_appendix = np.array([1/3, 0.0, 1/math.sqrt(3), 0.0, 0.0])
r0_appendix = np.array([0.0, 0.0, 1/math.sqrt(3), 0.0, 0.0])
initial_associator = assoc(ELL, r0_appendix, r0_appendix)
expected_initial_associator = np.array([0.0, -1/6, 0.0, 1/6, 1/3])
initial_associator_error = float(np.max(np.abs(initial_associator - expected_initial_associator)))

eta_deformed = 0.9
sol = solve_ivp(
    lambda t, state: associator_family_rhs(state, eta_deformed),
    (0.0, 2.0*math.pi),
    p0_appendix,
    method="DOP853",
    t_eval=tr,
    rtol=1e-11,
    atol=1e-13,
    max_step=0.01,
)
if not sol.success or sol.y.shape[1] != nr:
    raise RuntimeError("The appendix deformation integration failed")
Xdef = sol.y.T
deviation = np.linalg.norm(Xdef - Xr, axis=1)
return_defect = float(np.linalg.norm(Xdef[-1] - p0_appendix))
max_deviation = float(np.max(deviation))

np.savez_compressed(
    APP_DATA / "exact_orbit_high_resolution.npz",
    t=tr,
    x=Xr,
    period=np.array(2.0*math.pi),
    eta=np.array(1.0),
)
np.savez_compressed(
    APP_DATA / "deformed_eta_0p9_high_resolution.npz",
    t=tr,
    x=Xdef,
    deviation_from_exact=deviation,
    eta=np.array(eta_deformed),
    nominal_period=np.array(2.0*math.pi),
)
pd.DataFrame(
    np.column_stack((tr[::stride], Xr[::stride])),
    columns=["t", "w", "x_i", "y_j", "z_k", "u_ell"],
).to_csv(APP_DATA / "exact_orbit_sample.csv", index=False)
pd.DataFrame(
    np.column_stack((tr[::stride], Xdef[::stride], deviation[::stride])),
    columns=["t", "w", "x_i", "y_j", "z_k", "u_ell", "deviation_from_exact"],
).to_csv(APP_DATA / "deformed_eta_0p9_sample.csv", index=False)

appendix_summary = {
    "initial_condition": p0_appendix.tolist(),
    "compensated_eta": 1.0,
    "period": 2.0*math.pi,
    "high_resolution_samples": int(nr),
    "maximum_exact_orbit_residual": residual_compensated,
    "maximum_compact_component_error": compact_compensated_error,
    "maximum_q_projection_error": split_error,
    "initial_associator": initial_associator.tolist(),
    "initial_associator_error": initial_associator_error,
    "deformed_eta": eta_deformed,
    "deformed_one_period_return_defect": return_defect,
    "deformed_maximum_deviation_from_exact": max_deviation,
}
(APP_DATA / "summary.json").write_text(
    json.dumps(appendix_summary, indent=2), encoding="utf-8"
)

fig = plt.figure(figsize=(13.2, 9.4))
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.plot(Xr[:, 0], Xr[:, 2], Xr[:, 4], linewidth=0.75)
ax1.scatter([Xr[0, 0]], [Xr[0, 2]], [Xr[0, 4]], s=30, marker="o")
ax1.set_xlabel(r"$w$")
ax1.set_ylabel(r"$y$")
ax1.set_zlabel(r"$u$")
ax1.set_title(r"(a) Compensated periodic orbit, $\eta=1$")
ax1.view_init(24, 40)

ax2 = fig.add_subplot(2, 2, 2)
for j, label in enumerate(labels):
    ax2.plot(tr, Xr[:, j], linewidth=0.7, label=label)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel("canonical component")
ax2.set_title(r"(b) Exact $2{:}3{:}5$ components")
ax2.grid(True, alpha=0.25)
ax2.legend(ncol=3, fontsize=8)

ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.plot(Xdef[:, 0], Xdef[:, 2], Xdef[:, 4], linewidth=0.75)
ax3.scatter([Xdef[0, 0]], [Xdef[0, 2]], [Xdef[0, 4]], s=30, marker="o")
ax3.scatter([Xdef[-1, 0]], [Xdef[-1, 2]], [Xdef[-1, 4]], s=35, marker="x")
ax3.set_xlabel(r"$w$")
ax3.set_ylabel(r"$y$")
ax3.set_zlabel(r"$u$")
ax3.set_title(r"(c) Same initial value, $\eta=0.9$")
ax3.view_init(24, 40)

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(tr, deviation, linewidth=0.9)
ax4.axvline(2.0*math.pi, linestyle="--", linewidth=0.8)
ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\|p_{0.9}(t)-p_*(t)\|$")
ax4.set_title(rf"(d) Orbit defect; $\|p_{{0.9}}(2\pi)-p_0\|={return_defect:.3f}$")
ax4.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(FIG / "appendix_associator_235.pdf", bbox_inches="tight")
fig.savefig(FIG / "appendix_associator_235.png", dpi=360, bbox_inches="tight")
plt.close(fig)

report = f"""Exact analytical model generation and verification
=================================================

Irreducible genus-one flow
--------------------------
Jacobi modulus: {m}
Period: {genus_period:.15f}
High-resolution samples: {ng}
Exact-orbit/component maximum residual: {residual_genus:.3e}
Compact-Pentarnion/component maximum error: {compact_genus_error:.3e}

Appendix: associator-compensated 2:3:5 flow
-------------------------------------------
Period at eta=1: {2*math.pi:.15f}
High-resolution samples: {nr}
Exact-orbit/component maximum residual: {residual_compensated:.3e}
Compact-Pentarnion/component maximum error: {compact_compensated_error:.3e}
Canonical q-projection maximum error: {split_error:.3e}
Initial-associator exact-value error: {initial_associator_error:.3e}
One-period return defect at eta={eta_deformed}: {return_defect:.12f}
Maximum deviation from exact orbit at eta={eta_deformed}: {max_deviation:.12f}
"""
(ROOT / "notes" / "exact_model_generation_report.txt").write_text(report, encoding="utf-8")
print(report)

if max(
    residual_genus,
    compact_genus_error,
    residual_compensated,
    compact_compensated_error,
    split_error,
    initial_associator_error,
) > 2e-11:
    raise SystemExit("Exact analytical model verification failed")
