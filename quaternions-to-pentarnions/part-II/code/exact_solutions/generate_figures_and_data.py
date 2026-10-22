#!/usr/bin/env python3
"""Generate all numerical data and figures for the exact switching-integrable family.

The theorem is exact. Numerical integration and residuals are included only as
independent reproducibility checks and illustrations.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipk


def exact_solution(t: np.ndarray, gamma: float, mu: float) -> np.ndarray:
    """Return rows (w,x,y,z,u) of the exact solution."""
    m = mu * gamma
    sn, cn, dn, _ = ellipj(t, m)
    w = sn
    x = cn * dn
    y = sn**2
    z = sn * cn * dn
    u = gamma * sn**3
    return np.vstack((w, x, y, z, u))


def exact_derivative(t: np.ndarray, gamma: float, mu: float) -> np.ndarray:
    """Derivative obtained from the standard Jacobi identities."""
    m = mu * gamma
    Y = exact_solution(t, gamma, mu)
    w, x, y, z, u = Y
    dw = x
    dx = -(1.0 + m) * w + 2.0 * m * w**3
    dy = 2.0 * w * x
    dz = 1.0 - 2.0 * (1.0 + m) * w**2 + 3.0 * m * w**4
    du = 3.0 * gamma * w * z
    return np.vstack((dw, dx, dy, dz, du))


def vector_field(_t: float | np.ndarray, y: np.ndarray, gamma: float, mu: float) -> np.ndarray:
    """Five-dimensional switching-integrable vector field."""
    w, x, yy, z, u = y
    m = mu * gamma
    return np.array(
        [
            x - z + w*x + 2.0*(1.0+m)*w*z - 2.0*(1.0+m)*x*yy,
            yy - (1.0+m)*w - w**2 + 2.0*m*w*yy,
            2.0*w*x,
            1.0 - 2.0*(1.0+m)*w**2 + 3.0*mu*u*w,
            3.0*gamma*w*z,
        ],
        dtype=float,
    )


def invariants(y: np.ndarray, gamma: float, mu: float) -> np.ndarray:
    w, x, yy, z, u = y
    m = mu * gamma
    return np.vstack(
        (
            yy - w**2,
            z - w*x,
            u - gamma*w*yy,
            x**2 - m*yy**2 + (1.0+m)*yy - 1.0,
        )
    )


def period(gamma: float, mu: float) -> float:
    return float(4.0 * ellipk(mu * gamma))


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / 'outputs/exact_integrable')
    parser.add_argument("--mu", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 < args.mu < 1.0:
        parser.error("--mu must lie strictly between 0 and 1")
    root = args.output.resolve()
    figures = root / "plots"
    data_dir = root / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    mu = args.mu

    gammas = [0.0, 0.25, 0.60, 1.0]
    records: list[dict[str, float]] = []
    residual_records: list[dict[str, float]] = []

    # Figure 1: exact trajectory family in a projection that exposes the fifth direction.
    plt.figure(figsize=(8.4, 6.6))
    ax = plt.axes(projection="3d")
    for gamma in gammas:
        T = period(gamma, mu)
        t = np.linspace(0.0, T, 2400)
        Y = exact_solution(t, gamma, mu)
        ax.plot(Y[0], Y[3], Y[4], linewidth=1.0, label=fr"$\gamma={gamma:g}$")
        for idx in range(t.size):
            records.append(
                {
                    "gamma": gamma,
                    "mu": mu,
                    "t": float(t[idx]),
                    "w": float(Y[0, idx]),
                    "x": float(Y[1, idx]),
                    "y": float(Y[2, idx]),
                    "z": float(Y[3, idx]),
                    "u": float(Y[4, idx]),
                }
            )
    ax.set_xlabel("w")
    ax.set_ylabel("z (k component)")
    ax.set_zlabel("u (l component)")
    ax.set_title(fr"Exact quaternion-to-Pentarnion orbit family, $\mu={mu:g}$")
    ax.legend()
    save_figure(figures / "trajectory_family_3d.png")

    pd.DataFrame(records).to_csv(data_dir / "exact_trajectory_family.csv", index=False)

    # Figure 2: invariant plane curves x^2=(1-w^2)(1-m w^2).
    plt.figure(figsize=(8.2, 5.8))
    for gamma in gammas:
        T = period(gamma, mu)
        t = np.linspace(0.0, T, 2200)
        Y = exact_solution(t, gamma, mu)
        plt.plot(Y[0], Y[1], linewidth=1.0, label=fr"$\gamma={gamma:g}$")
    plt.xlabel("w")
    plt.ylabel("x")
    plt.title(r"Invariant curves $x^2=(1-w^2)(1-\mu\gamma w^2)$")
    plt.legend()
    plt.grid(True)
    save_figure(figures / "invariant_curves_wx.png")

    # Figures 3-5: five component time series at three representative values.
    for gamma, stem in [(0.0, "0"), (0.60, "0p60"), (1.0, "1")]:
        T = period(gamma, mu)
        t = np.linspace(0.0, T, 2200)
        Y = exact_solution(t, gamma, mu)
        plt.figure(figsize=(10.2, 5.8))
        for row, label in zip(Y, ["w", "x (i)", "y (j)", "z (k)", "u (l)"]):
            plt.plot(t, row, label=label)
        plt.xlabel("t")
        plt.ylabel("component")
        plt.title(fr"Exact one-period time series: $\gamma={gamma:g}$, $\mu={mu:g}$")
        plt.legend()
        plt.grid(True)
        save_figure(figures / f"timeseries_gamma_{stem}.png")

    # Figure 6: period law.
    gamma_grid = np.linspace(0.0, 1.0, 501)
    period_grid = np.array([period(g, mu) for g in gamma_grid])
    pd.DataFrame({"gamma": gamma_grid, "mu": mu, "period": period_grid}).to_csv(
        data_dir / "period_vs_gamma.csv", index=False
    )
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(gamma_grid, period_grid)
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$T(\gamma)=4K(\mu\gamma)$")
    plt.title(fr"Exact period deformation, $\mu={mu:g}$")
    plt.grid(True)
    save_figure(figures / "period_vs_gamma.png")

    # Figure 7: active fifth-coordinate amplitude.
    u_max = gamma_grid.copy()  # max |gamma sn^3| = gamma
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(gamma_grid, u_max)
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\max_t |u(t)|$")
    plt.title("Exact linear activation of the fifth component")
    plt.grid(True)
    save_figure(figures / "fifth_component_amplitude.png")

    # Figure 8: numerical affine-span audit via smallest singular value.
    smallest = []
    for gamma in gamma_grid:
        T = period(gamma, mu)
        t = np.linspace(0.0, T, 1600, endpoint=False)
        Y = exact_solution(t, gamma, mu).T
        centered = Y - Y.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        smallest.append(singular_values[-1] / singular_values[0])
    pd.DataFrame(
        {"gamma": gamma_grid, "relative_smallest_singular_value": smallest}
    ).to_csv(data_dir / "affine_span_audit.csv", index=False)
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(gamma_grid, smallest)
    plt.xlabel(r"$\gamma$")
    plt.ylabel("smallest / largest centered singular value")
    plt.title("Numerical audit of the affine-span transition")
    plt.grid(True)
    save_figure(figures / "affine_span_audit.png")

    # Exact residual checks at representative values.
    for gamma in gammas:
        T = period(gamma, mu)
        t = np.linspace(0.0, T, 2500)
        Y = exact_solution(t, gamma, mu)
        dY = exact_derivative(t, gamma, mu)
        rhs = np.column_stack([vector_field(ti, Y[:, i], gamma, mu) for i, ti in enumerate(t)]).T
        rhs = rhs.T
        G = invariants(Y, gamma, mu)
        residual_records.append(
            {
                "gamma": gamma,
                "mu": mu,
                "m": mu*gamma,
                "period": T,
                "max_ODE_residual": float(np.max(np.abs(dY-rhs))),
                "max_invariant_residual": float(np.max(np.abs(G))),
            }
        )
    residual_df = pd.DataFrame(residual_records)
    residual_df.to_csv(data_dir / "exact_solution_verification.csv", index=False)

    # Independent numerical integration check at gamma=0.6.
    gamma_check = 0.60
    T = period(gamma_check, mu)
    t_eval = np.linspace(0.0, T, 2500)
    initial = exact_solution(np.array([0.0]), gamma_check, mu)[:, 0]
    sol = solve_ivp(
        lambda t, y: vector_field(t, y, gamma_check, mu),
        (0.0, T),
        initial,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-13,
        max_step=T/3000.0,
    )
    exact = exact_solution(t_eval, gamma_check, mu)
    error = np.linalg.norm(sol.y-exact, axis=0)
    pd.DataFrame({"t": t_eval, "euclidean_error": error}).to_csv(
        data_dir / "numerical_vs_exact_error.csv", index=False
    )
    plt.figure(figsize=(8.4, 5.2))
    plt.semilogy(t_eval, np.maximum(error, np.finfo(float).tiny))
    plt.xlabel("t")
    plt.ylabel("Euclidean error")
    plt.title(r"Independent DOP853 check at $\gamma=0.6$")
    plt.grid(True)
    save_figure(figures / "numerical_vs_exact_error.png")

    # Summary of this numerical run.
    with (data_dir / "generation_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"mu = {mu}\n")
        handle.write(f"gamma values = {gammas}\n")
        handle.write(residual_df.to_string(index=False))
        handle.write("\n\nIndependent integration max error: ")
        handle.write(f"{float(np.max(error)):.16e}\n")

    print(residual_df.to_string(index=False))
    print(f"Independent integration maximum error: {float(np.max(error)):.3e}")
    print(f"Generated figures in {figures}")


if __name__ == "__main__":
    main()
