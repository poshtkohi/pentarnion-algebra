#!/usr/bin/env python3
"""Reintegrate the periodic point and period printed in the original paper.

The initial point is Equation (7.25), and the period is Equation (7.26).
This checks that particular eta=1 orbit; it does not search for new cycles or
certify existence using interval arithmetic.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from pentarnion_core import rhs, tangent_rhs


ROOT = Path(__file__).resolve().parents[1]
Q = np.array([0.352211456319145, 0.280943768469983, 0.580647718066596,
              0.274403530427830, 0.442578560477541], dtype=float)
T = 5.795118818486838


def main() -> None:
    identity = np.eye(5)

    def augmented(_t: float, state: np.ndarray) -> np.ndarray:
        q = state[:5]
        matrix = state[5:].reshape(5, 5)
        jacobian = np.column_stack([tangent_rhs(q, identity[:, j], 1.0) for j in range(5)])
        return np.r_[rhs(q, 1.0), (jacobian @ matrix).ravel()]

    solution = solve_ivp(augmented, (0.0, T), np.r_[Q, identity.ravel()],
                         method="DOP853", rtol=1e-12, atol=1e-14,
                         max_step=0.01, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    closure = solution.y[:5, -1] - Q
    monodromy = solution.y[5:, -1].reshape(5, 5)
    eigenvalues = np.linalg.eigvals(monodromy)
    flow = np.asarray(rhs(Q, 1.0))
    flow_residual = float(np.linalg.norm(monodromy @ flow - flow))
    neutral = int(np.argmin(np.abs(eigenvalues - 1.0)))
    transverse = np.delete(eigenvalues, neutral)
    radius = float(np.max(np.abs(transverse)))
    closure_norm = float(np.linalg.norm(closure))
    if closure_norm > 1e-8 or flow_residual > 1e-8 or radius >= 1.0:
        raise AssertionError(f"Reported-orbit check failed: closure={closure_norm}, "
                             f"flow residual={flow_residual}, transverse radius={radius}")
    output = ROOT / "data" / "bracketing_flow"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reported_periodic_orbit.npz", q=Q, T=T,
                        closure=closure, eig=eigenvalues)
    summary = {"eta": 1.0, "initial_point_source": "manuscript Equation (7.25)",
               "period_source": "manuscript Equation (7.26)",
               "phase_point": Q.tolist(), "period": T,
               "closure_norm": closure_norm, "flow_multiplier_residual": flow_residual,
               "max_transverse_multiplier": radius,
               "floquet": [{"real": float(z.real), "imag": float(z.imag),
                             "modulus": float(abs(z))} for z in eigenvalues],
               "verification": "floating-point reintegration; not an interval certificate"}
    (output / "reported_periodic_orbit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("PASS: the manuscript-seeded periodic orbit closes and has transverse multipliers inside the unit circle.")


if __name__ == "__main__":
    main()
