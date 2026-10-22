"""Exact orthogonal rotational shadows, holonomy and double rotations."""
from __future__ import annotations
import math
import numpy as np
# Core arithmetic is shared by the two paper companions.
import sys
from pathlib import Path
_STUDY = Path(__file__).resolve().parents[3]
if str(_STUDY) not in sys.path:
    sys.path.insert(0, str(_STUDY))
from shared.algebra_core import moving_frame, adapted_to_original_matrix


def skew3(u: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(u, dtype=float)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def rodrigues3(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    K = skew3(axis)
    return math.cos(theta) * np.eye(3) + (1 - math.cos(theta)) * np.outer(axis, axis) + math.sin(theta) * K


def rotor_matrix(gamma: float, axis: str | np.ndarray, theta: float, basis: str = "adapted") -> np.ndarray:
    axis_map = {"h": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0]), "c": np.array([0, 0, 1.0])}
    u = axis_map[axis] if isinstance(axis, str) else np.asarray(axis, dtype=float)
    block = np.eye(4)
    block[:3, :3] = rodrigues3(u, theta)
    P = moving_frame(gamma)
    R = P @ block @ P.T
    if basis == "original":
        U = adapted_to_original_matrix()
        R = U @ R @ U.T
    elif basis != "adapted":
        raise ValueError("basis must be adapted or original")
    return R


def holonomy_matrix(gamma1: float, gamma2: float, epsilon: float) -> np.ndarray:
    return (
        rotor_matrix(gamma2, "c", -epsilon)
        @ rotor_matrix(gamma1, "c", -epsilon)
        @ rotor_matrix(gamma2, "c", epsilon)
        @ rotor_matrix(gamma1, "c", epsilon)
    )


def holonomy_angle(gamma1: float, gamma2: float, epsilon: float) -> float:
    beta1 = math.atan2(gamma1, math.sqrt(3))
    beta2 = math.atan2(gamma2, math.sqrt(3))
    x = abs(math.sin(beta2 - beta1)) * math.sin(epsilon / 2) ** 2
    return 4 * math.asin(min(1.0, x))


def principal_angles_so4(Q: np.ndarray) -> tuple[float, float]:
    angles = sorted(float(a) for a in np.angle(np.linalg.eigvals(Q)) if a > 1e-10)
    return (angles[0], angles[1]) if len(angles) >= 2 else (0.0, angles[0] if angles else 0.0)


if __name__ == "__main__":
    eps = math.radians(45)
    H = holonomy_matrix(0, 1, eps)
    print("holonomy angle (deg)", math.degrees(holonomy_angle(0, 1, eps)))
    print("orthogonality error", np.linalg.norm(H.T @ H - np.eye(4)))
    Q = rotor_matrix(1, "c", math.pi / 2) @ rotor_matrix(0, "b", math.pi / 2)
    print("double angles (deg)", tuple(math.degrees(a) for a in principal_angles_so4(Q)))
