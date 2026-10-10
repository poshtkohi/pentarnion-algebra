"""Multiplication-induced Hamiltonian spin model and numerical diagnostics."""
from __future__ import annotations
import math
import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
SQRT6 = math.sqrt(6.0)
Y0 = np.array([0.45, 0.65, -0.61], dtype=float)
Y0 /= np.linalg.norm(Y0)


def gamma_fun(t: float, Omega: float) -> float:
    return 0.5 * (1 + math.sin(Omega * t))


def gamma_dot(t: float, Omega: float) -> float:
    return 0.5 * Omega * math.cos(Omega * t)


def native_inertia(g: float) -> tuple[float, float]:
    g2 = g * g
    A = 3 * (g2 * g2 + 2 * g2 + 3) / ((1 + g2) * (3 + g2))
    B = (4 * g2 + 3) / (1 + g2)
    return A, B


def native_inertia_derivative(g: float) -> tuple[float, float]:
    return (
        12 * g * (g**4 - 3) / ((g * g + 1) ** 2 * (g * g + 3) ** 2),
        2 * g / (1 + g * g) ** 2,
    )


def field(g: float) -> np.ndarray:
    return np.array([1 / math.sqrt(3 + g * g), 1 / SQRT2, 1 / SQRT6])


def rhs(t: float, y: np.ndarray, Omega: float, mu: float = 1.0) -> np.ndarray:
    g = gamma_fun(t, Omega)
    A, B = native_inertia(g)
    grad = np.array([A * y[0], B * y[1], B * y[2]]) - mu * field(g)
    return np.cross(y, grad)


def energy(t: float, y: np.ndarray, Omega: float, mu: float = 1.0) -> float:
    g = gamma_fun(t, Omega)
    A, B = native_inertia(g)
    return 0.5 * (A * y[0] ** 2 + B * (y[1] ** 2 + y[2] ** 2)) - mu * field(g) @ y


def jacobian(t: float, y: np.ndarray, Omega: float, mu: float = 1.0) -> np.ndarray:
    g = gamma_fun(t, Omega)
    A, B = native_inertia(g)
    g1, g2, g3 = mu * field(g)
    return np.array(
        [
            [0, -g3, g2],
            [(A - B) * y[2] + g3, 0, (A - B) * y[0] - g1],
            [(B - A) * y[1] - g2, (B - A) * y[0] + g1, 0],
        ]
    )


def rk4_step(t: float, y: np.ndarray, dt: float, Omega: float, mu: float = 1.0, project: bool = True) -> np.ndarray:
    k1 = rhs(t, y, Omega, mu)
    k2 = rhs(t + dt / 2, y + dt * k1 / 2, Omega, mu)
    k3 = rhs(t + dt / 2, y + dt * k2 / 2, Omega, mu)
    k4 = rhs(t + dt, y + dt * k3, Omega, mu)
    yn = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return yn / np.linalg.norm(yn) if project else yn


def simulate(Omega: float, tmax: float, dt: float = 0.01, sample: int = 10) -> np.ndarray:
    y = Y0.copy()
    rows = []
    t = 0.0
    for n in range(int(tmax / dt) + 1):
        if n % sample == 0:
            rows.append([t, *y, energy(t, y, Omega), gamma_fun(t, Omega)])
        y = rk4_step(t, y, dt, Omega)
        t += dt
    return np.asarray(rows)


if __name__ == "__main__":
    for Omega in (0.30, 1.20):
        data = simulate(Omega, 200)
        print(Omega, "norm error", np.max(np.abs(np.linalg.norm(data[:, 1:4], axis=1) - 1)))
        print(Omega, "energy range", data[:, 4].min(), data[:, 4].max())
