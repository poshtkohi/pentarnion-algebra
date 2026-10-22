"""Exact symbolic associator calculations for the switching algebra.

Writes:
- data/algebraic_invariants.csv
- data/full_associator_tensor_adapted_basis.csv
- plots/associator_norms.png (unless --no-plots is selected)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/structural")
    parser.add_argument("--no-plots", action="store_true", help="Write symbolic data only")
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / "data"
    figures = root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        figures.mkdir(parents=True, exist_ok=True)

    gamma = sp.symbols("gamma", nonnegative=True, real=True)
    kappa = 1 / sp.sqrt(1 + gamma**2)
    sigma = gamma / sp.sqrt(1 + gamma**2)
    sqrt2, sqrt3, sqrt6 = sp.sqrt(2), sp.sqrt(3), sp.sqrt(6)

    basis = [sp.eye(5)[:, r] for r in range(5)]
    zero = sp.zeros(5, 1)
    table: dict[tuple[int, int], sp.Matrix] = {}

    def setp(r: int, c: int, value: sp.Matrix) -> None:
        table[(r, c)] = sp.Matrix(value)

    setp(0, 0, basis[0])
    for r in range(1, 5):
        setp(0, r, basis[r])
        setp(r, 0, basis[r])
        setp(r, r, -basis[0])

    setp(1, 2, kappa*basis[3] + sigma*basis[4])
    setp(2, 1, -kappa*basis[3] - sigma*basis[4])
    setp(2, 3, kappa*basis[1] + sigma*basis[4])
    setp(3, 2, -kappa*basis[1] - sigma*basis[4])
    setp(3, 1, kappa*basis[2] + sigma*basis[4])
    setp(1, 3, -kappa*basis[2] - sigma*basis[4])

    setp(4, 1, (basis[2] + basis[3]) / sqrt2)
    setp(1, 4, -(basis[2] + basis[3]) / sqrt2)
    setp(4, 2, (basis[3] + basis[1]) / sqrt2)
    setp(2, 4, -(basis[3] + basis[1]) / sqrt2)
    setp(4, 3, (basis[1] + basis[2]) / sqrt2)
    setp(3, 4, -(basis[1] + basis[2]) / sqrt2)

    def mul(x: sp.Matrix, y: sp.Matrix) -> sp.Matrix:
        out = sp.zeros(5, 1)
        for r in range(5):
            for c in range(5):
                if x[r] != 0 and y[c] != 0:
                    out += x[r] * y[c] * table.get((r, c), zero)
        return sp.simplify(out)

    def assoc(x: sp.Matrix, y: sp.Matrix, z: sp.Matrix) -> sp.Matrix:
        return sp.simplify(mul(mul(x, y), z) - mul(x, mul(y, z)))

    a = (basis[1] + basis[2] + basis[3]) / sqrt3
    b = (basis[1] - basis[2]) / sqrt2
    c = (basis[1] + basis[2] - 2*basis[3]) / sqrt6
    d = basis[4]
    adapted = [basis[0], a, b, c, d]
    labels = ["1", "a", "b", "c", "d"]

    rows: list[list[str]] = []
    full_norm_sq = sp.Integer(0)
    for r in range(1, 5):
        for s in range(1, 5):
            for t in range(1, 5):
                value = assoc(adapted[r], adapted[s], adapted[t])
                coords = [sp.simplify((adapted[q].T * value)[0]) for q in range(5)]
                full_norm_sq += sp.simplify(sum(q*q for q in coords))
                if any(q != 0 for q in coords):
                    rows.append([labels[r], labels[s], labels[t], *[sp.sstr(q) for q in coords]])
    full_norm_sq = sp.factor(sp.simplify(full_norm_sq))

    core_norm_sq = sp.Integer(0)
    for r in (1, 2, 3):
        for s in (1, 2, 3):
            for t in (1, 2, 3):
                value = assoc(basis[r], basis[s], basis[t])
                core_norm_sq += sp.simplify(sum(q*q for q in value))
    core_norm_sq = sp.factor(sp.simplify(core_norm_sq))

    ijk = assoc(basis[1], basis[2], basis[3])

    with (data_dir / "full_associator_tensor_adapted_basis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["first", "second", "third", "scalar", "a", "b", "c", "d"])
        writer.writerows(rows)

    with (data_dir / "algebraic_invariants.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["invariant", "exact_formula", "interpretation"])
        writer.writerow([
            "full_associator_frobenius_norm_squared", sp.sstr(full_norm_sq),
            "Orthogonal-isomorphism invariant of the full 5D algebra",
        ])
        writer.writerow([
            "quaternion_core_associator_norm_squared", sp.sstr(core_norm_sq),
            "Restricted to span{i,j,k}; vanishes at gamma=0",
        ])
        writer.writerow([
            "associator_[i,j,k]", " ; ".join(sp.sstr(q) for q in ijk),
            "Coordinates in basis (1,i,j,k,l)",
        ])
        writer.writerow(["automorphism_group_gamma_0", "O(2)", "Exact"])
        writer.writerow(["automorphism_group_gamma_positive", "SO(2)", "Exact for 0<gamma<=1"])
        writer.writerow(["left_middle_right_nuclei", "R 1", "Exact for 0<=gamma<=1"])

    if not args.no_plots:
        grid = np.linspace(0.0, 1.0, 600)
        full = 3.0 * (41.0*grid**2 + 31.0) / (1.0 + grid**2)
        core = 42.0 * grid**2 / (1.0 + grid**2)
        fig = plt.figure(figsize=(8.8, 5.2))
        plt.plot(grid, full, label="full 5D associator tensor")
        plt.plot(grid, core, label="active quaternion-core associator")
        plt.xlabel(r"$\gamma$")
        plt.ylabel("squared Frobenius norm")
        plt.title("Algebraic nonassociativity measures")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(figures / "associator_norms.png", dpi=190, bbox_inches="tight")
        plt.close(fig)

    print("full associator norm squared:", full_norm_sq)
    print("core associator norm squared:", core_norm_sq)
    print("nonzero adapted-basis associators:", len(rows))


if __name__ == "__main__":
    main()
