#!/usr/bin/env python3
"""Compute the six full 5D Jacobian determinant quintics.

Algebra:
    basis {1,i,j,k,l}; A=i
    ij = c k + s l, jk = c i + s l, ki = c j + s l
    c=1/sqrt(1+gamma^2), s=gamma/sqrt(1+gamma^2)
    i*l=-(j+k)/sqrt(2), etc.; reversed products have opposite signs.

For exact generic factorization, the script uses L=l/sqrt(2) and a rational
parameter tau.  In that basis:
    c=(1-2*tau^2)/(1+2*tau^2)
    sqrt(2)*s=4*tau/(1+2*tau^2)
    gamma=2*sqrt(2)*tau/(1-2*tau^2)

The script writes expanded determinants, exact factorization summaries, and
quaternion-endpoint factorizations. It also verifies the exact nonzero
Jacobian values Q3(1+ell; gamma)=Q4(1+ell; gamma)=4*gamma/(1+gamma**2).
Here code indices 3,4 match manuscript map indices 3,4. In the scaled basis,
1+ell has coordinates (1,0,0,0,sqrt(2)), not (1,0,0,0,1).
Use --nonzero-only to check these two values without computing the full
determinant polynomials. The script does not classify projective singularities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import sympy as sp
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix


def structure_tensor_scaled(tau: sp.Symbol) -> sp.MutableDenseNDimArray:
    """Return structure constants in basis {1,i,j,k,L=l/sqrt(2)}."""
    c = (1 - 2 * tau**2) / (1 + 2 * tau**2)
    S = 4 * tau / (1 + 2 * tau**2)  # coefficient of L in ij

    C = sp.MutableDenseNDimArray.zeros(5, 5, 5)
    for a in range(5):
        C[0, a, a] = 1
        C[a, 0, a] = 1
    for a in (1, 2, 3):
        C[a, a, 0] = -1
    C[4, 4, 0] = -sp.Rational(1, 2)

    def put(a: int, b: int, terms: dict[int, sp.Expr]) -> None:
        for q, value in terms.items():
            C[a, b, q] = value

    put(1, 2, {3: c, 4: S})
    put(2, 1, {3: -c, 4: -S})
    put(2, 3, {1: c, 4: S})
    put(3, 2, {1: -c, 4: -S})
    put(3, 1, {2: c, 4: S})
    put(1, 3, {2: -c, 4: -S})

    half = sp.Rational(1, 2)
    put(1, 4, {2: -half, 3: -half})
    put(4, 1, {2: half, 3: half})
    put(2, 4, {3: -half, 1: -half})
    put(4, 2, {3: half, 1: half})
    put(3, 4, {1: -half, 2: -half})
    put(4, 3, {1: half, 2: half})
    return C


def multiply(a: sp.Matrix, b: sp.Matrix, C: sp.MutableDenseNDimArray) -> sp.Matrix:
    return sp.Matrix(
        [
            sum(a[i] * b[j] * C[i, j, q] for i in range(5) for j in range(5))
            for q in range(5)
        ]
    )


def left_matrix(a: sp.Matrix, C: sp.MutableDenseNDimArray) -> sp.Matrix:
    eye = sp.eye(5)
    return sp.Matrix.hstack(*[multiply(a, eye[:, j], C) for j in range(5)])


def right_matrix(a: sp.Matrix, C: sp.MutableDenseNDimArray) -> sp.Matrix:
    eye = sp.eye(5)
    return sp.Matrix.hstack(*[multiply(eye[:, j], a, C) for j in range(5)])


def domain_matrix(M: sp.Matrix, domain) -> DomainMatrix:
    rows = [
        [domain.from_sympy(sp.cancel(sp.expand(M[i, j]))) for j in range(M.cols)]
        for i in range(M.rows)
    ]
    return DomainMatrix.from_list(rows, domain)


def total_degree(f) -> int:
    return max(sum(monomial) for monomial, _ in f.terms()) if f else -1


def build_jacobians(tau: sp.Symbol, variables: Iterable[sp.Symbol]):
    w, x, y, z, v = variables
    C = structure_tensor_scaled(tau)
    p = sp.Matrix([w, x, y, z, v])
    A = sp.Matrix([0, 1, 0, 0, 0])

    Lp, Rp = left_matrix(p, C), right_matrix(p, C)
    LA, RA = left_matrix(A, C), right_matrix(A, C)
    Ap, pA = multiply(A, p, C), multiply(p, A, C)

    jacobians = [
        LA * (Rp + Lp),
        Rp * LA + left_matrix(Ap, C),
        right_matrix(Ap, C) + Lp * LA,
        Rp * RA + left_matrix(pA, C),
        right_matrix(pA, C) + Lp * RA,
        RA * (Rp + Lp),
    ]
    return [J.applyfunc(lambda e: sp.cancel(sp.expand(e))) for J in jacobians]


def verify_nonzero_jacobians(
    output: Path,
    tau: sp.Symbol,
    variables: tuple[sp.Symbol, ...],
    jacobians: list[sp.Matrix],
    generic_determinants: dict[int, sp.Expr] | None = None,
) -> None:
    """Verify the two middle-map determinants at the actual point 1+ell.

    Determinants are recomputed after substituting the point, over the exact
    field Q(sqrt(2))(tau). A full run also checks agreement with the freshly
    computed generic determinant polynomials. Any nonzero residual is fatal.
    """
    sqrt2 = sp.sqrt(2)
    point = dict(zip(variables, (1, 0, 0, 0, sqrt2)))
    kappa = (1 - 2 * tau**2) / (1 + 2 * tau**2)
    sigma = 2 * sqrt2 * tau / (1 + 2 * tau**2)
    gamma = 2 * sqrt2 * tau / (1 - 2 * tau**2)
    expected = sp.cancel(4 * kappa * sigma)
    gamma_residual = sp.cancel(expected - 4 * gamma / (1 + gamma**2))
    if gamma_residual != 0:
        raise RuntimeError(f"Nonzero-witness parameter check failed: {gamma_residual}")

    field = QQ.algebraic_field(sqrt2).frac_field(tau)
    lines = [
        "Exact nonzero Jacobian checks for the two middle-position maps",
        "Arithmetic: exact over Q(sqrt(2))(tau); no floating-point sampling.",
        "",
        "Code indices (1,2,3,4,5,6) correspond to manuscript maps (M0,M2,M3,M4,M5,M1).",
        "M3(p) = p*(i*p); M4(p) = (p*i)*p; the parentheses are retained.",
        "Code Q3 and Q4 therefore equal manuscript Q_3 and Q_4, respectively.",
        "Basis: (1,i,j,k,L), with L=ell/sqrt(2).",
        "Point: 1+ell; scaled coordinates (w,x,y,z,v)=(1,0,0,0,sqrt(2)).",
        "kappa=(1-2*tau**2)/(1+2*tau**2)",
        "sigma=2*sqrt(2)*tau/(1+2*tau**2)",
        "gamma=2*sqrt(2)*tau/(1-2*tau**2)",
        "Expected value in tau: 8*sqrt(2)*tau*(1-2*tau**2)/(1+2*tau**2)**2",
        "Expected value in gamma: 4*gamma/(1+gamma**2)",
        f"Parameter-identity residual: {gamma_residual}",
        "For gamma>=0 use 0<=tau<1/sqrt(2); gamma in [0,1] corresponds to",
        "tau in [0,1-1/sqrt(2)]. The verified value is positive for gamma>0.",
    ]
    for index in (3, 4):
        matrix_at_point = jacobians[index - 1].subs(point).applyfunc(sp.cancel)
        determinant = field.to_sympy(domain_matrix(matrix_at_point, field).det())
        determinant = sp.factor(sp.radsimp(determinant))
        residual = sp.cancel(determinant - expected)
        if residual != 0:
            raise RuntimeError(f"Q{index}(1+ell) check failed: residual={residual}")

        lines.extend([
            "", f"Code Q{index} / manuscript Q_{index}:",
            "Jacobian at 1+ell in the scaled basis (rows):",
            *[sp.sstr(list(matrix_at_point.row(r))) for r in range(5)],
            f"Determinant: {sp.sstr(determinant)}",
            f"Residual against expected value: {residual}",
        ])
        if generic_determinants is not None:
            generic_value = generic_determinants[index].subs(point)
            generic_residual = sp.cancel(generic_value - determinant)
            if generic_residual != 0:
                raise RuntimeError(
                    f"Q{index} generic/specialized determinant mismatch: {generic_residual}"
                )
            lines.append(f"Residual against generic determinant polynomial: {generic_residual}")
        else:
            lines.append("Generic determinant comparison: not requested (--nonzero-only).")
        lines.append("PASS")
        print(f"PASS Q{index}(1+ell; gamma) = 4*gamma/(1+gamma**2)", flush=True)

    lines.extend([
        "", "BOTH EXACT POINT CHECKS PASSED.",
        "Each determinant polynomial is therefore not identically zero for gamma>0.",
        "This does not assert nonvanishing at every point, or equality of Q3 and Q4",
        "as full polynomials. The point values above both vanish at gamma=0.",
    ])
    output.mkdir(parents=True, exist_ok=True)
    (output / "nonzero_jacobian_checks.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def compute(output: Path, *, nonzero_only: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tau = sp.symbols("tau")
    w, x, y, z, v = sp.symbols("w x y z v")
    variables = (w, x, y, z, v)

    jacobians = build_jacobians(tau, variables)
    if nonzero_only:
        verify_nonzero_jacobians(output, tau, variables, jacobians)
        print(f"Results written to: {output.resolve()}")
        return

    coefficient_field = QQ.frac_field(tau)
    polynomial_ring = coefficient_field.poly_ring(*variables)

    det_expr: dict[int, sp.Expr] = {}
    factor_summary: dict[str, object] = {}

    for index, J in enumerate(jacobians, start=1):
        print(f"Computing Q{index}...", flush=True)
        determinant = domain_matrix(J, polynomial_ring).det()
        expression = polynomial_ring.to_sympy(determinant)
        det_expr[index] = expression
        coefficient, factors = determinant.factor_list()

        factor_summary[str(index)] = {
            "coefficient": str(coefficient),
            "factors": [
                {
                    "expression": str(polynomial_ring.to_sympy(factor)),
                    "exponent": exponent,
                    "degree": total_degree(factor),
                    "terms": len(factor.terms()),
                }
                for factor, exponent in factors
            ],
        }

        (output / f"Q{index}_generic_tau.txt").write_text(
            f"Q_{index}=det(J_{index})\n\n{sp.sstr(expression)}\n",
            encoding="utf-8",
        )

    (output / "generic_factorization_summary.json").write_text(
        json.dumps(factor_summary, indent=2), encoding="utf-8"
    )

    # Exact conjugation relations.
    conjugate = {w: w, x: -x, y: -y, z: -z, v: -v}
    assertions = {
        "Q6_plus_Q1": sp.expand(det_expr[6] + det_expr[1]),
        "Q5_plus_Q2_conjugate": sp.expand(det_expr[5] + det_expr[2].subs(conjugate)),
        "Q4_plus_Q3_conjugate": sp.expand(det_expr[4] + det_expr[3].subs(conjugate)),
    }
    if not all(value == 0 for value in assertions.values()):
        raise RuntimeError(f"A determinant symmetry check failed: {assertions}")

    # Exact nonzero values required for the two middle-position maps.
    verify_nonzero_jacobians(output, tau, variables, jacobians, det_expr)

    # Exact gamma=0 (tau=0) factorization over Q.
    rational_ring = QQ.poly_ring(*variables)
    endpoint_lines: list[str] = []
    for index in range(1, 7):
        endpoint_expr = sp.expand(sp.cancel(det_expr[index].subs(tau, 0)))
        endpoint_poly = rational_ring.from_sympy(endpoint_expr)
        coefficient, factors = endpoint_poly.factor_list()
        if not factors:
            factor_text = "0"
        else:
            pieces = []
            for factor, exponent in factors:
                text = f"({rational_ring.to_sympy(factor)})"
                if exponent != 1:
                    text += f"^{exponent}"
                pieces.append(text)
            factor_text = f"{coefficient} * " + " * ".join(pieces)
        endpoint_lines.append(f"Q_{index}|_(gamma=0) = {factor_text}")

    (output / "gamma0_factorizations.txt").write_text(
        "\n\n".join(endpoint_lines) + "\n", encoding="utf-8"
    )

    print("Computed all six determinant quintics.")
    print("Generic factorization classes:")
    for index in range(1, 7):
        factors = factor_summary[str(index)]["factors"]
        print(f"  Q{index}: {[(f['degree'], f['exponent']) for f in factors]}")
    print(f"Results written to: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs/bracket_maps",
        help="Directory for exact symbolic results.",
    )
    parser.add_argument(
        "--nonzero-only",
        action="store_true",
        help="Verify Q3 and Q4 at 1+ell without computing full determinant polynomials.",
    )
    args = parser.parse_args()
    compute(args.output, nonzero_only=args.nonzero_only)


if __name__ == "__main__":
    main()
