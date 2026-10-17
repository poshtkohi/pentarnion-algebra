#!/usr/bin/env python3
"""Exactly verify the Riccati characteristic-polynomial factorisation.

Source: From Quaternions to Pentarnions: geometry and nonlinear dynamics.
Basis: (1, i, j, k, ell), with actual (not reference/state-map) coefficients.
A_gamma = (-1/2, -2, 1/2, -1, -gamma/2).

The matrix is constructed from the multiplication table and compared entry by
entry with the Appendix A left-multiplication matrix. Its characteristic
polynomial is computed by SymPy, then compared coefficient by coefficient with
formula eq:riccati-characteristic and coefficients eq:riccati-bc.

Arithmetic is exact: sigma = gamma*kappa and kappa**2 = 1/(1+gamma**2).
Polynomial remainders in kappa are taken over Q(sqrt(2))(gamma); no numerical
sampling, root approximation, trajectory data or stored result is used.
This verifies the factorisation, not the Hopf coefficient or numerical dynamics.

Run from quaternions-to-pentarnions/:
    python code/06_riccati_dynamics/section9_validation/verify_characteristic_polynomial.py
Default output: outputs/riccati_validation/characteristic_polynomial_checks.txt
Use --output DIR to select a different directory. Existing reference files in
code/ and data/ are not read or overwritten by the default command.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import sys
import tempfile

try:
    import sympy as sp
except ImportError as exc:
    raise SystemExit(
        'SymPy is required. Install it with: python -m pip install sympy'
    ) from exc

GAMMA = sp.Symbol('gamma', real=True)
KAPPA = sp.Symbol('kappa', real=True)
LAMBDA = sp.Symbol('lambda')
SQRT2 = sp.sqrt(2)
HALF = sp.Rational(1, 2)
STUDY_ROOT = Path(__file__).resolve().parents[3]
REPORT_NAME = 'characteristic_polynomial_checks.txt'


class VerificationError(RuntimeError):
    """A claimed exact identity failed."""


@dataclass(frozen=True)
class VerificationResult:
    matrix: sp.Matrix
    coefficients: tuple[sp.Expr, ...]
    coefficient_residuals: tuple[sp.Expr, ...]
    quadratic: sp.Expr
    cubic: sp.Expr
    b_gamma: sp.Expr
    c_gamma: sp.Expr


def coefficient_A() -> sp.Matrix:
    """Canonical coordinates of the paper's A_gamma; no extra state scaling."""
    return sp.Matrix([-HALF, -2, HALF, -1, -GAMMA * HALF])


def multiplication_table() -> list[list[sp.Matrix]]:
    """Return the bilinear product table in the canonical five-element basis."""
    basis = [sp.eye(5)[:, index] for index in range(5)]
    table = [[sp.zeros(5, 1) for _ in range(5)] for _ in range(5)]
    for index in range(5):
        table[0][index] = basis[index]
        table[index][0] = basis[index]
    for index in range(1, 5):
        table[index][index] = -basis[0]

    def set_pair(a: int, b: int, value: sp.Matrix) -> None:
        table[a][b] = value
        table[b][a] = -value

    sigma = GAMMA * KAPPA
    set_pair(1, 2, KAPPA * basis[3] + sigma * basis[4])
    set_pair(2, 3, KAPPA * basis[1] + sigma * basis[4])
    set_pair(3, 1, KAPPA * basis[2] + sigma * basis[4])
    set_pair(1, 4, -(basis[2] + basis[3]) / SQRT2)
    set_pair(2, 4, -(basis[3] + basis[1]) / SQRT2)
    set_pair(3, 4, -(basis[1] + basis[2]) / SQRT2)
    return table


def left_matrix_from_table(p: sp.Matrix) -> sp.Matrix:
    """Construct L_p by applying the bilinear product to each basis vector."""
    if p.shape != (5, 1):
        raise ValueError('p must be a column with five canonical coefficients')
    table = multiplication_table()
    columns = []
    for column in range(5):
        image = sp.zeros(5, 1)
        for row in range(5):
            image += p[row] * table[row][column]
        columns.append(image)
    return sp.Matrix.hstack(*columns)


def appendix_left_matrix(p: sp.Matrix) -> sp.Matrix:
    """The displayed Appendix A formula, kept separate for the matrix check."""
    if p.shape != (5, 1):
        raise ValueError('p must be a column with five canonical coefficients')
    p0, p1, p2, p3, p4 = p
    k, sigma = KAPPA, GAMMA * KAPPA
    return sp.Matrix([
        [p0, -p1, -p2, -p3, -p4],
        [p1, p0, -k*p3+p4/SQRT2, k*p2+p4/SQRT2, -(p2+p3)/SQRT2],
        [p2, k*p3+p4/SQRT2, p0, -k*p1+p4/SQRT2, -(p1+p3)/SQRT2],
        [p3, -k*p2+p4/SQRT2, k*p1+p4/SQRT2, p0, -(p1+p2)/SQRT2],
        [p4, sigma*(-p2+p3), sigma*(p1-p3), sigma*(-p1+p2), p0],
    ])


def declared_factors() -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return the manuscript's two factors and its b_gamma, c_gamma."""
    g, lam = GAMMA, LAMBDA
    b = 3 * (16 + g**2 - g**4) / (8 * (1 + g**2))
    c = (44 - g**2 - 3*g**4 + SQRT2*g*(g**4 + g**2 + 194)) / (
        16 * (1 + g**2)
    )
    quadratic = lam**2 + lam + (22 + g**2) / 4
    cubic = lam**3 + sp.Rational(3, 2)*lam**2 + b*lam + c
    return quadratic, cubic, b, c


def canonical_coefficients(expressions: list[sp.Expr]) -> tuple[sp.Expr, ...]:
    """Reduce exact polynomials in kappa using (1+gamma^2)*kappa^2 = 1.

    The coefficient domain is the rational-function field Q(sqrt(2))(gamma).
    A zero remainder is an identity under the displayed relation. Since
    1+gamma^2 > 0 for real gamma, this has no real excluded parameter values.
    The physical branch kappa = 1/sqrt(1+gamma^2) is included in the relation.
    """
    field = sp.QQ.algebraic_field(SQRT2).frac_field(GAMMA)
    relation = sp.Poly((1 + GAMMA**2)*KAPPA**2 - 1, KAPPA, domain=field)
    reduced = []
    for expression in expressions:
        if expression.atoms(sp.Float):
            raise VerificationError('Floating-point input found in exact calculation')
        polynomial = sp.Poly(sp.expand(expression), KAPPA, domain=field)
        value = sp.factor(polynomial.rem(relation).as_expr())
        if value.has(KAPPA):
            raise VerificationError('Uncancelled kappa in a characteristic coefficient')
        reduced.append(value)
    return tuple(reduced)


def verify() -> VerificationResult:
    """Recompute the matrix and characteristic polynomial; raise on failure."""
    # Check the general matrix before selecting A_gamma.
    p = sp.Matrix(sp.symbols('p0:5', real=True))
    matrix_residual = left_matrix_from_table(p) - appendix_left_matrix(p)
    if any(sp.expand(entry) != 0 for entry in matrix_residual):
        raise VerificationError('Multiplication table and Appendix A matrix disagree')

    matrix = left_matrix_from_table(coefficient_A())
    # charpoly computes det(lambda I - L_A), not det(L_A - lambda I).
    raw = matrix.charpoly(LAMBDA).all_coeffs()
    computed = canonical_coefficients(raw)
    quadratic, cubic, b, c = declared_factors()
    expected = sp.Poly(quadratic*cubic, LAMBDA).all_coeffs()
    if len(computed) != 6 or len(expected) != 6:
        raise VerificationError('Expected a degree-five characteristic polynomial')
    residuals = tuple(
        sp.cancel(actual - claimed) for actual, claimed in zip(computed, expected)
    )
    for power, residual in zip(range(5, -1, -1), residuals):
        if residual != 0:
            raise VerificationError(
                f'Coefficient of lambda^{power} differs by {sp.sstr(residual)}'
            )
    return VerificationResult(matrix, computed, residuals, quadratic, cubic, b, c)


def make_report(result: VerificationResult) -> str:
    """Produce a readable exact result without a numerical approximation."""
    lines = [
        'Riccati characteristic-polynomial verification',
        'Result: PASS',
        f'Environment: Python {platform.python_version()}; SymPy {sp.__version__}',
        'Basis: (1,i,j,k,ell); all coefficients are actual canonical coefficients.',
        'A_gamma = (-1/2,-2,1/2,-1,-gamma/2).',
        'Arithmetic: exact over Q(sqrt(2))(gamma), with polynomial reduction in kappa.',
        'Relations: sigma=gamma*kappa; (1+gamma^2)*kappa^2=1.',
        'Intended branch: kappa=1/sqrt(1+gamma^2); gamma in [0,1].',
        'The resulting identity in fact holds for every real gamma.',
        'No sampled gamma values, floating-point approximations or stored results are used.',
        '',
        'General L_p constructed from the multiplication table equals Appendix A:',
        '25 of 25 entry residuals are exactly zero.',
        '',
        'Computed L_A_gamma (rows; sigma already replaced by gamma*kappa):',
    ]
    lines += [sp.sstr(list(result.matrix.row(row))) for row in range(5)]
    lines += [
        '',
        'Manuscript factors:',
        f'b_gamma = {sp.sstr(result.b_gamma)}',
        f'c_gamma = {sp.sstr(result.c_gamma)}',
        f'Quadratic = {sp.sstr(result.quadratic)}',
        f'Cubic = {sp.sstr(result.cubic)}',
        '',
        'Computed characteristic-polynomial coefficients, after exact reduction:',
    ]
    for power, coefficient in zip(range(5, -1, -1), result.coefficients):
        lines.append(f'lambda^{power}: {sp.sstr(coefficient)}')
    lines += ['', 'Coefficient residuals: computed - (Quadratic * Cubic)']
    for power, residual in zip(range(5, -1, -1), result.coefficient_residuals):
        lines.append(f'lambda^{power}: {sp.sstr(residual)}')
    lines += [
        '',
        'PASS det(lambda I_5 - L_A_gamma) = Quadratic * Cubic identically.',
        'Scope: verifies eq:riccati-characteristic and eq:riccati-bc from the stated',
        'multiplication table and A_gamma. No nonlinear Hopf coefficient, orbit,',
        'Lyapunov exponent, period-doubling branch or numerical simulation is verified.',
    ]
    return '\n'.join(lines) + '\n'


def save_report(directory: Path, report: str) -> Path:
    """Write the report atomically, only after all mathematical checks pass."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / REPORT_NAME
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='\n', dir=directory,
            prefix='.characteristic_', suffix='.tmp', delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(report)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path,
        default=STUDY_ROOT / 'outputs' / 'riccati_validation',
        help='Result directory (default: study outputs/riccati_validation)',
    )
    args = parser.parse_args()
    result = verify()
    destination = save_report(args.output.expanduser().resolve(), make_report(result))
    print('PASS multiplication table agrees with Appendix A (25 exact matrix entries).')
    print('PASS characteristic-polynomial coefficient residuals: [0, 0, 0, 0, 0, 0].')
    print('PASS exact quadratic-times-cubic factorisation for every gamma in [0,1].')
    print(f'Report: {destination}')


if __name__ == '__main__':
    try:
        main()
    except (VerificationError, OSError, ValueError, sp.PolynomialError) as exc:
        print(f'VERIFICATION FAILED: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
