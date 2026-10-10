#!/usr/bin/env python3
"""Read-only checks of the standalone switching-algebra code and data.

Checks Python syntax, the component layout, sampled algebra identities and
selected data sizes/headers. It does not rerun long numerical experiments or
certify the mathematical results.
"""
from __future__ import annotations
import argparse
import ast
import csv
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_layout() -> None:
    expected = {
        '00_core_algebra', '01_structural_invariants', '02_rotational_geometry',
        '03_exact_integrable_path', '04_bracket_quintics', '05_spin_dynamics',
        '06_riccati_dynamics', '07_time_dependent_switching',
    }
    actual = {p.name for p in (ROOT/'code').iterdir() if p.is_dir() and not p.name.startswith('.') and p.name != '__pycache__'}
    require(actual == expected, f'Code directories differ: expected {sorted(expected)}, got {sorted(actual)}')
    files = sorted((ROOT/'code').rglob('*.py')) + [ROOT/'run_audit.py', ROOT/'run_full_verification.py']
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print(f'PASS current eight-component hierarchy and Python syntax ({len(files)} sources)')


def check_core() -> None:
    core = load_module(ROOT/'code/00_core_algebra/algebra_core.py', 'algebra_core')
    e = np.eye(5)
    for gamma in (0.0, 0.3, 1.0):
        for q in e:
            require(np.allclose(core.product(gamma, e[0], q), q), 'Left identity failed')
            require(np.allclose(core.product(gamma, q, e[0]), q), 'Right identity failed')
        for j in range(1, 5):
            require(np.allclose(core.product(gamma, e[j], e[j]), -e[0]), 'Basis square failed')
        p = np.array([.7, -.2, .4, .1, -.3]); q = np.array([-.1, .5, -.4, .2, .6])
        require(np.allclose(core.conjugate(core.product(gamma,p,q)),core.product(gamma,core.conjugate(q),core.conjugate(p))), 'Conjugation failed')
        n = core.product(gamma,p,core.conjugate(p))
        require(np.allclose(n[1:],0) and math.isclose(n[0],float(p@p),rel_tol=1e-12), 'Norm identity failed')
    require(np.allclose(core.product(0.,e[1],e[2]),e[3]), 'Quaternion endpoint failed')
    require(np.allclose(core.product(1.,e[1],e[2]),(e[3]+e[4])/math.sqrt(2)), 'Pentarnion endpoint failed')
    print('PASS sampled core algebra, conjugation, norm and endpoint checks')


def check_records() -> None:
    expected = {
        'data/spin_dense/regular_timeseries.csv': 24001,
        'data/spin_dense/regular_poincare.csv': 900,
        'data/spin_dense/chaotic_timeseries.csv': 91001,
        'data/spin_dense/chaotic_poincare.csv': 3000,
        'data/spin_dense/regular_display_window.csv': 2293,
        'data/spin_dense/chaotic_display_window.csv': 10429,
        'data/riccati_dense/static_gamma_0p60_timeseries.csv': 200001,
        'data/riccati_dense/static_gamma_0p60_poincare.csv': 803,
        'data/riccati_dense/static_gamma_0p70_timeseries.csv': 60001,
        'data/riccati_dense/static_gamma_0p70_poincare.csv': 287,
        'data/riccati_dense/protocol_I_omega_0p2_timeseries.csv': 50001,
        'data/riccati_dense/protocol_I_omega_0p8_timeseries.csv': 50001,
        'data/riccati_dense/protocol_II_omega_0p2_timeseries.csv': 50001,
        'data/riccati_dense/protocol_II_omega_0p8_timeseries.csv': 50001,
        'data/riccati_dense/protocol_I_omega_0p8_dense_timeseries.csv': 230001,
        'data/riccati_dense/protocol_II_omega_0p2_dense_timeseries.csv': 230001,
    }
    for rel, count in expected.items():
        with (ROOT/rel).open('rb') as handle:
            actual = sum(1 for _ in handle)-1
        require(actual == count, f'{rel}: {actual} rows, expected {count}')
    for rel in ('data/structural/algebraic_invariants.csv','data/exact_integrable/period_vs_gamma.csv',
                'data/spin/representative_lyapunov_scan.csv','data/riccati/primary_period_doubling_cascade.csv',
                'data/time_switching/time_scan_transition_band.csv','data/time_switching/time_scan_full_range_regularized.csv'):
        with (ROOT/rel).open(newline='',encoding='utf-8-sig') as handle:
            rows = csv.reader(handle); header = next(rows,[]); first = next(rows,[])
        require(len(header)>=2 and len(first)>=2, f'Empty/malformed CSV: {rel}')
    for rel in ('code/03_exact_integrable_path/ordered_product_maps_flagship.txt',
                'data/riccati/section9_validation/symbolic_and_hopf_checks.json',
                'data/riccati/section9_validation/independent_orbit_checks.csv'):
        require((ROOT/rel).is_file() and (ROOT/rel).stat().st_size>0,f'Missing retained verification asset: {rel}')
    print(f'PASS {len(expected)} retained record sizes, CSV samples and retained verification inventory')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    check_layout(); check_core(); check_records()
    print('CODE AND DATA CHECKS PASSED. No long numerical experiments were rerun.')
    print('Scope: some Riccati generator sources are not supplied; see docs/reproducibility.md.')

if __name__=='__main__':
    try: main()
    except (OSError,ValueError,RuntimeError,SyntaxError) as exc:
        raise SystemExit(f'AUDIT FAILED: {exc}') from exc
