#!/usr/bin/env python3
"""Run the supplied exact-calculation scripts in an isolated working copy.

Default: code/data checks, associator calculation, selected rotational formulas
and invariant-ideal identities. --full adds the exact ordered-product
representation and the six Jacobian determinants. Long numerical simulations
and continuation are not run by this command.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
ENV=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
         OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def run(command: list[str], cwd: Path, timeout: float) -> None:
    print('+ '+' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=ENV, check=True, timeout=timeout)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='Also recompute the ordered-product representation and determinants')
    parser.add_argument('--timeout', type=float, default=1800,
                        help='Per-command timeout in seconds (default: 1800)')
    parser.add_argument('--output', type=Path,
                        help='Keep generated results in a new directory; must not already exist')
    args=parser.parse_args()
    if args.timeout<=0: parser.error('--timeout must be positive')
    destination=args.output.expanduser().resolve() if args.output else None
    if destination and destination.exists(): parser.error('--output must be a new directory')
    run([sys.executable, str(ROOT/'run_audit.py')], ROOT, args.timeout)
    with tempfile.TemporaryDirectory(prefix='pentarnion_checks_') as tmp:
        work=Path(tmp)/'study'
        shutil.copytree(ROOT/'code', work/'code',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'regenerated'))
        commands=[
            ('01_structural_invariants', ['generate_algebraic_invariants.py', '--no-plots']),
            ('02_rotational_geometry', ['verify_exact_results.py']),
            ('03_exact_integrable_path', ['symbolic_verification.py']),
        ]
        if args.full:
            commands.extend([
                ('03_exact_integrable_path', ['verify_ordered_product_representation.py']),
                ('04_bracket_quintics', ['compute_symbolic.py']),
            ])
        for directory, command in commands:
            run([sys.executable, *command], work/'code'/directory, args.timeout)
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(work/'outputs', destination)
    print('REQUESTED EXACT-CALCULATION SCRIPTS COMPLETED; supplied data were not changed.')
    print('Scope: this is not a verification of every theorem or numerical claim.')


if __name__=='__main__':
    try:
        main()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f'CALCULATION FAILED: {exc}') from exc
