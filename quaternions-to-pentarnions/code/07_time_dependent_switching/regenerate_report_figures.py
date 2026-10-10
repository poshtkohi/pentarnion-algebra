#!/usr/bin/env python3
"""Compatibility entry point for plotting the supplied Riccati/protocol data.

The filename is retained for existing callers. The current renderer reads the
study's data/ tree; generated images go to outputs/plots or --output.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Local directory for generated plots')
    args = parser.parse_args()
    env = dict(os.environ)
    if args.output:
        env['PENTARNION_OUTPUT_DIR'] = str(args.output.expanduser().resolve())
    target = Path(__file__).resolve().parents[1] / '06_riccati_dynamics/make_dense_figures.py'
    subprocess.run([sys.executable, str(target)], env=env, check=True)


if __name__ == '__main__':
    main()
