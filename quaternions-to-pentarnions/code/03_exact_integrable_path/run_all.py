#!/usr/bin/env python3
"""Run the exact symbolic certificates and regenerate all figures/data."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE
commands = [
    [sys.executable, str(HERE/'symbolic_verification.py')],
    [sys.executable, str(HERE/'verify_ordered_product_representation.py')],
    [sys.executable, str(HERE/'generate_figures_and_data.py'), '--output', str(HERE.parents[1]/'outputs/exact_integrable')],
]
for command in commands:
    print('+', ' '.join(command))
    subprocess.run(command, check=True)
print('All exact checks and figure generation completed successfully.')
