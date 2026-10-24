#!/usr/bin/env python3
"""Run the supplied original-paper checks in an isolated output workspace."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoints", action="store_true",
        help="also run the supplied numerical endpoint program with --quick settings",
    )
    args = parser.parse_args()
    code = ROOT / "code"
    workspace = ROOT / "outputs" / "checks"
    workspace.mkdir(parents=True, exist_ok=True)

    for path in code.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    shutil.copytree(code, workspace / "code", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbc", "*.nbi"))

    steps = [
        ["verify_direct_coordinates.py"],
        ["verify_tensor_basis.py"],
        ["genus_one_symbolic_verification.py"],
        ["rank_four_riccati_theory.py"],
        ["generate_exact_models.py"],
        ["check_reported_periodic_orbit.py"],
    ]
    if args.endpoints:
        steps.append(["validate_bracketing_endpoints.py", "--quick", "--out", "python_validation_results"])
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    summary = {"python": sys.version, "steps": [], "status": "running"}
    summary_path = workspace / "run_summary.json"
    try:
        for step in steps:
            command = [sys.executable, str(Path("code") / step[0]), *step[1:]]
            print("\nRunning " + " ".join(command), flush=True)
            result = subprocess.run(command, cwd=workspace, env=env, check=False)
            summary["steps"].append({"script": step[0], "arguments": step[1:],
                                     "returncode": result.returncode})
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if result.returncode:
                summary["status"] = "failed"
                print(f"Check failed: {step[0]} (exit {result.returncode})", file=sys.stderr)
                return result.returncode if result.returncode > 0 else 1
        summary["status"] = "passed"
        print(f"\nAll {len(steps)} requested programs completed successfully. Outputs: {workspace}", flush=True)
        return 0
    except (OSError, KeyboardInterrupt) as exc:
        summary["status"] = "interrupted_or_error"
        summary["error"] = str(exc)
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
