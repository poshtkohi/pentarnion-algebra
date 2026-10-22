# Rotational geometry

Run from `quaternions-to-pentarnions/`:

```bash
python part-I/code/rotational_geometry/verify_exact_results.py
```

Selected symbolic formulas; rotor_geometry.py supplies numerical rotation matrices. build_assets.py creates illustrative figures and reads the earlier spin reference data. The MATLAB examples are supplementary and were not run here.

The paper-specific guide is `part-I/docs/reproducibility.md` relative to the
study root. Standalone scripts write generated results below `part-I/outputs/`;
reference files in `data/`, `symbolic/` and `verification/` are inputs, not output
destinations. The common `run_checks.py` runner retains logs and keeps symbolic
work isolated.
