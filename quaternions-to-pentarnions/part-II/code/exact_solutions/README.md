# Exact elliptic and hyperbolic solutions

Run from `quaternions-to-pentarnions/`:

```bash
python part-II/code/exact_solutions/symbolic_verification.py
```

verify_ordered_product_representation.py performs the coefficient solve; verify_hyperbolic_invariant_plane.py checks the new affine-plane proposition. generate_figures_and_data.py evaluates the Jacobi formulas and a small numerical comparison. run_all.py runs the component sequence.

The paper-specific guide is `part-II/docs/reproducibility.md` relative to the
study root. Standalone scripts write generated results below `part-II/outputs/`;
reference files in `data/`, `symbolic/` and `verification/` are inputs, not output
destinations. The common `run_checks.py` runner retains logs and keeps symbolic
work isolated.
