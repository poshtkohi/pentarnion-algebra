# Structural invariants

Run from `quaternions-to-pentarnions/`:

```bash
python part-I/code/structural_invariants/generate_algebraic_invariants.py --no-plots
```

Exact associator calculation. Labels for nuclei and automorphism groups in the generated CSV are not independent proofs of those results.

The paper-specific guide is `part-I/docs/reproducibility.md` relative to the
study root. Standalone scripts write generated results below `part-I/outputs/`;
reference files in `data/`, `symbolic/` and `verification/` are inputs, not output
destinations. The common `run_checks.py` runner retains logs and keeps symbolic
work isolated.
