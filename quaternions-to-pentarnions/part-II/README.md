# Part II — Quadratic differential equations

Companion to *From Quaternions to Pentarnions II: Quadratic differential equations*.

`code/exact_solutions/` contains invariant-ideal, Jacobi, ordered-product and
hyperbolic-plane calculations. `code/riccati_dynamics/` contains static and
periodic scans, spectra, dense trajectories, continuation/refinement and adaptive
validation. `data/` contains immutable paper reference records.

```bash
python quaternions-to-pentarnions/run_checks.py --part II --level full --plots
python quaternions-to-pentarnions/run_numerics.py --task scans
python quaternions-to-pentarnions/run_numerics.py --task branches
python quaternions-to-pentarnions/run_numerics.py --task validation
```

For all numerical stages and plots of the newly computed records, use
`run_numerics.py --task all --plots`. Read [the reproduction guide](docs/reproducibility.md),
[result map](docs/result-map.md) and [data guide](docs/data-guide.md).
