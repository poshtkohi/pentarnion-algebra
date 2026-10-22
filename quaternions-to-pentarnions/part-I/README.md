# Part I — Algebra, geometry, and a spin model

Companion to *From Quaternions to Pentarnions I: Algebra, geometry, and a spin model*.

- `code/structural_invariants/`: associator tensors and invariant formulas.
- `code/rotational_geometry/`: rotations, identities and figures.
- `code/bracket_geometry/`: Jacobian determinants and equation-based geometry plots.
- `code/spin_dynamics/`: model, dense trajectories, tangent-space Lyapunov scan and figures.
- `data/`: read-only paper reference records.

From the repository root:

```bash
python quaternions-to-pentarnions/run_checks.py --part I --level full --plots
python quaternions-to-pentarnions/run_numerics.py --task spin
python quaternions-to-pentarnions/run_numerics.py --task spin-scan --threads 4
```

[Reproduction guide](docs/reproducibility.md) · [Result map](docs/result-map.md) ·
[Data guide](docs/data-guide.md) · [Numerical notes](../docs/numerical-notes.md)
