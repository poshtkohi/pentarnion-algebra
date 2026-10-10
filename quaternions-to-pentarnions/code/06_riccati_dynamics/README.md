# Riccati results and plotting

This folder contains the current renderer for the supplied static scans,
periodic branches, trajectories and both forcing protocols. From `quaternions-to-pentarnions/`:

```bash
python code/06_riccati_dynamics/make_dense_figures.py
```

The routine reads `data/riccati/`, `data/riccati_dense/` and
`data/time_switching/`, then runs the included additional-validation plotter.
Generated images go to `outputs/plots/`. No manuscript or pre-existing image
is needed. This is plotting, not a new continuation or simulation campaign.

The algebra-valued differential equation and tangent action are implemented in
`code/00_core_algebra/switching_algebra_model.py`. Several original numerical
drivers are not included in this distribution. See `docs/reproducibility.md`
from `quaternions-to-pentarnions/` for the exact coverage and limitations.
