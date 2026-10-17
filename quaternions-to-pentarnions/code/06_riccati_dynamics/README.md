# Riccati results, exact verification and plotting

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

## Exact origin characteristic polynomial

A standalone exact verifier is supplied in `section9_validation/`:

```bash
python code/06_riccati_dynamics/section9_validation/verify_characteristic_polynomial.py
```

Run this command from `quaternions-to-pentarnions/`. It constructs the
multiplication matrix for `A_gamma`, computes its characteristic polynomial and
checks the manuscript's quadratic-times-cubic factorisation identically. It uses
SymPy only and requires no trajectory files. See the
[verification README](section9_validation/README.md#exact-characteristic-polynomial-verification)
for the arithmetic, output report and scope. This exact check is included in
`run_full_verification.py`; it does not regenerate any numerical result.
