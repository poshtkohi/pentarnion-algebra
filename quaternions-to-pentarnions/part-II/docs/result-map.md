# Part II result-to-code map

| Result or evidence | Code | Evidence type |
| --- | --- | --- |
| Invariant elliptic ideal | `code/exact_solutions/symbolic_verification.py` | exact symbolic identities |
| Jacobi family, period and affine-span illustrations | `code/exact_solutions/generate_figures_and_data.py` | explicit formulas and numerical illustration |
| Two ordered-product representation | `code/exact_solutions/verify_ordered_product_representation.py` | coefficient solve and exact substitution |
| Hyperbolic invariant plane | `code/exact_solutions/verify_hyperbolic_invariant_plane.py` | exact substitution including endpoints |
| Origin characteristic polynomial | `code/riccati_dynamics/validation/verify_characteristic_polynomial.py` | exact symbolic verification |
| Static and two-protocol exponents | `code/riccati_dynamics/static_time_scan.cpp` | finite-time nearby-trajectory estimates |
| Global maxima diagram | `code/riccati_dynamics/static_bifurcation_peaks.cpp` | numerical grid experiment |
| Full spectra and dense trajectories | `full_lyapunov_spectra.cpp`, `generate_dense_riccati.cpp` in `code/riccati_dynamics/` | numerical integration |
| Periodic branches and flips | `code/riccati_dynamics/periodic_branches/` | shooting, continuation, multipliers, sampled interpolation |
| Hopf coefficient and coexisting cycles | `code/riccati_dynamics/validation/verify_section9.py` | floating-point/adaptive validation |
| Targeted spectra and divergence | `code/riccati_dynamics/validation/targeted_spectrum.cpp` | step/horizon checks |
| Forced response at omega=2 | `verify_section9.py --extended` | adaptive state/variational checks |
| Stroboscopic points | `validation/sample_archived_forcing_phase.py` | local phase-alignment integration |

The numerical runner records each program's settings and output locations.
See the [reproduction guide](reproducibility.md) for commands and
[numerical notes](../../docs/numerical-notes.md) for reference comparisons.
