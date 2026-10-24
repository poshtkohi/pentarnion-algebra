# Computational guide

Companion to *A five-dimensional nonassociative algebra: Elliptic Integrability
and Bracketing-Dependent Dynamics*. This guide contains the computational
hierarchy formerly given in Appendix B of the manuscript.

## Computational code hierarchy

The reproducibility scripts for this article are available in the `code/`
directory of `pentarnion-dynamics/` within the
[Pentarnion algebra repository](https://github.com/poshtkohi/pentarnion-algebra).
Their dependency structure has four layers.

1. **Core arithmetic.** `pentarnion_core.py` implements the canonical product,
   the left and right multiplication matrices, the sparse bracketing vector
   field, its tangent equations, and the numerical integrators imported by the
   higher-level Python routines.
2. **Exact and symbolic certificates.** `rank_four_riccati_theory.py` verifies
   the six quadratic words, rank jump, first integrals, and rank-four elliptic
   example. `genus_one_symbolic_verification.py` checks the invariant ideal,
   elliptic-normal-sextic relations, compact representation, irreducibility,
   Riccati exclusions, and modulus selection. `generate_exact_models.py`
   regenerates the exact-orbit assets. The MATLAB counterparts are
   `elliptic_pentarnion_exact.m` and `associator_235_deformation.m`.
3. **Bracketing-dependent computations.** `pentarnion_bracketing_validation.m`
   performs the principal MATLAB validation. The Python sequence consists of
   `validate_bracketing_endpoints.py`, `continue_periodic_orbit.py`,
   `regenerate_bracketing_flow.py`, and `resume_bracketing_regeneration.py`;
   `plot_bracketing_flow.py` then constructs the publication figures.
4. **Final audit.** `verify_unified_package.py` independently checks the
   canonical products and the consistency of the generated numerical assets.

Thus the natural order is: exact certificates, exact-model generation, endpoint
validation, periodic-orbit continuation, long-run regeneration, figure
generation, and final audit. Scripts that read stored trajectories retain the
relative `data/` and `notes/` paths used by the original programs.

The two further supplied scripts `verify_direct_coordinates.py` and
`verify_tensor_basis.py` independently check the canonical Jacobiator-Hodge
identity, respectively using general coordinates and basis-level tensors.
All 15 supplied Python/MATLAB source files are retained unchanged. An additional
`check_reported_periodic_orbit.py` reintegrates the periodic point and period
printed in manuscript Equations (7.25) and (7.26), and checks its closure and
transverse Floquet multipliers numerically.

## Software and workspace

Install `pentarnion-dynamics/requirements.txt`. The Python dependencies are
NumPy, SciPy, SymPy, pandas, Matplotlib and Numba. The bundled MATLAB validation
program documents MATLAB R2020a or later and does not require a special toolbox.

Run from the repository root:

```bash
python -m pip install -r pentarnion-dynamics/requirements.txt
python pentarnion-dynamics/run_checks.py
```

The runner creates `pentarnion-dynamics/outputs/checks/` containing `code/`,
`data/`, `figures/` and `notes/` as needed. It runs, in order:

```text
verify_direct_coordinates.py
verify_tensor_basis.py
genus_one_symbolic_verification.py
rank_four_riccati_theory.py
generate_exact_models.py
check_reported_periodic_orbit.py
```

The rank-four and exact-model programs include numerical checks as well as
analytical formulas. Exact symbolic assertions and floating-point residuals
should be interpreted separately. A nonzero exit status stops the runner.

## Independent endpoint validation and continuation

The endpoint program does not require prior trajectory files. To run its quick
settings in the workspace:

```bash
python pentarnion-dynamics/run_checks.py --endpoints
```

For the longer settings, after preparing the workspace above:

```bash
cd pentarnion-dynamics/outputs/checks
python code/validate_bracketing_endpoints.py --out python_validation_results
python code/continue_periodic_orbit.py --periodic data/bracketing_flow/reported_periodic_orbit.npz --out periodic_continuation.csv
```

The first command writes endpoint NPZ records, a refined periodic-orbit NPZ and
a JSON summary. The second uses the manuscript-seeded orbit produced by
`check_reported_periodic_orbit.py` and writes continuation samples. The original
endpoint program estimates its own shooting seed from adjacent peaks; that
iteration can converge to a different periodic branch. Its zero exit status
is not by itself a verification of the manuscript's particular attracting cycle.
The file `periodic_orbit.npz` has keys `q`, `T`, `closure` and `eig`; it is not
interchangeable with an already sampled cycle `periodic_orbit_dense.npz`.

## Reference-record workflows

The long-run regeneration, composite-figure and complete audit scripts use a
specific `data/bracketing_flow/` record layout. In particular, the supplied
programs refer to:

- `equilibria_eta0.csv`, refined or sampled periodic-cycle records,
  `periodic_orbit_metrics.csv`, `periodic_orbit_phase_point.csv`, and
  `floquet_multipliers.csv`;
- full Lyapunov, timestep, power-spectrum, Poincare, energy-balance and
  continuation tables;
- `high_resolution/trajectory_eta0_high_resolution.npz`,
  `high_resolution/trajectory_eta1_high_resolution.npz`, and Lyapunov histories.

The original paper source archive supplied the programs and finished figures,
not these bracketing reference records. Do not substitute the switching-family
Part II data: that paper varies a different parameter. The complete composite
plotting/audit sequence requires its own records with the filenames, columns
and NPZ keys expected by the programs. Endpoint validation and the continuation
command above can be run without those reference inputs; their output schemas
are documented in the source and are not automatically converted into the
composite plotting layout.

When the required records are installed in the workspace, the original sequence
is:

```bash
python code/regenerate_bracketing_flow.py
python code/plot_bracketing_flow.py
python code/verify_unified_package.py
```

`resume_bracketing_regeneration.py` resumes later stages from saved
high-resolution trajectories; it is not a substitute for generating the
prerequisite records. The MATLAB program supplies a separate numerical
workflow with its own documented output names:

```matlab
addpath('pentarnion-dynamics/code');
results = pentarnion_bracketing_validation(false); % quick settings
% results = pentarnion_bracketing_validation(true); % publication settings
```

## Figure and result map

| Manuscript material | Program(s) |
| --- | --- |
| Canonical Jacobiator identity | `verify_direct_coordinates.py`; `verify_tensor_basis.py` |
| Quadratic words, rank jump and rank-four elliptic orbit | `rank_four_riccati_theory.py` |
| Invariant ideal, normal sextic, irreducibility and compact representation | `genus_one_symbolic_verification.py` |
| Full-dimensional elliptic solution and Appendix A | `generate_exact_models.py`; the two exact-model MATLAB files |
| Bracketing endpoint diagnostics and periodic orbit | `pentarnion_core.py`; `validate_bracketing_endpoints.py` |
| Periodic branch continuation | `continue_periodic_orbit.py` |
| Long-time bracketing records | `regenerate_bracketing_flow.py`; `resume_bracketing_regeneration.py` |
| Three composite bracketing figures | `plot_bracketing_flow.py` |
| Complete record audit | `verify_unified_package.py` |

These scripts are specific to the original fixed-algebra paper. Keep their
outputs separate from those of the Part I/II switching-family packages.
