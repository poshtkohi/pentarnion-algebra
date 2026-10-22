# Part I reproduction guide

## Algebra and geometry

Run `run_checks.py --part I --level full --plots` from the study folder (or use
its repository-relative path). This executes associator checks, rotation
identities, middle-map determinants, all generic Jacobian determinant
calculations, and the 21 Part I image inputs.

`code/bracket_geometry/generate_figures.py` constructs the three bracket/zero-divisor
figures directly from the multiplication. Before plotting it checks all six
maps on the input circle, the ellipse parametrisation, and left/right determinant
zeros on the unit sphere. The figures are rendered directly from these equations.

## Spin trajectories

`run_numerics.py --task spin` compiles `generate_dense_spin.cpp`, runs both original
dense-trajectory recipes and checks the output sizes, norm and work residuals.
The regular case uses Omega=0.3, 300 transient and 900 retained periods, 800
steps per period. The positive-exponent reference case uses Omega=1.2, 500
transient and 3000 retained periods, 910 steps per period. Reference records
remain in `data/spin_dense/`; new records go under the runner's output directory.

## Spin tangent-space Lyapunov scan

`run_numerics.py --task spin-scan --threads 4` builds and executes
`code/spin_dynamics/lyapunov_scan.cpp`. The independent implementation uses the
manuscript's field and analytic Jacobian, RK4, unit-state projection with its
correct tangent derivative, two tangent vectors and twice-applied
Gram–Schmidt. It performs QR every 10 steps and uses an integer number of steps
per forcing period, chosen so that dt <= 0.01.

The declared recipe uses 500 discarded and 3000 accumulation periods at each of
14 frequencies. Both exponents, their sum, the step count and norm residual are
written. This makes the recomputation's settings explicit. The original
`data/spin/representative_lyapunov_scan.csv` remains the distinct paper reference.
The two tables need not coincide numerically; the
[numerical notes](../../docs/numerical-notes.md) give the differences at
Omega=0.8 and Omega=1.2.

The low-frequency duration grows when a fixed number of forcing periods is used.
A positive finite-time exponent is numerical evidence, not a rigorous chaotic-set
certificate. Step/horizon refinement and the near-zero exponent sum are diagnostic
checks, not proofs of convergence.

## Figures from generated records

The full `run_numerics.py --task all --plots` command plots its new records.
For a standalone spin run, set `PENTARNION_DATA_ROOT` to its generated Part I
`data` directory and `PENTARNION_OUTPUT_DIR` to a new plot directory before
running `code/spin_dynamics/make_dense_figures.py`. To redraw the published
reference records, leave `PENTARNION_DATA_ROOT` unset.
