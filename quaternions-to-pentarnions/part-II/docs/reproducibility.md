# Part II reproduction guide

## Exact calculations

`run_checks.py --part II --level full` verifies the invariant ideal, hyperbolic
invariant plane, characteristic polynomial and ordered-product coefficient
system. The ordered-product solve reports generic rank/nullity; the manuscript's
analytical argument supplies the uniform kernel-dimension statement. The
plane check includes gamma=0 and the explicit endpoint hyperbolic solution.

## Numerical drivers

Use the common `run_numerics.py` runner so compilation, paths, reference-data
protection and logs are handled consistently. The individual stages are:

| Task | Programs | Recipe |
| --- | --- | --- |
| `scans` | `static_time_scan.cpp` | 201 static values; 15 frequencies for each periodic protocol |
| `peaks` | `static_bifurcation_peaks.cpp` | 551 gamma values; at most 150 post-transient maxima each |
| `spectra` | `full_lyapunov_spectra.cpp` | 12 full spectra; state and full variational system |
| `dense` | `generate_dense_riccati.cpp` | two static and six periodically forced dense records |
| `branches` | `periodic_branches/*.cpp`, `summarise_branches.py` | original base branches; 11 explicit seeded refinements; Floquet tables, crossings and orbit samples |
| `validation` | `validation/verify_section9.py`, `targeted_spectrum.cpp`, phase/event scripts | symbolic/Hopf checks, seven adaptive orbit checks, three targeted spectra and forced response |

The recipes and parameter values are in `configs/paper_settings.json`
and `configs/refinement_recipes.json` at the common study root.

### Fixed-parameter and periodic scans

The static scan uses RK4 dt=0.01, 300 discarded and 500 accumulation time units.
Both periodic scans use reference coordinates, RK4 dt=0.005, 400 discarded and
800 accumulation time units. Nearby trajectories are reset every 0.1 time unit,
with initial separation 1e-8. The coordinate correction -gamma_dot*q4/gamma is
included. Full spectra use dt=0.005, 1000 discarded and 2000 accumulation time
units. The peaks scan uses dt=0.01 and 1000+500 time units.

The retained `blowup` column is a legacy **numerical stopping flag**, not evidence
of a mathematically unbounded solution. Failed integrations have no valid
Lyapunov estimate. In particular, Protocol II at omega=2 is examined independently
with adaptive integration. At omega=0.01, an 800-unit accumulation covers only
about 1.27 periods; it does not establish asymptotic convergence.

### Periodic branches and refinements

The C++ drivers perform shooting with a phase condition and simultaneous
variational integration, then natural-parameter continuation. Refined sweeps use
explicit starting orbits from `configs/refinement_recipes.json`; these are
numerical seeds, not asserted exact periodic points. The equations are solved
again at every requested parameter. The declared dt is 0.0005 for ordinary
refinements and 0.0015 for the P32 refinements.

`summarise_branches.py` recomputes multipliers from generated monodromy matrices,
selects the neutral direction by alignment with the flow, and interpolates
actual sampled -1 crossings. It does not hard-code the six reported event values.
Its P32 crossing is not a continued P64 orbit. Natural-parameter endpoints do not
prove a fold or a closed isola; interpolants have no rigorous error certificate.

### Adaptive and targeted checks

`verify_section9.py --extended` independently builds the field, verifies the
characteristic polynomial, computes the floating-point Hopf coefficient,
reintegrates seven supplied periodic seeds using DOP853 and checks Protocol II
at omega=2 in both coordinate systems. These periodic-seed checks intentionally
use the paper reference seeds; the continuation stage separately generates new
ones. Three targeted gamma=0.6 spectra use twice-applied modified Gram–Schmidt
and compare their sum with the mean divergence.

The forcing-phase helper integrates from the preceding stored sample. In a full
`all` run it uses the **new dense records**; a standalone validation run uses the
paper reference records. These are derived phase diagnostics, not independent
long-time integrations.

## Reference versus recomputed records

The recorded build profile uses the documented `-march=native` flags for the
three scan/spectrum sources. In the recorded environment this reproduces their
reference tables; the portable build gives different long-time values.
Floating-point operation ordering, compilation and sensitivity can affect these
experiments. See the [numerical notes](../../docs/numerical-notes.md) for the
comparison and interpretation. Generated records do not overwrite reference
data. Neither a completed run nor a positive finite-time exponent constitutes
an interval proof of a chaotic attractor.

Use `run_checks.py --plots` to redraw the manuscript reference figures; use
`run_numerics.py --task all --plots` to plot the new experiment records.
