# Reproduction guide

All commands below are run from `quaternions-to-pentarnions/` unless otherwise
stated. The code operates independently of the manuscript. No pre-rendered
images or document-build files are required.

## 1. Environment

Use Python 3.10 or later and install the packages listed in `requirements.txt`:
NumPy, SciPy, SymPy, pandas, Matplotlib and mpmath. The supplied file gives lower
version bounds, not a fully locked environment. An isolated environment is
recommended:

```bash
python -m venv .venv
```

Activate `.venv` using the command appropriate for the operating system, then:

```bash
python -m pip install -r requirements.txt
```

A C++17 compiler is needed only for `generate_dense_spin.cpp`. The `.m` files
under `code/02_rotational_geometry/matlab/` are independent MATLAB demonstrations.
No external document renderer is required for Python plot labels.

## 2. Organisation

| Relative path | Contents |
| --- | --- |
| `code/00_core_algebra/` | Multiplication, conjugation, associator, moving frame, Riccati vector field and tangent action |
| `code/01_structural_invariants/` | Exact associator calculations |
| `code/02_rotational_geometry/` | Rotation matrices, symbolic checks, demonstrations and optional plots |
| `code/03_exact_integrable_path/` | Invariant-ideal identities, ordered-product maps and elliptic solution calculations |
| `code/04_bracket_quintics/` | Six quadratic maps and their Jacobian determinants |
| `code/05_spin_dynamics/` | Spin equations, dense trajectory generator and plotting |
| `code/06_riccati_dynamics/` | Plots of the supplied Riccati/protocol results |
| `code/06_riccati_dynamics/section9_validation/` | Plots of the additional supplied validation results |
| `code/07_time_dependent_switching/` | Protocol documentation and a plotting entry point |
| `data/` | Supplied numerical and exact-symbolic results |
| `outputs/` | Created locally for generated results; not distributed or tracked |

The component `04_bracket_quintics` computes the six quadratic maps and their
Jacobian determinant polynomials.

## 3. Read-only code and data checks

```bash
python run_audit.py
```

The script checks the eight code-directory names, Python syntax, sampled identity
and conjugation/norm relations, selected CSV headers, sixteen trajectory/section
sizes, and the presence of several exact and numerical result files. It does
not use a checksum manifest, access the network, regenerate plots, run numerical
campaigns, or inspect a manuscript.

A successful check establishes only these stated tests. It is not interval
validation or an audit of every mathematical proof.

## 4. Exact symbolic calculations

The following entry points use an isolated temporary copy of the source:

```bash
python run_full_verification.py
python run_full_verification.py --full --output outputs/exact_checks
```

The default runs the read-only checks, recomputes the associator formulas,
evaluates selected rotational expressions and verifies the invariant-ideal
identities. `--full` additionally solves the ordered-product coefficient system
and computes the six determinant polynomials. `--output` must name a directory
that does not already exist. `--timeout SECONDS` changes the per-command timeout
(default 1800 seconds).

Individual programs can also be run directly. Their defaults write generated
results below `outputs/`, leaving supplied reference files unchanged:

```bash
python code/01_structural_invariants/generate_algebraic_invariants.py --no-plots
python code/02_rotational_geometry/verify_exact_results.py
python code/03_exact_integrable_path/symbolic_verification.py
python code/03_exact_integrable_path/verify_ordered_product_representation.py
python code/04_bracket_quintics/compute_symbolic.py
```

The associator generator evaluates its sums exactly; its CSV also repeats the
stated automorphism-group and nuclei labels. Those labels are not independently
proved by that generator. The rotational verifier evaluates particular matrices
and residual identities, not every rotational theorem. The exact ordered-product
calculation is over rational functions; its result should be interpreted with
the parameter assumptions of that construction.

Reference expressions are supplied in `data/structural/`,
`data/exact_integrable/`, `code/02_rotational_geometry/verification/`,
`code/03_exact_integrable_path/ordered_product_maps_flagship.txt`, and
`code/04_bracket_quintics/symbolic/`.

## 5. Elliptic-solution numerical illustration

```bash
python code/03_exact_integrable_path/generate_figures_and_data.py
```

This computes the specified elliptic solutions, periods and numerical residuals,
and writes data plus optional plots beneath `outputs/exact_integrable/`.
`--mu` changes the modulus-scaling coefficient; use the mathematical domain
`0 < mu < 1`. `--output NEW_DIR` changes the output location.

The `run_all.py` entry point in this component also runs both symbolic verifiers
before the numerical illustration. This is distinct from rerunning the spin or
Riccati simulations.

## 6. Spin simulations

For a short Python demonstration:

```bash
python code/05_spin_dynamics/mechanical_spin.py
```

For the supplied dense spin generator, first create `outputs/bin`, then compile:

```bash
c++ -O3 -std=c++17 code/05_spin_dynamics/generate_dense_spin.cpp -o outputs/bin/generate_dense_spin
```

On Linux/macOS run:

```bash
./outputs/bin/generate_dense_spin outputs/spin_dense
```

On Windows use the executable name produced by the selected C++ toolchain.
The command takes an optional output-directory argument. It generates both
cases with the original integration formulas, steps and horizons:

| Frequency | Steps per forcing period | Transient periods | Retained periods | Trajectory samples | Stroboscopic samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 800 | 300 | 900 | 24,001 | 900 |
| 1.20 | 910 | 500 | 3,000 | 91,001 | 3,000 |

The C++ program computes state trajectories and work-identity diagnostics; it
is not a generator of the supplied Lyapunov scan. Its output directory defaults
to `outputs/spin_dense` relative to the working directory. The supplied `data/spin_dense/` files are not overwritten.

## 7. Riccati data and the two switching protocols

`code/00_core_algebra/switching_algebra_model.py` supplies the algebra product,
coefficient selection, static Riccati vector field, exact tangent action and the
time-dependent reference-coordinate vector field.

The protocols are:

- Protocol I: gamma(t) = 0.68 + 0.08 sin(omega t).
- Protocol II: gamma(t) = 0.525 + 0.475 sin(omega t).

Their frequency scans are in `data/time_switching/`. Static scans and periodic
continuation results are in `data/riccati/`; dense trajectories and their
integration metadata are in `data/riccati_dense/`.

The two long protocol records contain 230,001 samples over 1,150 retained time
units, after a transient of 1,200 units, with step and sample interval 0.005.
The static gamma=0.6 and gamma=0.7 records contain 200,001 and 60,001 samples,
respectively, at interval 0.01. The adjacent metadata files specify their time
windows and section-crossing counts.

**Source availability:** continuation and validation results are supplied,
but several generating programs are not included. In particular, this
repository does not include the complete static/frequency-scan drivers,
periodic-branch continuation drivers, or the additional validation drivers
`run_checks.py`, `verify_section9.py`, `targeted_spectrum.cpp`,
`sample_archived_forcing_phase.py` and `localise_archived_events.py`.
The supplied results can be inspected and plotted; rerunning these calculations
requires their generating programs.

`data/riccati/section9_validation/` contains the supplied first-Lyapunov-coefficient
result, selected periodic-orbit checks, targeted spectra and Protocol II omega=2
response data. Its execution metadata identifies the runs that produced those
results. The recorded source paths document those runs; they are not runtime
paths for the supplied programs.

The first-Lyapunov-coefficient convention is recorded in
`symbolic_and_hopf_checks.json`: f = Lx + B2(x,x)/2, unit critical eigenvector,
adjoint pairing one, and l1 = Re(G21)/(2 omega). Finite-time Lyapunov estimates,
Floquet checks and scan failures are distinct types of numerical results.
In particular, a failed integration is not itself proof of solution blow-up.
The retained Protocol II omega=2 response data must not be merged silently with
the original failed frequency-scan entry.

## 8. Optional plotting from supplied data

```bash
python code/05_spin_dynamics/make_dense_figures.py
python code/06_riccati_dynamics/make_dense_figures.py
```

Both write to `outputs/plots/`. The Riccati renderer invokes the included
`section9_validation/make_validation_figures.py` and needs no external
validation program to plot the supplied results.

For a different output directory, set `PENTARNION_OUTPUT_DIR` before running
these scripts. The compatibility entry point also accepts an argument directly:

```bash
python code/07_time_dependent_switching/regenerate_report_figures.py --output outputs/protocol_plots
```

The rotational illustration program `code/02_rotational_geometry/build_assets.py`
can be run on its own; it reads `data/spin/` and writes only generated images.
It includes an illustrative four-step rotation cycle that is not a substitute
for checking a differently specified cycle. It reads the supplied representative
spin-frequency table rather than overwriting that table with embedded values.

The stored trajectory samples are not smoothed or replaced. The supplied
plotting routines sometimes select a time window or decimate uniformly for
rendering; the complete source CSV remains available. Generated images and
local numerical outputs are excluded from version control.

## 9. Citation

Alireza Poshtkohi and Susanne Pumplün,
*From Quaternions to Pentarnions: geometry and nonlinear dynamics*.

See [CITATION.cff](../CITATION.cff). Record the Git commit or release used and
the software versions installed for each calculation. The bounds in
`requirements.txt` do not specify a fully locked environment.
