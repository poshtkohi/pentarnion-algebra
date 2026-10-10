# Data guide

This guide describes the numerical and exact-symbolic results in `data/`,
including their coordinate conventions and file formats.

## Coordinates

The five algebra coordinates are ordered `(1, i, j, k, ell)`.
The names `(w, x, y, z, u)` and `(p0, p1, p2, p3, p4)` denote these same five
positions in different result files. In time-dependent Riccati data,
`u_active` is the actual coefficient of `ell`, while `q4_hidden` is the reference
coordinate: `u_active = gamma * q4_hidden`.

In spin records, `(y_h, y_b, y_c)` are the three moving-basis state components.
Columns `(a, b, c, d)` give the corresponding four-dimensional pure-space
coordinates; `d` is the `ell` coefficient. `H` or `energy` is the Hamiltonian
value, not the squared state norm.

## Result groups

| Folder | Contents |
| --- | --- |
| `data/structural/` | Associator sums and nonzero adapted-basis tensor entries |
| `data/exact_integrable/` | Elliptic trajectories, periods, residuals, affine-span calculations and invariant-ideal identities |
| `data/spin/` | Earlier spin trajectories, work diagnostics and representative frequency estimates |
| `data/spin_dense/` | Long spin records, fixed-phase samples, display windows and integration metadata |
| `data/riccati/` | Fixed-parameter scans, spectrum results and branch summaries |
| `data/riccati/periodic_branches/` | Continuation samples, selected periodic orbits and Floquet data |
| `data/riccati/section9_validation/` | Additional supplied numerical checks and phase-aligned samples |
| `data/riccati_dense/` | Long static and time-dependent Riccati trajectories and metadata |
| `data/time_switching/` | The two protocol frequency scans |

## Principal record sizes

| Record | Time-series samples | Section samples |
| --- | ---: | ---: |
| Spin, frequency 0.30 | 24,001 | 900 |
| Spin, frequency 1.20 | 91,001 | 3,000 |
| Static Riccati, gamma=0.60 | 200,001 | 803 |
| Static Riccati, gamma=0.70 | 60,001 | 287 |
| Protocol I, omega=0.80, long record | 230,001 | Separate fixed-phase result supplied |
| Protocol II, omega=0.20, long record | 230,001 | Separate fixed-phase result supplied |

Each CSV includes its own header. The `*_metadata.txt` files contain the
integration step, discarded transient and retained horizon where these were
recorded. Sampling interval and integration step are not automatically the
same quantity.

## Branch names and numbering

The data keys `Broad_P2`, `Window_C1`, `Window_C2` and `Window_C4` identify the broad
regular-window branch, the secondary branch and its first two doubled branches.
The primary sequence uses `P1`, `P2`, `P4`, `P8`, `P16` and `P32`. A reported
flip of the last parent does not imply that a separate `P64` data file is present.

Raw continuation tables contain initial state columns `x0` through `x4`, period
`T`, residual/iteration diagnostics, and 25 monodromy-matrix entries. Processed
Floquet tables contain their own explicit spectral columns. Do not interpret
matrix entries as multipliers. Read the header of each table.

The exact determinant component uses code indices `(1,2,3,4,5,6)` corresponding
to quadratic-map indices `(M0,M2,M3,M4,M5,M1)`. Its last basis vector is
`L = ell/sqrt(2)`; the generator documents its rational parameter `tau`.

## Interpretation

Stored integration-failure flags and finite-time exponent estimates describe
the computation that produced them. They do not prove finite-time blow-up or
establish asymptotic behaviour on their own. The Protocol II omega=2 additional
response check is supplied separately from the original scan and uses different
numerical diagnostics.

The JSON execution record in the additional Riccati results is preserved as
provenance. The source paths and hashes it contains identify its original run;
they are not runtime paths for the supplied programs. Some of those generating
programs are not included, as documented in the
[reproduction guide](reproducibility.md).
