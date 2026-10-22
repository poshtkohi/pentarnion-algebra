# Part II data guide

| Folder | Contents |
| --- | --- |
| `data/exact_integrable/` | Jacobi trajectories, periods, residuals, affine-span illustration and exact identities |
| `data/riccati/` | Fixed-gamma scans, representative spectra and branch summaries |
| `data/riccati/periodic_branches/` | Continuation records, selected orbit samples and processed Floquet tables |
| `data/riccati/validation/` | Hopf, orbit, spectrum and forcing-phase checks |
| `data/riccati_dense/` | Long static/protocol trajectories, three compressed CSVs and their metadata |
| `data/time_switching/` | Both finite-time protocol frequency scans |

Canonical coordinates are `(w,x,y,z,u)=(p0,p1,p2,p3,p4)` in `(1,i,j,k,ell)`.
`u` is the actual ell coefficient. Reference coordinates obey `p4=gamma*q4`.
In the exact-solution generator arrays have shape `(5,number_of_times)`; CSVs
store one time sample per row. Parameter `q` in the exact verifier is the
elliptic scale sigma, not the state vector used in the time-dependent protocols.

| Reference record | Time-series rows | Section samples |
| --- | ---: | ---: |
| Static gamma=0.6 | 200,001 (compressed CSV) | 803 upward p1=0 crossings |
| Static gamma=0.7 | 60,001 | 287 upward p1=0 crossings |
| Protocol I omega=0.8, long | 230,001 (compressed CSV) | 147 separately supplied phase samples |
| Protocol II omega=0.2, long | 230,001 (compressed CSV) | 36 separately supplied phase samples |
| Four short protocol component records | 50,001 each | Not fixed-phase samples |

ZIPs are original single-member archives. The reader checks the member name and
reads directly; manual extraction is unnecessary. Do not upload an extra
extracted CSV alongside each ZIP.

Data keys `Broad_P2`, `Window_C1`, `Window_C2`, `Window_C4` mean the broad-window
branch, secondary branch and its two doubled branches. Primary branches run
through `P32`; a detected last flip does not mean a separate `P64` orbit was
continued. Raw continuation columns `x0` to `x4`, `T` and 25 monodromy entries
are not all Floquet multipliers; processed Floquet files have explicit headers.

JSON metadata records the settings associated with the reference calculations.
New runs record their own settings and source hashes in separate output manifests.

See [reproducibility.md](reproducibility.md) for workflows and numerical
interpretation, and [reference-file checksums](../../docs/reference_integrity.json)
for integrity checks.
