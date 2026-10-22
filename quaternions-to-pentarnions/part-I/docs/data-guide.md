# Part I data guide

`data/structural/` contains exact associator expressions and tensor entries.
`data/spin/` contains earlier illustrative spin records and the representative
Lyapunov-frequency table. `data/spin_dense/` contains the long reference records
used for the dense displays, fixed-phase samples, work checks and metadata.

Spin state columns `(y_h,y_b,y_c)` are coordinates in the moving orthonormal
basis `(h_gamma,b,c)`. Columns `(a,b,c,d)` are the fixed adapted pure-space
coordinates, with `d` the actual ell coefficient. `H` or `energy` is the
Hamiltonian, not the squared state norm. The field scaling parameter named `mu`
in the short spin model is unrelated to Part II's elliptic modulus scaling;
the manuscript spin example uses field scale 1.

| Record | Trajectory samples | Fixed-phase samples | Display-window samples |
| --- | ---: | ---: | ---: |
| regular, Omega=0.30 | 24,001 | 900 | 2,293 |
| chaotic, Omega=1.20 | 91,001 | 3,000 | 10,429 |

The inherited filenames containing `chaotic` are retained as data identifiers.
They do not upgrade the manuscript's numerical evidence to a chaos proof.
Each CSV keeps its original header and bytes. Adjacent metadata describes the
step, transient, horizon and work residual. New generator output goes outside
`data/` and is not a replacement for these reference inputs.

Reference-file checksums are in
[reference_integrity.json](../../docs/reference_integrity.json).
