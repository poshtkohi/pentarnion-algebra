# Numerical conventions and comparisons

## Reference data and generated results

The paper reference values are in `part-I/data/` and `part-II/data/`.
`run_checks.py --plots` renders those records. `run_numerics.py` writes fresh
results to a separate output directory; `--plots` renders the generated records.
Each numerical run records its settings, commands, software versions, output
checks and reference comparisons. The configurations specify the numerical
recipes and continuation seeds.

## Compiler sensitivity

The `recorded` build profile uses `-march=native` for the static/frequency scans,
peak scan and full-spectrum driver. The `portable` profile omits native flags.
Floating-point ordering and CPU/compiler differences can change long-time
trajectories and finite-time exponents.

In the [recorded environment](tested_environment.json), the recorded profile
reproduced the stored numeric columns of the static and periodic Riccati scans,
peak table and full spectra. A portable-build comparison gave maximum differences
in the largest finite-time exponent of approximately 0.10064 for the static
scan, 0.05314 for Protocol I and 0.01912 for Protocol II. Numerical agreement
should therefore be assessed together with step-size and time-horizon checks;
it is not guaranteed across platforms.

## Spin-frequency scan

The supplied tangent-space scanner is an independent implementation with explicit
settings: 500 transient and 3000 accumulation forcing periods, time step at most
0.01, projected RK4 including the derivative of the state projection, and two
orthogonalised tangent vectors. This recipe differs numerically from the paper's
reference table. For example:

| Forcing frequency | Reference largest exponent | Independent scan |
| --- | ---: | ---: |
| 0.8 | 0.0131 | 0.03691 |
| 1.2 | 0.05772 | 0.04843 |

The largest absolute difference over the 14 sampled frequencies was about
0.02381. The near-zero sum of the two exponents and an independent short-horizon
DOP853 check are implementation diagnostics, not proofs of long-time convergence.
The reference table and generated results are kept separate.

## Interpretation

The `blowup` column is a numerical stopping flag, not a theorem of finite-time
blow-up. Protocol II at omega=2 is evaluated separately by adaptive integration.
At omega=0.01, an accumulation time of 800 units covers only about 1.27 forcing
periods. Period-doubling values are sampled interpolants, not certified bounds;
continuation endpoints alone do not prove folds or a closed isola.

Finite-time exponents, residual tests and numerical continuation are not
interval-certified proofs of chaos or global existence. The Python/C++ workflows
were tested on Linux; Windows and the supplementary MATLAB examples were not
executed in that environment.
