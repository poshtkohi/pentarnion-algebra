# Quadratic maps and Jacobian determinants

From `quaternions-to-pentarnions/`:

```bash
python code/04_bracket_quintics/compute_symbolic.py
```

The generator constructs six quadratic maps, their Jacobians and determinants,
and verifies exact conjugation identities. Output defaults to
`outputs/bracket_maps/`; `--output` selects another directory. The calculation
can take several minutes. Reference expressions are in `symbolic/`.

Code indices `(1,2,3,4,5,6)` correspond to `(M0,M2,M3,M4,M5,M1)`.
The rationalized basis uses `L=ell/sqrt(2)`; see the generator's docstring for
the definition of `tau`. This component does not classify projective singularities.

## Exact nonzero values for the two middle-position maps

The manuscript uses

```text
M3(p) = p * (i * p)
M4(p) = (p * i) * p
Qr(p; gamma) = det(DMr(p))
```

where `*` is the switching multiplication and the displayed parentheses are
retained. For these two maps the manuscript and code indices agree. At the
actual algebra element `p = 1 + ell`, the exact identities are

```text
Q3(1+ell; gamma) = Q4(1+ell; gamma)
                 = 4*kappa_gamma*sigma_gamma
                 = 4*gamma/(1+gamma**2).
```

A nonzero value at one point proves that each determinant polynomial is not
identically zero for `gamma > 0`. It does not imply nonvanishing everywhere,
and does not assert that `Q3` and `Q4` are the same polynomial. At `gamma = 0`
both displayed point values are zero.

For a quick verification, without recomputing the six full polynomials, run:

```bash
python code/04_bracket_quintics/compute_symbolic.py --nonzero-only
```

The script builds the Jacobians from the multiplication, substitutes the point,
and calculates both determinants using exact arithmetic over
`Q(sqrt(2))(tau)`. It raises an error unless each residual against the stated
value is exactly zero. This command does not need the trajectory data.

**Coordinate convention:** the calculation uses `L = ell/sqrt(2)`, so
`1 + ell` has coordinates `(w,x,y,z,v) = (1,0,0,0,sqrt(2))`. Using `v=1`
would test a different element. The same basis change is applied to domain and
codomain, so it does not change the Jacobian determinant. The rational parameter satisfies

```text
gamma = 2*sqrt(2)*tau/(1-2*tau**2)
4*gamma/(1+gamma**2) = 8*sqrt(2)*tau*(1-2*tau**2)/(1+2*tau**2)**2.
```

The nonnegative parameter branch `gamma >= 0` corresponds to
`0 <= tau < 1/sqrt(2)`; the paper's interval `0 <= gamma <= 1` corresponds to
`0 <= tau <= 1-1/sqrt(2)`.

The result, evaluated matrices and zero residuals are written to
`outputs/bracket_maps/nonzero_jacobian_checks.txt`. A reference result from a
full run is supplied in
[`symbolic/nonzero_jacobian_checks.txt`](symbolic/nonzero_jacobian_checks.txt).

The default full calculation also recomputes these point determinants and
checks their equality with the specializations of its newly computed generic
`Q3` and `Q4` polynomials. The existing
`python run_full_verification.py --full` entry point includes these checks
because it already invokes this generator. Its read-only data check runs first;
see the [reproduction guide](../../docs/reproducibility.md) for the preparation
of the three compressed trajectory CSVs in the GitHub download.

Use `--output DIRECTORY` with either command to choose a different destination.
The supplied files in `symbolic/` and all files in `data/` remain unchanged.
