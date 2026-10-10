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
