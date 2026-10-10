# Rotational calculations

From `quaternions-to-pentarnions/`:

```bash
python code/02_rotational_geometry/verify_exact_results.py
python code/02_rotational_geometry/rotor_geometry.py
```

The first writes selected symbolic expressions and residuals to
`outputs/rotations/symbolic_verification.txt`. The second demonstrates rotation
matrices and a particular four-step rotation composition numerically.
`matlab/` contains independent MATLAB demonstrations.

Optional illustrative plots can be generated with:

```bash
python code/02_rotational_geometry/build_assets.py
```

This reads the supplied earlier spin results and writes local images to
`outputs/plots/`. Set `PENTARNION_OUTPUT_DIR` to change that output directory.
The different cycle constructions in the programs must be distinguished; an
illustration does not verify every possible switching-cycle theorem.
