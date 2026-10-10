# Additional Riccati validation data: plotting

`make_validation_figures.py` reads the results already supplied in
`data/riccati/section9_validation/` and the periodic-branch tables. It generates
five local diagnostic plots. It does not need the original validation drivers,
which are not present in this distribution.

Run from `quaternions-to-pentarnions/`:

```bash
python code/06_riccati_dynamics/section9_validation/make_validation_figures.py
```

Outputs go to `outputs/plots/`, or the directory in `PENTARNION_OUTPUT_DIR`.
For the distinction between data inspection and rerunning the calculations,
see `docs/reproducibility.md` in `quaternions-to-pentarnions/`.
