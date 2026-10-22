# Quaternions to Pentarnions: computational companions

[Part I](part-I/README.md) covers algebra, rotational and bracket geometry, and
the spin model. [Part II](part-II/README.md) covers exact solutions and Riccati
dynamics. `shared/` contains common numerical algebra and reference-data readers.

From the repository root:

```bash
python -m pip install -r quaternions-to-pentarnions/requirements.txt
python quaternions-to-pentarnions/run_checks.py --level full --plots
python quaternions-to-pentarnions/run_numerics.py --task all --threads 4 --plots
```

`run_checks.py` verifies symbolic identities and plots formulas/reference data.
`run_numerics.py` compiles C++17 programs and computes numerical records in a
separate output tree. The runners accept paths independent of the working directory.

See the [reproduction guide](docs/reproducibility.md),
[figure inventory](docs/figure_map.md), [data guide](docs/data-guide.md), and
[numerical conventions and comparisons](docs/numerical-notes.md).
