# Pentarnion Algebra

Code and data accompanying the papers by **Alireza Poshtkohi and Susanne Pumplün**.

| Paper | Computational companion |
| --- | --- |
| *A five-dimensional nonassociative algebra: Elliptic Integrability and Bracketing-Dependent Dynamics* | [Original paper](pentarnion-dynamics/README.md) |
| *From Quaternions to Pentarnions I: Algebra, geometry, and a spin model* | [Part I](quaternions-to-pentarnions/part-I/README.md) |
| *From Quaternions to Pentarnions II: Quadratic differential equations* | [Part II](quaternions-to-pentarnions/part-II/README.md) |

## Original fixed-algebra paper

The [`pentarnion-dynamics/`](pentarnion-dynamics/README.md) folder contains
its Python/MATLAB source, exact-model sample data and the computational guide
moved from Appendix B. Its installation and check command is separate:

```bash
python -m pip install -r pentarnion-dynamics/requirements.txt
python pentarnion-dynamics/run_checks.py
```

## Parts I and II: installation and verification

From the repository root, using Python 3.10 or later:

```bash
python -m pip install -r quaternions-to-pentarnions/requirements.txt
python quaternions-to-pentarnions/run_checks.py
python quaternions-to-pentarnions/run_checks.py --level full --plots
```

The default check runs the unit and reference-data integrity tests. The full
command also runs the symbolic verifications and generates the manuscript
figures from formulas and reference data. Compressed CSV inputs are read directly.

## Parts I and II: numerical experiments

With a C++17 compiler (`g++` or `clang++`) on PATH:

```bash
python quaternions-to-pentarnions/run_numerics.py --task smoke
python quaternions-to-pentarnions/run_numerics.py --task all --threads 4 --plots
```

Individual tasks include `scans`, `peaks`, `spectra`, `dense`, `branches`,
`validation`, `spin` and `spin-scan`. Runs write to separate output directories
and do not overwrite the paper reference data.

- [Reproduction instructions](quaternions-to-pentarnions/docs/reproducibility.md)
- [Result and figure inventory](quaternions-to-pentarnions/docs/figure_map.md)
- [Data guide](quaternions-to-pentarnions/docs/data-guide.md)
- [Numerical conventions and comparisons](quaternions-to-pentarnions/docs/numerical-notes.md)

## Citation and licence

Cite the relevant paper and the repository version or commit used.
[CITATION.cff](CITATION.cff) describes the software collection; each paper folder
also contains citation metadata. Code is distributed under the
[GNU General Public License, version 3](LICENSE).
