# Pentarnion Algebra — Parts I and II

Code and data accompanying the papers by **Alireza Poshtkohi and Susanne Pumplün**.

| Paper | Computational companion |
| --- | --- |
| *From Quaternions to Pentarnions I: Algebra, geometry, and a spin model* | [Part I](quaternions-to-pentarnions/part-I/README.md) |
| *From Quaternions to Pentarnions II: Quadratic differential equations* | [Part II](quaternions-to-pentarnions/part-II/README.md) |

## Installation and verification

From the repository root, using Python 3.10 or later:

```bash
python -m pip install -r quaternions-to-pentarnions/requirements.txt
python quaternions-to-pentarnions/run_checks.py
python quaternions-to-pentarnions/run_checks.py --level full --plots
```

The default check runs the unit and reference-data integrity tests. The full
command also runs the symbolic verifications and generates the manuscript
figures from formulas and reference data. Compressed CSV inputs are read directly.

## Numerical experiments

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
