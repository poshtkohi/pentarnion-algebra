# Pentarnion dynamics

Computational companion to **A five-dimensional nonassociative algebra:
Elliptic Integrability and Bracketing-Dependent Dynamics**, by
**Alireza Poshtkohi and Susanne Pumplün**.

This is the original fixed-algebra paper. The later switching-family papers,
*From Quaternions to Pentarnions I* and *II*, have separate companions in
[quaternions-to-pentarnions](../quaternions-to-pentarnions/README.md).

## Contents

- `code/`: the supplied Python and MATLAB programs, including the two
  independent Jacobiator verifiers.
- `data/`: compact numerical samples and summaries for the rank-four
  elliptic example, the full-dimensional Jacobi solution and the
  associator-compensated 2:3:5 example, plus a targeted check of the reported
  periodic orbit.
- [Computational guide](docs/reproducibility.md): the script hierarchy from
  Appendix B, execution commands, outputs and numerical-data prerequisites.
- [Data guide](data/README.md): variables and provenance of the supplied samples.

## Installation and exact-model checks

From the repository root, using Python 3.10 or later:

```bash
python -m pip install -r pentarnion-dynamics/requirements.txt
python pentarnion-dynamics/run_checks.py
```

The check runner executes the supplied symbolic certificates, exact-model
examples and a targeted reintegration of the reported periodic orbit in `pentarnion-dynamics/outputs/checks/`. The original scientific
scripts are copied into that workspace, so their relative output paths work
without overwriting the supplied sample data. Generated figures and reports
stay in the workspace.

For a numerical endpoint check as well:

```bash
python pentarnion-dynamics/run_checks.py --endpoints
```

This adds the original Python program's `--quick` endpoint calculation. It is
not a rerun of the complete long-time continuation and parameter scans.
For those workflows and the MATLAB entry points, see the computational guide.

## Citation and licence

Please cite:

A. Poshtkohi and S. Pumplün, *A five-dimensional nonassociative algebra:
Elliptic Integrability and Bracketing-Dependent Dynamics*, manuscript, 2026.

Also record the repository commit or release used for computations.
[CITATION.cff](CITATION.cff) provides software citation metadata.
The repository's [GPL-3.0 licence](../LICENSE) applies.
