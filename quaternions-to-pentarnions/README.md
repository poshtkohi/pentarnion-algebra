# From Quaternions to Pentarnions: geometry and nonlinear dynamics

**Alireza Poshtkohi and Susanne Pumplün**

Code and supporting data for a one-parameter family of five-dimensional
quadratic algebras, its rotational geometry and associated differential equations.

## Getting started

Run the following commands from `quaternions-to-pentarnions/`:

```bash
python -m pip install -r requirements.txt
python run_audit.py
```

Python 3.10 or later is required. The dense-spin simulation uses a C++17 compiler;
the MATLAB demonstrations run independently of Python.

## Available calculations and data

| Material | Contents |
| --- | --- |
| Multiplication and elementary algebra identities | Python implementations and sampled checks |
| Associator tensors and selected rotational identities | Exact symbolic programs and expressions |
| Elliptic solutions and ordered-product representation | Exact verifiers, numerical calculations and results |
| Six quadratic maps and Jacobian determinants | Exact symbolic generator and expressions |
| Spin trajectories | Python model, C++ dense-trajectory generator and data |
| Riccati equation and tangent action | Python model in `code/00_core_algebra/` |
| Riccati origin characteristic polynomial | Exact standalone verifier and coefficient-by-coefficient result |
| Riccati scans, periodic-orbit continuation and additional validation | Numerical results and plotting routines; the full set of generating programs is not included |

The complete Riccati continuation and frequency scans cannot be regenerated
from the supplied programs alone. The [reproduction guide](docs/reproducibility.md)
describes the available calculations and identifies the missing numerical drivers.

## Exact calculations

```bash
python run_full_verification.py
python run_full_verification.py --full --output outputs/exact_checks
```

The commands run in a temporary copy. The second includes the ordered-product
and determinant calculations and saves their outputs to a new directory.
Neither command overwrites `data/`. Both include the exact Riccati
characteristic-polynomial verifier. To run only that check, without the data
audit or any numerical data:

```bash
python code/06_riccati_dynamics/section9_validation/verify_characteristic_polynomial.py
```

See the [verification details](code/06_riccati_dynamics/section9_validation/README.md#exact-characteristic-polynomial-verification).

## Plotting

```bash
python code/05_spin_dynamics/make_dense_figures.py
python code/06_riccati_dynamics/make_dense_figures.py
```

The plotting routines read the supplied data and generate images locally in
`outputs/plots/`. Generating a plot does not rerun the underlying simulation.

## Documentation

- [Reproduction guide](docs/reproducibility.md): software requirements, commands and numerical settings.
- [Data guide](docs/data-guide.md): coordinate conventions, result files and column definitions.

## Citation

Alireza Poshtkohi and Susanne Pumplün,
*From Quaternions to Pentarnions: geometry and nonlinear dynamics*.

See [CITATION.cff](CITATION.cff) and record the repository revision used.
