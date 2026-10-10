# Pentarnion Algebra

Code and supporting data for Pentarnion algebras, their geometry and
algebra-valued differential equations.

## From Quaternions to Pentarnions: geometry and nonlinear dynamics

**Alireza Poshtkohi and Susanne Pumplün**

The computational material is in
[`quaternions-to-pentarnions/`](quaternions-to-pentarnions/).
It includes multiplication and associator calculations, rotational constructions,
Jacobi elliptic solutions, a Hamiltonian spin model, and numerical results for
algebra-valued Riccati equations.

See the [code and data guide](quaternions-to-pentarnions/README.md),
[reproduction instructions](quaternions-to-pentarnions/docs/reproducibility.md),
and [data descriptions](quaternions-to-pentarnions/docs/data-guide.md).
The reproduction instructions identify which calculations can be rerun with
the supplied programs and which results are provided as data only.

## Getting started

From the repository root, with Python 3.10 or later:

```bash
python -m pip install -r quaternions-to-pentarnions/requirements.txt
python quaternions-to-pentarnions/run_audit.py
```

The check inspects source syntax, sampled algebra identities and selected data
files. Numerical simulations and plotting commands are described in the
reproduction instructions. Generated results are written to local `outputs/`
directories and are not tracked by Git.

## Contents

```text
pentarnion-algebra/
├── README.md
├── CITATION.cff
├── .gitignore
├── .gitattributes
└── quaternions-to-pentarnions/
    ├── README.md
    ├── CITATION.cff
    ├── docs/
    ├── code/
    ├── data/
    ├── requirements.txt
    ├── run_audit.py
    └── run_full_verification.py
```

## Citation

Alireza Poshtkohi and Susanne Pumplün,
*From Quaternions to Pentarnions: geometry and nonlinear dynamics*.

Citation metadata are provided in [CITATION.cff](CITATION.cff).
Please also record the Git commit or release used for a calculation.
