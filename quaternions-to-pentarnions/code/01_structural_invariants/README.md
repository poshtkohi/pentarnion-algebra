# Associator calculations

From `quaternions-to-pentarnions/`:

```bash
python code/01_structural_invariants/generate_algebraic_invariants.py --no-plots
```

The script computes the associator entries and norm sums exactly with SymPy.
It writes new results to `outputs/structural/data/`. Omit `--no-plots` to also
create a local plot under `outputs/structural/plots/`. `--root` changes the output
base. The supplied reference tables in `data/structural/` are not overwritten.
The group/nuclei labels repeated in the CSV are not proofs supplied by this script.
