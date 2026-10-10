# Spin model and trajectories

`mechanical_spin.py` provides the spin equations, energy, Jacobian and a short
Python demonstration. `generate_dense_spin.cpp` generates the two longer
trajectory/work-diagnostic cases. It does not compute the supplied Lyapunov scan.

From `quaternions-to-pentarnions/`, create `outputs/bin`, then:

```bash
c++ -O3 -std=c++17 code/05_spin_dynamics/generate_dense_spin.cpp -o outputs/bin/generate_dense_spin
./outputs/bin/generate_dense_spin outputs/spin_dense
```

Use the executable name produced by your toolchain on Windows. The output path
is optional and defaults to `outputs/spin_dense` in the current working directory.
Integration steps and simulation horizons are specified in the generator.

To plot the supplied reference data, independently of rerunning the generator:

```bash
python code/05_spin_dynamics/make_dense_figures.py
```

The plotter reads `data/spin_dense/` and writes local images to `outputs/plots/`.
A simulation into `outputs/spin_dense/` does not replace those reference inputs.
