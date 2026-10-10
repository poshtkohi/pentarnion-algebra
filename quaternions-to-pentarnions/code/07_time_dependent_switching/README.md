# Time-dependent protocols

Protocol I uses `gamma(t) = 0.68 + 0.08*sin(omega*t)`.
Protocol II uses `gamma(t) = 0.525 + 0.475*sin(omega*t)`.
Their model definitions are in `code/00_core_algebra/switching_algebra_model.py`;
the supplied frequency scans are in `data/time_switching/`.

From `quaternions-to-pentarnions/`:

```bash
python code/07_time_dependent_switching/regenerate_report_figures.py --output outputs/protocol_plots
```

This entry point invokes the Riccati plotting routine.
The command plots the supplied Riccati results, including both protocols; it
does not regenerate the numerical scans.
