# Part I result-to-code map

| Manuscript material | Implementation | Check/output |
| --- | --- | --- |
| Associator norms and tensor coefficients | `code/structural_invariants/generate_algebraic_invariants.py` | exact symbolic coefficients and plots |
| Selected rotational identities and determinant checks | `code/rotational_geometry/verify_exact_results.py` | exact residuals |
| Moving frames and rotation illustrations | `code/rotational_geometry/build_assets.py` | equation-based figures |
| Six Jacobian determinant polynomials; middle-map evaluations | `code/bracket_geometry/compute_symbolic.py` | exact symbolic determinants |
| Circle images and left/right zero-divisor curves | `code/bracket_geometry/generate_figures.py` | three generated figures and residual assertions |
| Dense regular and irregular spin records | `code/spin_dynamics/generate_dense_spin.cpp` | trajectory, phase samples, work residual |
| Tangent-space spin Lyapunov scan | `code/spin_dynamics/lyapunov_scan.cpp` | independent declared 14-frequency recipe, two exponents and sum |
| Spin plots | `code/spin_dynamics/make_dense_figures.py` | reference data by default; generated data via environment override |

See the [54-image inventory](../../docs/figure_map.md) and
[numerical conventions and comparisons](../../docs/numerical-notes.md).
