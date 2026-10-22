# Manuscript figure-to-code inventory

All 54 image inputs have generating/plotting code. Reference plots and numerical
recomputations are separate workflows. Equation-based reconstructions need not be
pixel-identical to earlier renderings. Group numbers correspond to the supplied
manuscript snapshot; labels and filenames provide stable identifiers.

## Part I

| Figure group | Image | Script | Input / verification |
| --- | --- | --- | --- |
| 1 | `associator_norms.png` | `part-I/code/structural_invariants/generate_algebraic_invariants.py` | exact formula |
| 1 | `tensor_rotation_deformation_norms.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 2 | `raw_sandwich_norm_defect.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 2 | `tilted_rotation_circles_on_sphere.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 2 | `tilted_rotation_plane_disks.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 3 | `switching_holonomy_loop.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 3 | `switching_holonomy_phase.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 3 | `double_rotation_torus_projection.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 4 | `gamma_bracketing_ellipse_3d.png` | `part-I/code/bracket_geometry/generate_figures.py` | Equation-based reconstruction; identities and determinant residuals checked. |
| 4 | `six_bracketings_gamma1_circle.png` | `part-I/code/bracket_geometry/generate_figures.py` | Equation-based reconstruction; identities and determinant residuals checked. |
| 4 | `gamma1_left_right_determinant_geometry.png` | `part-I/code/bracket_geometry/generate_figures.py` | Equation-based reconstruction; identities and determinant residuals checked. |
| 5 | `multiplication_induced_anisotropy.png` | `part-I/code/rotational_geometry/build_assets.py` | formula or reference data |
| 6 | `spin_lyapunov_frequency_scan.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 6 | `spin_regular_trajectory_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 7 | `spin_regular_poincare_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 7 | `spin_regular_timeseries_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 8 | `spin_chaotic_trajectory_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 8 | `spin_chaotic_poincare_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 9 | `spin_chaotic_timeseries_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 9 | `spin_chaotic_fixed_frame_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |
| 10 | `spin_energy_balance_dense.png` | `part-I/code/spin_dynamics/make_dense_figures.py` | reference data |

## Part II

| Figure group | Image | Script | Input / verification |
| --- | --- | --- | --- |
| 1 | `trajectory_family_3d.png` | `part-II/code/exact_solutions/generate_figures_and_data.py` | explicit Jacobi formula and small numerical illustration |
| 1 | `invariant_curves_wx.png` | `part-II/code/exact_solutions/generate_figures_and_data.py` | explicit Jacobi formula and small numerical illustration |
| 1 | `period_vs_gamma.png` | `part-II/code/exact_solutions/generate_figures_and_data.py` | explicit Jacobi formula and small numerical illustration |
| 1 | `affine_span_audit.png` | `part-II/code/exact_solutions/generate_figures_and_data.py` | explicit Jacobi formula and small numerical illustration |
| 2 | `riccati_event_timeline.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 2 | `riccati_static_largest_lyapunov.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 3 | `riccati_static_bifurcation_diagram.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 4 | `riccati_primary_branch_periods.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 4 | `riccati_primary_floquet_flip_sequence.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 5 | `riccati_primary_cascade_zoom.png` | `part-II/code/riccati_dynamics/validation/make_validation_figures.py` | reference data |
| 6 | `riccati_primary_cascade_scaling.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 6 | `riccati_primary_orbit_growth.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 7 | `riccati_broad_P2_stability.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 7 | `riccati_regular_window_cascade.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 8 | `riccati_narrow_window_stability.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 8 | `riccati_periodic_branch_atlas.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 9 | `riccati_static_chaotic_trajectory_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 9 | `riccati_static_chaotic_poincare_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 9 | `riccati_static_chaotic_timeseries_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 10 | `riccati_static_regular_trajectory_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 10 | `riccati_static_regular_poincare_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 10 | `riccati_static_regular_timeseries_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 11 | `both_protocols_frequency_scan.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 12 | `protocol_band_omega_0p2_timeseries.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 12 | `protocol_band_omega_0p8_timeseries.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 13 | `protocol_transition_band_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 13 | `protocol_I_stroboscopic.png` | `part-II/code/riccati_dynamics/validation/make_validation_figures.py` | reference data |
| 14 | `protocol_full_omega_0p2_timeseries.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 14 | `protocol_full_omega_0p8_timeseries.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 15 | `protocol_near_full_dense.png` | `part-II/code/riccati_dynamics/make_dense_figures.py` | reference data |
| 15 | `protocol_II_stroboscopic.png` | `part-II/code/riccati_dynamics/validation/make_validation_figures.py` | reference data |
| 16 | `protocol_II_omega2_periodic_response.png` | `part-II/code/riccati_dynamics/validation/make_validation_figures.py` | reference data |
| 16 | `protocol_II_omega2_return_residual.png` | `part-II/code/riccati_dynamics/validation/make_validation_figures.py` | reference data |
