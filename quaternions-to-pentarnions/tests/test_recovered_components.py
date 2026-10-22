import importlib.util,json,unittest
from pathlib import Path
from tests.support import ROOT
class NumericalComponents(unittest.TestCase):
    def test_bracket_geometry(self):
        p=ROOT/'part-I/code/bracket_geometry/generate_figures.py';spec=importlib.util.spec_from_file_location('bracket_figures',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);r=mod.checks();self.assertLess(r['determinant_max_residual'],1e-12)
    def test_numerical_sources_present(self):
        sources=['part-II/code/riccati_dynamics/static_time_scan.cpp', 'part-II/code/riccati_dynamics/static_bifurcation_peaks.cpp', 'part-II/code/riccati_dynamics/full_lyapunov_spectra.cpp', 'part-II/code/riccati_dynamics/generate_dense_riccati.cpp', 'part-II/code/riccati_dynamics/periodic_branches/branch_continue.cpp', 'part-II/code/riccati_dynamics/periodic_branches/branch_refine.cpp', 'part-II/code/riccati_dynamics/periodic_branches/cascade_continue.cpp', 'part-II/code/riccati_dynamics/periodic_branches/p32_continue.cpp', 'part-II/code/riccati_dynamics/periodic_branches/window_continue.cpp', 'part-II/code/riccati_dynamics/validation/verify_section9.py', 'part-II/code/riccati_dynamics/validation/targeted_spectrum.cpp', 'part-II/code/riccati_dynamics/validation/localise_archived_events.py', 'part-II/code/riccati_dynamics/validation/sample_archived_forcing_phase.py']
        for path in sources:self.assertTrue((ROOT/path).is_file(),path)
    def test_refinement_recipes(self):
        rows=json.loads((ROOT/'configs/refinement_recipes.json').read_text())['recipes'];self.assertEqual(len(rows),11)
        for r in rows:
            self.assertEqual(len(r['seed_state']),5);self.assertGreater(r['seed_period'],0);self.assertGreater(r['gamma_step'],0);self.assertTrue((ROOT/r['source_reference']).is_file())
