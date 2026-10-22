import ast
import json
from pathlib import Path
import unittest
from tests.support import ROOT

class LayoutTests(unittest.TestCase):
    def test_syntax(self):
        for p in ROOT.rglob('*.py'):
            if 'outputs' not in p.parts and '__pycache__' not in p.parts:
                ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    def test_manuscript_figure_inventory(self):
        expected=json.loads((ROOT/'tests/figure_inputs.json').read_text())
        mapping=json.loads((ROOT/'docs/figure_map.json').read_text())
        for part, figure_inputs in expected.items():
            names={r['figure_file'] for r in mapping if r['part']==part}
            self.assertEqual(names,set(figure_inputs))
        for row in mapping:
            for rel in row['scripts']:
                self.assertTrue((ROOT/rel).is_file(),rel)
    def test_no_manuscripts_or_rendered_figures_in_source_tree(self):
        for p in ROOT.parent.rglob('*'):
            if not p.is_file() or 'outputs' in p.parts or '.git' in p.parts:continue
            self.assertNotIn(p.suffix.lower(),{'.tex','.pdf','.png','.jpg','.exe','.o'},str(p))
