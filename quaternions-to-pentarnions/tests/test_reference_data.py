import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile
from tests.support import ROOT
from shared.reference_data import open_reference_text, read_reference_csv

class ReferenceDataTests(unittest.TestCase):
    def test_reference_checksums(self):
        inventory=json.loads((ROOT/'docs/reference_integrity.json').read_text())
        part=os.environ.get('PENTARNION_TEST_PART','all')
        for item in inventory['files']:
            rel=item['path']
            if part!='all' and '/part-'+part+'/' not in '/'+rel:continue
            path=ROOT.parent/rel
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest,item['sha256'],rel)
    def test_record_counts_including_compressed_csv(self):
        records={
          'part-I/data/spin_dense/regular_timeseries.csv':24001,
          'part-I/data/spin_dense/regular_poincare.csv':900,
          'part-I/data/spin_dense/chaotic_timeseries.csv':91001,
          'part-I/data/spin_dense/chaotic_poincare.csv':3000,
          'part-I/data/spin_dense/regular_display_window.csv':2293,
          'part-I/data/spin_dense/chaotic_display_window.csv':10429,
          'part-II/data/riccati_dense/static_gamma_0p60_timeseries.csv':200001,
          'part-II/data/riccati_dense/static_gamma_0p60_poincare.csv':803,
          'part-II/data/riccati_dense/static_gamma_0p70_timeseries.csv':60001,
          'part-II/data/riccati_dense/static_gamma_0p70_poincare.csv':287,
          'part-II/data/riccati_dense/protocol_I_omega_0p2_timeseries.csv':50001,
          'part-II/data/riccati_dense/protocol_I_omega_0p8_timeseries.csv':50001,
          'part-II/data/riccati_dense/protocol_II_omega_0p2_timeseries.csv':50001,
          'part-II/data/riccati_dense/protocol_II_omega_0p8_timeseries.csv':50001,
          'part-II/data/riccati_dense/protocol_I_omega_0p8_dense_timeseries.csv':230001,
          'part-II/data/riccati_dense/protocol_II_omega_0p2_dense_timeseries.csv':230001,
        }
        part=os.environ.get('PENTARNION_TEST_PART','all')
        for rel,count in records.items():
            if part!='all' and not rel.startswith('part-'+part+'/'):continue
            with open_reference_text(ROOT/rel) as f:
                self.assertEqual(sum(1 for _ in f)-1,count,rel)
    def test_zip_helper_does_not_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'small.csv'
            with zipfile.ZipFile(root/'small.zip','w') as z:z.writestr('small.csv','x,y\n1,2\n')
            d=read_reference_csv(target)
            self.assertEqual(d.iloc[0].tolist(),[1,2]);self.assertFalse(target.exists())
    def test_zip_helper_rejects_ambiguous_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with zipfile.ZipFile(root/'small.zip','w') as z:
                z.writestr('small.csv','x\n1\n');z.writestr('other.csv','y\n2\n')
            with self.assertRaises(ValueError):read_reference_csv(root/'small.csv')
