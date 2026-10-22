import unittest
import numpy as np
from tests.support import ROOT
from shared import algebra_core as table
from shared.coordinate_product import product

class SharedAlgebraTests(unittest.TestCase):
    def test_two_independent_products_all_basis_pairs(self):
        e=np.eye(5)
        for g in (0.,0.05,0.3,0.7,1.):
            for a in e:
                for b in e:
                    np.testing.assert_allclose(product(a,b,g),table.product(g,a,b),rtol=1e-14,atol=1e-14)
    def test_seeded_random_vectors(self):
        rng=np.random.default_rng(20261021)
        for g in (0.,0.4,1.):
            for _ in range(20):
                a,b=rng.normal(size=(2,5))
                np.testing.assert_allclose(product(a,b,g),table.product(g,a,b),rtol=1e-13,atol=1e-13)
    def test_unit_squares_conjugation_and_norm(self):
        e=np.eye(5);a=np.array([.7,-.2,.4,.1,-.3]);b=np.array([-.1,.5,-.4,.2,.6])
        for g in (0.,0.3,1.):
            for v in e:
                np.testing.assert_allclose(product(e[0],v,g),v)
                np.testing.assert_allclose(product(v,e[0],g),v)
            for v in e[1:]:np.testing.assert_allclose(product(v,v,g),-e[0])
            np.testing.assert_allclose(table.conjugate(product(a,b,g)),product(table.conjugate(b),table.conjugate(a),g),atol=1e-14)
            np.testing.assert_allclose(product(a,table.conjugate(a),g),float(a@a)*e[0],atol=1e-14)
    def test_endpoint_and_frame(self):
        e=np.eye(5)
        np.testing.assert_allclose(product(e[1],e[2],0),e[3])
        np.testing.assert_allclose(product(e[1],e[2],1),(e[3]+e[4])/np.sqrt(2))
        for g in (0,.5,1):
            P=table.moving_frame(g)
            np.testing.assert_allclose(P.T@P,np.eye(4),atol=1e-14)
    def test_invalid_product_inputs(self):
        for fn in (lambda:product(np.ones(4),np.ones(5),.5),lambda:product(np.ones(5),np.ones(5),2)):
            with self.assertRaises(ValueError):fn()
