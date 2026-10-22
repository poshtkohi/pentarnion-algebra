import os
import unittest
import numpy as np
from tests.support import load

@unittest.skipIf(os.environ.get('PENTARNION_TEST_PART')=='II','Part II selected')
class PartITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rot=load('part-I/code/rotational_geometry/rotor_geometry.py','tested_rotor_geometry')
        cls.spin=load('part-I/code/spin_dynamics/mechanical_spin.py','tested_spin')
    def test_rotation_orthogonality(self):
        for g in (0.,.5,1.):
            for axis in ('h','b','c'):
                M=self.rot.rotor_matrix(g,axis,.7)
                np.testing.assert_allclose(M.T@M,np.eye(4),atol=1e-13)
                self.assertAlmostEqual(np.linalg.det(M),1.,places=12)
    def test_spin_tangent_and_jacobian(self):
        model=self.spin; y=np.array([.4,.5,-.7]);h=np.array([.2,-.3,.1]);eps=1e-6
        for t in (0.,.3,1.):
            f=model.rhs(t,y,.3)
            self.assertLess(abs(y@f),1e-13)
            fd=(model.rhs(t,y+eps*h,.3)-model.rhs(t,y-eps*h,.3))/(2*eps)
            np.testing.assert_allclose(fd,model.jacobian(t,y,.3)@h,atol=1e-9,rtol=1e-8)
    def test_spin_short_simulation(self):
        result=self.spin.simulate(.3,2.,dt=.01,sample=5)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(np.linalg.norm(result[:,1:4],axis=1),1.,atol=1e-13)
