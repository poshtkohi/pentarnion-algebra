import os
import unittest
import numpy as np
from tests.support import load

@unittest.skipIf(os.environ.get('PENTARNION_TEST_PART')=='I','Part I selected')
class PartIITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load('part-II/code/riccati_dynamics/switching_algebra_model.py','tested_riccati')
        cls.elliptic=load('part-II/code/exact_solutions/generate_figures_and_data.py','tested_elliptic')
    def test_riccati_jacobian_action(self):
        m=self.model;p=np.array([.2,-.1,.3,.5,-.4]);h=np.array([.1,.4,-.2,.3,.1]);eps=1e-6
        for g in (0.,.3,1.):
            fd=(m.static_rhs(0,p+eps*h,g)-m.static_rhs(0,p-eps*h,g))/(2*eps)
            np.testing.assert_allclose(fd,m.tangent_action(p,h,g),rtol=1e-8,atol=1e-9)
    def test_reference_coordinate_equivalence(self):
        m=self.model; q=np.array([.2,.1,.4,-.1,.3]);t=.31;omega=.8
        for protocol in (m.TRANSITION_BAND,m.REGULARIZED_FULL_RANGE):
            g=protocol.gamma(t,omega);gd=protocol.gamma_dot(t,omega)
            p=q.copy();p[4]*=g
            qdot=m.hidden_rhs(t,q,omega,protocol)
            actual=qdot.copy();actual[4]=g*qdot[4]+gd*q[4]
            np.testing.assert_allclose(actual,m.static_rhs(t,p,g),atol=1e-13)
    def test_jacobi_formula_residual(self):
        e=self.elliptic
        times=np.linspace(0,2,17)
        for g in (0.,.2,1.):
            y=e.exact_solution(times,g,.5)
            derivative=e.exact_derivative(times,g,.5)
            field=np.array([e.vector_field(t,p,g,.5) for t,p in zip(times,y.T)]).T
            np.testing.assert_allclose(derivative,field,rtol=1e-11,atol=1e-11)
