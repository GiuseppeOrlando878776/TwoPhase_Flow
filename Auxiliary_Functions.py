from fenics import *
from operator import *
import numpy as np

"""Define symmetric gradient"""
def D(u):
    return sym(nabla_grad(u))

"""Define stress tensor"""
def sigma(mu, u, p):
    return 2*mu*D(u) - p*Identity(len(u))


"""'Continuous Heaviside approximation'"""
def CHeaviside(psi, eps):
    val = 0.5*(1.0 + psi/eps + 1/np.pi*np.sin(np.pi*psi/eps))
    return conditional(lt(np.abs(psi),eps),val,(np.sign(psi) + 1)/2)


"""'Continuous Dirac's delta approximation'"""
def CDelta(psi, eps):
    val = 1.0/(2.0*eps)*(1.0 + np.cos(np.pi*psi.vector().get_local()/eps))
    return conditional(lt(np.abs(psi.vector().get_local()),eps),val,0.0)
