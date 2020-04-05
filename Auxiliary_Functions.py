from fenics import nabla_grad, Identity
import numpy as np

"""Define symmetric gradient"""
def D(u):
    return sym(nabla_grad(u))

"""Define stress tensor"""
def sigma(u, p):
    return 2*mu*D(u) - p*Identity(len(u))

"""'Continuous Heaviside approximation'"""
def CHeaviside(psi, eps):
    val = 0.5*(1.0 + psi/eps + 1/np.pi()*np.sin(np.pi()*psi/eps))
    return conditional(lt(abs(psi),eps),val,(np.sign(psi) + 1)/2)
