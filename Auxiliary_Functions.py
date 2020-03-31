from fenics import nabla_grad
import numpy as np

"""Define symmetric gradient"""
def D(u):
    return sym(nabla_grad(u))

"""Define stress tensor"""
def sigma(u, p):
    return 2*mu*D(u) - p*Identity(len(u))

"""'Continuous Heaviside approximation'"""
def Sign(q, eps):
    val = 0.5*(1.0 + q/eps + 1/np.pi()*np.sin(np.pi()*q/eps))
    return conditional(lt(abs(q),eps),val,sign(q))
