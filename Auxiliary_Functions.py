from dolfin import *
import numpy as np
import ufl

"""Define symmetric gradient"""
def D(u):
    return sym(grad(u))

"""Euclidean norm gradient"""
def mgrad(b):
    return(sqrt(b.dx(0)**2 + b.dx(1)**2))


"""Approximate sign function"""
def signp(l, eps):
    return l/sqrt(l*l + eps*eps*mgrad(l)*mgrad(l))

"""'Continuous Heaviside approximation'"""
def CHeaviside(psi, eps):
    val = 0.5*(1.0 + psi/eps + 1/np.pi*sin(np.pi*psi/eps))
    return conditional(lt(abs(val),eps), val, (ufl.sign(psi) + 1)/2.0)

"""'Continuous Dirac's delta approximation'"""
def CDelta(psi, eps):
    val = 1.0/(2.0*eps)*(1.0 + cos(np.pi*psi/eps))
    return conditional(lt(abs(val),eps), val, 0.0)
