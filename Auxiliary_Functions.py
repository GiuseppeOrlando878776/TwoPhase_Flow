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
    return conditional(lt(abs(psi),eps), 0.5*(1.0 + psi/eps + 1/np.pi*sin(np.pi*psi/eps)), (ufl.sign(psi) + 1)/2.0)

"""'Continuous Dirac's delta approximation'"""
def CDelta(psi, eps):
    return conditional(lt(abs(psi),eps), 1.0/(2.0*eps)*(1.0 + cos(np.pi*psi/eps)), 0.0)

"""'Continuous Dirac's delta approximation'"""
def CDelta_LS(psi, eps):
    return 1.0/eps*(ufl.exp(-psi)/eps)/((ufl.exp(-psi)/eps + 1.0)**2)
