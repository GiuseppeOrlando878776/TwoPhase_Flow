from dolfin import *
import numpy as np
import ufl

"""Define symmetric gradient"""
def D(u):
    return sym(grad(u))

"""Euclidean norm for the gradient"""
def mgrad(b):
    return(sqrt(b.dx(0)**2 + b.dx(1)**2))


"""Approximate sign function"""
def signp(l, eps):
    return l/sqrt(l*l + eps*eps*mgrad(l)*mgrad(l))

"""'Continuous Heaviside approximation'"""
def CHeaviside(psi, eps):
    return conditional(lt(abs(psi),eps), 0.5*(1.0 + psi/eps + 1/np.pi*ufl.sin(np.pi*psi/eps)), (ufl.sign(psi) + 1)/2.0)

"""'Continuous Dirac's delta approximation'"""
def CDelta(psi, eps):
    return conditional(lt(abs(psi),eps), 1.0/(2.0*eps)*(1.0 + ufl.cos(np.pi*psi/eps)), 0.0)


"""Surface gradient"""
def grad_s(f, n):
    return grad(f) - inner(grad(f),n)*n


"""Surface divergence"""
def dive_s(f,n):
    return div(f) - inner(dot(n, nabla_grad(f)), n)
