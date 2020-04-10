from fenics import *

"""Sub domain for Periodic boundary condition"""
class PeriodicBoundary(SubDomain):

    #Left boundary is "target domain"
    def inside(self, x, on_boundary):
        return abs(x[0]) < DOLFIN_EPS and on_boundary

    #Map right boundary to left boundary
    def map(self, x, y):
        y[0] = x[0] - 0.41
        y[1] = x[1]
