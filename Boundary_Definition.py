from dolfin import *

"""No-slip boundary"""
class NoSlip_Boundary(SubDomain):
    def __init__(self, height):
        super().__init__()
        self.height = height

    def inside(self, x, on_boundary):
        return near(x[1], 0.0) or near(x[1], self.height)


"""Free-slip boundary"""
class FreeSlip_Boundary(SubDomain):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def inside(self, x, on_boundary):
        return near(x[0], 0.0) or near(x[0], self.base)
