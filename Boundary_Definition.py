from dolfin import *

"""No-slip boundary"""
class NoSlip_Boundary(SubDomain):
    #Constructor to set the proper parameter
    def __init__(self, height):
        super().__init__()
        self.height = height

    #Override of the 'inside' function
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) or near(x[1], self.height)


"""Free-slip boundary"""
class FreeSlip_Boundary(SubDomain):
    #Constructor to set the proper parameter
    def __init__(self, base):
        super().__init__()
        self.base = base

    #Override of the 'inside' function
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) or near(x[0], self.base)
