from My_Parameters import My_Parameters
from fenics import *

class RayleghTaylor:

    def __init__(self, param_name):
        """
        Param --- class Parameters to store desired configuration
        Ra    --- Raylegh number
        Pr    --- Prandtl number
        dt    --- Specified time step
        deg   --- Polynomial degree
        """

        self.Param = My_Parameters(param_name).get_param()

        try:
            self.Ra = self.Param["Raylegh_number"]
            self.Pr = self.Param["Prandtl_number"]
            self.dt = self.Param["Time_step"]
            self.deg = self.Param["Polynomial_degree"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")
