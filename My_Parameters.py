from dolfin import Parameters


"""This class reads parameter files specific for the problem.
It is necessary because the 'Parameters' class provided by FENICS does not
accept a file, so basically we encapsulate it in a new class"""
class My_Parameters:

    """Class constructor which reads or eventually creates the file with parameters
       and instantiates the FENICS  Parameter class"""
    def __init__(self, param_name):
        self.Param = Parameters()

        try:
            self.file = open(param_name, "r")
        except IOError:
            print("Input parameter file '" + param_name + "' not found. Creating a default one")
            f = open(param_name, "w")
            f.write("Reynolds_number = 1000\n")
            f.write("Atwood_number = 0.5\n")
            f.write("Surface_tension = 1.0\n")
            f.write("Lighter_density = 1.0\n")
            f.write("Time_step = 0.1\n")
            f.write("End_time = 2.0\n")
            f.write("Polynomial_degree = 1\n")
            f.write("Number_vertices = 64\n")
            f.write("Base = 1.0\n")
            f.write("Height = 3.0\n")
            f.write("x_center = 0.5\n")
            f.write("y_center = 0.5\n")
            f.write("Radius = 0.2\n")
            f.write("Reinit_Type = 1\n")
            f.write("Stabilization_Type = 1\n")
            f.close()
            self.file = open(param_name, "r")

        try:
            self.parse_parameters(self.file)
        except Exception as e:
            print("Caught an exception: " + str(e))


    """Parse parameters file into the standard FENICS Parameters class"""
    def parse_parameters(self, param_file):
        for line in param_file.readlines():
            if line.strip(): #Avoid reading blank lines
                idx_eq = line.find(' = ')
                if(idx_eq != -1):
                    self.Param.add(line[0 : idx_eq],float(line[idx_eq + 3 : -1]))
                else:
                    raise Exception("Invalid format to read parameters. \
                                     Please check the configuration file: \
                                     you need a space before and after the equal")


    """Return the standard FENICS Parameters class"""
    def get_param(self):
        return self.Param


    """Set type of reinitialization"""
    def reinit_type(i):
        switcher = {1:'Non_Conservative',2:'Conservative'}
        return switcher.get(i, raise RuntimeError("Invalid option for reinitialization"))


    """Set type of stabilization"""
    def stabilization_type(i):
        switcher = {1:'IP',2:'SUPG'}
        return switcher.get(i, raise RuntimeError("Invalid option for stabilization"))
