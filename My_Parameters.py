from dolfin import Parameters


"""This class reads parameter files specific for the problem.
It is necessary because the 'Parameters' class provided by FENICS does not
accept a file, so basically we encapsulate it in a new class"""
class My_Parameters:

    """Class constructor which reads or eventually creates the file with parameters
       and instantiates the FENICS  Parameter class"""
    def __init__(self, param_name):
        self.Param = Parameters()

        #Since these parameters are more related to the numeric part
        #rather than physics we choose to set a default value in order to
        #avoid problems in case they will not present in the file
        self.Param.add("Polynomial_degree_NS", 1)
        self.Param.add("Polynomial_degree_VF", 2)
        self.Param.add("Number_vertices_x", 80)
        self.Param.add("Number_vertices_y", 160)
        self.Param.add("Log_Level", 21) #more than INFO level by default
        self.Param.add("Reinit_Type", 'Non_Conservative_Hyperbolic')
        self.Param.add("Stabilization_Type", 'SUPG')
        self.Param.add("NS_Procedure", 'ICT')
        self.Param.add("VF_Procedure", 'Continuous')
        self.Param.add("Interface_Thickness", 0.025)
        self.Param.add("Stabilization_Parameter", 0.01)
        self.Param.add("Reference_Dimensionalization", 'Dimensional')
        self.Param.add("Settings_Type", 'Physical')
        self.Param.add("Maximum_subiters_recon", 10)
        self.Param.add("Tolerance_recon", 1.0e-4)
        self.Param.add("Saving_Frequency", 50)
        self.Param.add("Reinitialization_Frequency", 1)
        self.Param.add("Saving_Directory", 'Sim')
        self.Param.add("Interface_Perturbation_RT", 'Cos')
        self.Param.add("Problem", 'Bubble')

        try:
            self.file = open(param_name, "r")
        except IOError:
            print("Input parameter file '" + param_name + "' not found. Creating a default one (based on the rising bubble benchmark)")
            f = open(param_name, "w")
            f.write("Gravity = 0.98\n")
            f.write("Surface_tension = 1.96\n")
            f.write("Lighter_density = 1.0\n")
            f.write("Heavier_density = 1000.0\n")
            f.write("Viscosity_lighter_fluid = 1.0\n")
            f.write("Viscosity_heavier_fluid = 10.0\n")
            f.write("Time_step = 0.0008\n")
            f.write("End_time = 3.0\n")
            f.write("Base = 1.0\n")
            f.write("Height = 2.0\n")
            f.write("x_center = 0.5\n")
            f.write("y_center = 0.5\n")
            f.write("Radius = 0.25\n")
            f.close()
            self.file = open(param_name, "r")

        try:
            self.parse_parameters(self.file)
            #Close the file
            self.file.close()
        except Exception as e:
            print("Caught an exception: " + str(e))


    """Parse parameters file into the standard FENICS Parameters class"""
    def parse_parameters(self, param_file):
        for line in param_file.read().splitlines():
            if line.strip(): #Avoid reading blank lines
                idx_eq = line.find(' = ')
                if(idx_eq != -1):
                    if(line[0 : idx_eq] in self.Param.keys()):
                        self.Param[line[0 : idx_eq]] = type(self.Param[line[0 : idx_eq]])(line[idx_eq + 3 :])
                    else:
                        self.Param.add(line[0 : idx_eq],line[idx_eq + 3 :])
                else:
                    raise Exception("Invalid format to read parameters. \
                                     Please check the configuration file: \
                                     you need a space before and after the equal")


    """Return the standard FENICS Parameters class"""
    def get_param(self):
        return self.Param
