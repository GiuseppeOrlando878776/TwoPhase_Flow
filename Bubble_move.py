from TwoPhaseFlows import *
from My_Parameters import My_Parameters
from Boundary_Definition import *

from sys import exit
import os

class BubbleMove(TwoPhaseFlows):
    """Class constructor"""
    def __init__(self, param_handler):
        """
        Param --- class Parameters to store desired configuration
        rho1  --- Lighter density
        rho2  --- Heavier density
        mu1   --- Viscosity lighter fluid
        mu2   --- Viscosity heavier fluid
        dt    --- Specified time step
        t_end --- End time of the simulation
        deg   --- Polynomial degree
        """

        #Call the base class constructor
        super(BubbleMove, self).__init__()

        #Start with the specific problem settings
        self.Param = param_handler

        #MPI settings
        self.comm = MPI.comm_world
        self.rank = MPI.rank(self.comm)

        #Check coherence of dimensional choice
        if(self.Param["Reference_Dimensionalization"] != 'Dimensional'):
            raise ValueError("This instance of the problem 'BubbleMove' works in a dimensional framework")

        try:
            self.rho1  = float(self.Param["Lighter_density"])
            self.rho2  = float(self.Param["Heavier_density"])
            self.mu1   = float(self.Param["Viscosity_lighter_fluid"])
            self.mu2   = float(self.Param["Viscosity_heavier_fluid"])
            self.g     = float(self.Param["Gravity"])
            self.sigma = float(self.Param["Surface_tension"])
            self.dt    = float(self.Param["Time_step"])
            self.t_end = float(self.Param["End_time"])
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")
            exit(1)

        #Check correctness of data read
        if(self.rho1 < DOLFIN_EPS or self.rho2 < DOLFIN_EPS or self.mu1 < DOLFIN_EPS or self.mu2 < DOLFIN_EPS \
           or self.dt < DOLFIN_EPS or self.t_end < DOLFIN_EPS or self.g < 0.0 or self.sigma < 0.0):
            raise ValueError("Invalid parameter read in the configuration file (read a non positive value for some parameters)")
        if(self.dt > self.t_end):
            raise ValueError("Time-step greater than final time")
        if(self.rho2 < self.rho1 and self.rank == 0):
            warnings.warn("The heavier density is not greater than the lighter one")

        #Since this parameters are more related to the numeric part
        #rather than physics we set a default value
        #and so they are present for sure
        self.deg = self.Param["Polynomial_degree"]
        self.stab_method   = self.Param["Stabilization_Type"]
        self.NS_sol_method = self.Param["NS_Procedure"]

        #Check correctness of Level-set method
        if(self.stab_method not in self.stab_dict):
            raise ValueError("Stabilization method not available")

        #Check correctness of Navier-Stokes method
        if(self.NS_sol_method not in self.NS_sol_dict):
            raise ValueError("Solution method for Navier-Stokes not available")

        #Set more adequate solvers in case of one core execution
        if(MPI.size(self.comm) == 1):
            if(self.NS_sol_method == 'Standard'):
                self.solver_Standard_NS = "umfpack"
            elif(self.NS_sol_method == 'ICT'):
                self.solver_ICT_3 = "cg"
                self.precon_ICT_3 = "icc"

        #Prepare useful variables for stabilization
        self.switcher_parameter = {self.stab_method: None}
        if(self.stab_method == 'IP'):
            self.alpha = Constant(self.Param["Stabilization_Parameter"])
            #Auxiliary dictionary in order to set the proper parameter for stabilization
            self.switcher_parameter['IP'] = self.alpha
            #Share interior facets
            if(MPI.size(self.comm) > 1):
                parameters["ghost_mode"] = "shared_facet"
        elif(self.stab_method == 'SUPG'):
            self.scaling = Constant(self.Param["Stabilization_Parameter"])
            #Auxiliary dictionary in order to set the proper parameter for stabilization
            self.switcher_parameter['SUPG'] = self.scaling

        #Convert useful constants to constant FENICS functions
        self.DT   = Constant(self.dt)
        self.g    = Constant(self.g)

        #Set parameter for standard output
        set_log_level(self.Param["Log_Level"] if self.rank == 0 else 1000)

        #Detect properties for reconstrution step
        self.tol_recon = self.Param["Tolerance_recon"]
        self.max_subiters = self.Param["Maximum_subiters_recon"]


    """Return the communicator"""
    def get_communicator(self):
        return self.comm


    """Build the mesh for the simulation"""
    def build_mesh(self):
        #Generate mesh
        try:
            self.base   = float(self.Param["Base"])
            self.height = float(self.Param["Height"])
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")
            exit(1)
        self.mesh = RectangleMesh(Point(0.0, 0.0), Point(self.base, self.height), \
                                  self.Param["Number_vertices_x"], self.Param["Number_vertices_y"])

        #Define FE spaces
        if(self.deg == 0):
            raise ValueError("Invalid degree for polynomials employed in Navier-Stokes (the pair P1-P0 is not stable)")
        Velem = VectorElement("Lagrange", self.mesh.ufl_cell(), self.deg + 1)
        Pelem = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.deg)
        self.V  = FunctionSpace(self.mesh, Velem)
        self.P  = FunctionSpace(self.mesh, Pelem)
        if(self.NS_sol_method == 'Standard'):
            self.W  = FunctionSpace(self.mesh, Velem*Pelem)
        self.Q  = FunctionSpace(self.mesh, "CG", 2)
        self.Q2 = VectorFunctionSpace(self.mesh, "CG", 1)

        #Define trial and test functions
        if(self.NS_sol_method == 'Standard'):
            (self.u, self.p) = TrialFunctions(self.W)
            (self.v, self.q) = TestFunctions(self.W)
        elif(self.NS_sol_method == 'ICT'):
            self.u = TrialFunction(self.V)
            self.v = TestFunction(self.V)
            self.p = TrialFunction(self.P)
            self.q = TestFunction(self.P)
        self.alpha = TrialFunction(self.Q)
        self.l   = TestFunction(self.Q)

        #Define functions to store solution
        self.u_curr   = Function(self.V)
        self.u_old    = Function(self.V)
        self.p_curr   = Function(self.P)
        self.p_old    = Function(self.P)
        if(self.NS_sol_method == 'Standard'):
            self.w_curr = Function(self.W)
        self.alpha_curr = Function(self.Q)
        self.alpha_old  = Function(self.Q)

        #Define function to store the normal
        self.n = Function(self.Q2)

        #Define function and vector for plotting and computing volume
        self.rho_interp = Function(self.Q)


    """"Set the proper initial condition"""
    def set_initial_condition(self):
        #Read from configuration file center and radius
        try:
            center = Point(float(self.Param["x_center"]), float(self.Param["y_center"]))
            radius = float(self.Param["Radius"])
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")
            exit(1)

        #Check geoemtric limits
        if(center[0] - radius < 0.0 or center[0] + radius > self.base or \
           center[1] - radius < 0.0 or center[1] + radius > self.height):
           raise ValueError("Initial condition of interface goes outside the domain")

        #Assign initial condition
        self.u_old.assign(interpolate(Constant((0.0,0.0)), self.V))
        self.p_old.assign(interpolate(Constant(0.0), self.P))
        f = Expression("sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B)) - r < 0 ? 0 : 1",
                        A = center[0], B = center[1], r = radius, degree = 2)
        self.alpha_old.assign(interpolate(f, self.Q))


    """Assemble boundary condition"""
    def assembleBC(self):
        if(self.NS_sol_method == 'Standard'):
            self.bcs = [DirichletBC(self.W.sub(0), Constant((0.0,0.0)),  NoSlip_Boundary(self.height)), \
                        DirichletBC(self.W.sub(0).sub(0), Constant(0.0), FreeSlip_Boundary(self.base))]

            #Useful dictionaries for solver in order to avoid too many ifs
            self.switcher_NS_solve = {'Standard': self.solve_Standard_NS_system}
            self.switcher_arguments_NS_solve = {'Standard': (self.bcs, self.w_curr)}
        elif(self.NS_sol_method == 'ICT'):
            self.bcs = [DirichletBC(self.V, Constant((0.0,0.0)),  NoSlip_Boundary(self.height)), \
                        DirichletBC(self.V.sub(0), Constant(0.0), FreeSlip_Boundary(self.base))]

            #Useful dictionaries for solver in order to avoid too many ifs
            self.switcher_NS_solve = {'ICT': self.solve_ICT_NS_systems}
            self.switcher_arguments_NS_solve = {'ICT': (self.bcs, self.u_curr, self.p_curr)}


    """Auxiliary function to select proper Heavised approximation"""
    def Appr_Heaviside(self, x):
        return x


    """Auxiliary function to select proper Dirac's delta approximation"""
    def Appr_Delta(self, x):
        return 1.0


    """Auxiliary function to compute density"""
    def rho(self, x):
        return self.rho2*self.Appr_Heaviside(x) + self.rho1*(1.0 - self.Appr_Heaviside(x))


    """Auxiliary function to compute viscosity"""
    def mu(self, x):
        return self.mu2*self.Appr_Heaviside(x) + self.mu1*(1.0 - self.Appr_Heaviside(x))


    """Set weak formulations"""
    def set_weak_forms(self):
        try:
            #Set variational problem for step 1 (Volume fraction)
            self.VF_weak_form(self.alpha, self.l, self.alpha_old, self.u_old, self.DT, self.mesh, \
                              self.stab_method, self.switcher_parameter[self.stab_method])

            #Set variational problem for step 2 (Navier-Stokes)
            if(self.NS_sol_method == 'Standard'):
                self.NS_weak_form(self.u, self.p, self.v, self.q, self.u_old, self.DT, self.rho, self.mu, \
                                  self.alpha_curr, self.alpha_old, self.n, self.Appr_Delta, g = self.g, sigma = self.sigma)
            elif(self.NS_sol_method == 'ICT'):
                self.ICT_weak_form_1(self.u, self.v, self.u_old, self.p_old, self.DT, self.rho, self.mu, \
                                     self.alpha_curr, self.alpha_old, self.n, self.Appr_Delta, g = self.g, sigma = self.sigma)
                self.ICT_weak_form_2(self.p, self.q, self.DT, self.p_old, self.u_curr, self.rho, self.alpha_curr)
                self.ICT_weak_form_3(self.u, self.v, self.DT, self.u_curr, self.p_curr, self.p_old, self.rho, self.alpha_curr)
        except ValueError as e:
            if(self.rank == 0):
                print(str(e))
                print("Aborting simulation...")
            exit(1)


    """Save the current state for post-processing and compute benchmark quantities"""
    def plot_and_volume(self):
        #Save the actual state for visualization
        if(self.n_iter % self.save_iters == 0):
            self.vtkfile_u << (self.u_old, self.t)
            self.rho_interp.assign(project(self.rho(self.alpha_old), self.Q))
            self.vtkfile_rho << (self.rho_interp, self.t)

        #Compute benchamrk quantities
        Vol = assemble(conditional(lt(self.alpha_old, DOLFIN_EPS), 1.0, 0.0)*dx)
        Pa = 2.0*sqrt(np.pi*Vol)
        Pb = assemble(mgrad(self.alpha_old)*self.Appr_Delta(self.alpha_old)*dx)
        Chi = Pa/Pb
        Xc = assemble(Expression("x[0]", degree = 1)*(conditional(lt(self.alpha_old, DOLFIN_EPS), 1.0, 0.0))*dx)/Vol
        Yc = assemble(Expression("x[1]", degree = 1)*(conditional(lt(self.alpha_old, DOLFIN_EPS), 1.0, 0.0))*dx)/Vol
        Uc = assemble(inner(self.u_old,self.e1)*(conditional(lt(self.alpha_old, DOLFIN_EPS), 1.0, 0.0))*dx)/Vol
        Vc = assemble(inner(self.u_old,self.e2)*(conditional(lt(self.alpha_old, DOLFIN_EPS), 1.0, 0.0))*dx)/Vol
        L2_gradalpha = sqrt(assemble(inner(grad(self.alpha_old),grad(self.alpha_old))*dx)/(self.base*self.height))
        timeseries_vec = [self.t,Vol,Chi,Xc,Yc,Uc,Vc,L2_gradalpha]

        if(self.rank == 0):
            np.savetxt(self.timeseries, timeseries_vec)


    """Execute simulation"""
    def run(self):
        #Build the mesh
        self.build_mesh()

        #Set the initial condition
        self.set_initial_condition()

        #Assemble boundary conditions
        self.assembleBC()

        #Set weak formulations
        self.set_weak_forms()

        #Time-stepping loop parameters
        self.t = 0.0
        self.n_iter = 0
        self.save_iters = self.Param["Saving_Frequency"]

        #File for plotting
        self.vtkfile_u = File(os.getcwd() + '/' + self.Param["Saving_Directory"] + '/u.pvd')
        self.vtkfile_rho = File(os.getcwd() + '/' + self.Param["Saving_Directory"] + '/rho.pvd')

        #File for benchamrk comparisons
        self.timeseries = open(os.getcwd() + '/' + self.Param["Saving_Directory"] + '/benchmark_series.dat','wb')

        #Save initial state and start loop
        self.plot_and_volume()
        self.timeseries.close() #Close for safety in case some system fails to reach convergence
        self.t += self.dt
        while self.t <= self.t_end:
            begin(int(LogLevel.INFO) + 1,"t = " + str(self.t) + " s")
            self.n_iter += 1

            #Solve volume of fraction and update unit normal (if needed)
            begin(int(LogLevel.INFO) + 1,"Solving volume of fraction")
            self.solve_VolumeFraction_system(self.alpha_curr)
            end()
            if(self.sigma > DOLFIN_EPS):
                self.n.assign(project(grad(self.alpha_curr)/mgrad(self.alpha_curr), self.Q2)) #Compute normal vector

            #Solve Navier-Stokes
            begin(int(LogLevel.INFO) + 1,"Solving Navier-Stokes")
            self.switcher_NS_solve[self.NS_sol_method](*self.switcher_arguments_NS_solve[self.NS_sol_method])
            if(self.NS_sol_method == 'Standard'):
                (self.u_curr, self.p_curr) = self.w_curr.split(True)
            end()

            #Prepare to next step assign previous-step solution
            self.u_old.assign(self.u_curr)
            self.p_old.assign(self.p_curr)
            self.alpha_old.assign(self.alpha_curr)

            #Save and compute benchmark quantities
            begin(int(LogLevel.INFO) + 1,"Computing benchmark quantities")
            self.timeseries = open(os.getcwd() + '/' + self.Param["Saving_Directory"] + '/benchmark_series.dat','ab')
            self.plot_and_volume()
            self.timeseries.close() #Close for safety in case some system fails to reach convergence
            end()

            end()

            self.t = self.t + self.dt if self.t + self.dt <= self.t_end or abs(self.t - self.t_end) < DOLFIN_EPS else self.t_end

        #Save the final state
        if(self.n_iter % self.save_iters != 0):
            self.vtkfile_u << (self.u_old, self.t_end)
            self.rho_interp.assign(project(self.rho(self.alpha_old), self.Q))
            self.vtkfile_rho << (self.rho_interp, self.t_end)
