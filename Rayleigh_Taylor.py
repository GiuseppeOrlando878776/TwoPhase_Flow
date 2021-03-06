from TwoPhaseFlows import *
from My_Parameters import My_Parameters
from Boundary_Definition import *

from sys import exit
import os

class RayleighTaylor(TwoPhaseFlows):
    """Class constructor"""
    def __init__(self, param_handler):
        """
        Param --- class Parameters to store desired configuration
        mu1   --- Viscosity_lighter_fluid
        mu2   --- Viscosity_heavier_fluid
        g     --- Gravity force
        dt    --- Specified time step
        t_end --- End time of the simulation
        deg   --- Polynomial degree
        """

        #Call the base class constructor
        super(RayleighTaylor, self).__init__()

        #MPI settings
        self.comm = MPI.comm_world
        self.rank = MPI.rank(self.comm)

        #Start with the specific problem settings
        self.Param = param_handler

        #Check coherence of dimensional choice
        if(self.Param["Reference_Dimensionalization"] != 'Non_Dimensional'):
            raise ValueError("This instance of the problem 'RayleighTaylor' works in a non-dimensional framework")

        try:
            self.set_type = self.Param["Settings_Type"]
            self.mu1      = float(self.Param["Viscosity_lighter_fluid"])
            self.mu2      = float(self.Param["Viscosity_heavier_fluid"])
            self.g        = float(self.Param["Gravity"])
            self.dt       = float(self.Param["Time_step"])
            self.t_end    = float(self.Param["End_time"])
        except RuntimeError as e:
            if(self.rank == 0):
                print(str(e) +  "\nPlease check configuration file")
            exit(1)

        if(self.set_type not in {'Physical', 'Parameters'}):
            raise ValueError("Unknown value for settings values")

        #Check correctness of data read
        if(self.dt < DOLFIN_EPS or self.t_end < DOLFIN_EPS or self.mu1 < DOLFIN_EPS or \
           self.mu2 < DOLFIN_EPS or self.g < DOLFIN_EPS):
            raise ValueError("Invalid parameter in the configuration file (read a non positive value for some parameters)")
        if(self.dt > self.t_end):
            raise ValueError("Time-step greater than final time")

        #Since this parameters are more related to the numeric part
        #rather than physics we set a default value
        #and so they are present for sure
        self.deg = self.Param["Polynomial_degree"]
        self.reinit_method = self.Param["Reinit_Type"]
        self.stab_method   = self.Param["Stabilization_Type"]
        self.NS_sol_method = self.Param["NS_Procedure"]

        #Check correctness of Level-set method
        if(self.stab_method not in self.stab_dict):
            raise ValueError("Stabilization method not available")

        #Check correctness of Navier-Stokes method
        if(self.NS_sol_method not in self.NS_sol_dict):
            raise ValueError("Solution method for Navier-Stokes not available")

        #Check correctness of reinitialization method
        if(self.reinit_method not in self.reinit_method_dict):
            raise ValueError("Reinitialization method not available")

        #Set more adequate solvers in case of one core execution
        if(MPI.size(self.comm) == 1):
            if(self.reinit_method == 'Non_Conservative_Hyperbolic'):
                self.solver_recon = "cg"
                self.precon_recon = "icc"
            if(self.NS_sol_method == 'Standard'):
                self.solver_Standard_NS = "umfpack"
            elif(self.NS_sol_method == 'ICT'):
                self.solver_ICT_3 = "cg"
                self.precon_ICT_3 = "icc"

        self.L0 = 1.0 #Reference length
        #Compute the Atwood number and the Reynolds number according to how the settings has been imposed
        if(self.set_type == 'Physical'):
            try:
                self.rho1 = float(self.Param["Lighter_density"])
                self.rho2 = float(self.Param["Heavier_density"])
                self.At = (self.rho2 - self.rho1)/(self.rho2 + self.rho1)
                if(self.rho2 < self.rho1 and self.rank == 0):
                    warnings.warn("The heavier density is not greater than the lighter one")
                self.Re = self.rho1*self.L0*np.sqrt(self.At*self.L0*self.g)/self.mu1
                assert self.Re > 1.0, "Invalid Reynolds number computed"
            except RuntimeError as e:
                if(self.rank == 0):
                    print(str(e) +  "\nPlease check configuration file")
                exit(1)
        elif(self.set_type == 'Parameters'):
            try:
                self.At = float(self.Param["Atwood_number"])
                if(self.At < 0.0 or self.At > 1.0):
                    raise ValueError("Invalid Atwood number")
                self.Re = float(self.Param["Reynolds_number"])
                if(self.Re < 1.0):
                    raise ValueError("Invalid Reynolds number")
                self.rho1 = self.Re*self.mu1/(self.L0*np.sqrt(self.At*self.L0*self.g))
                self.rho2 = self.rho1*(1.0 + self.At)/(1.0 - self.At)
            except RuntimeError as e:
                if(self.rank == 0):
                    print(str(e) +  "\nPlease check configuration file")
                exit(1)

        #Compute density and viscosity ratio
        self.rho2_rho1 = self.rho2/self.rho1
        self.mu2_mu1   = self.mu2/self.mu1

        #Compute reference time: we assume that self.dt is the one that the user wants to use in the weak
        #form (the step) while self.t_end is the dimensional final time
        self.U0 = np.sqrt(self.L0*self.At*self.g)
        self.t0 = self.L0/self.U0
        self.t_stop = self.t_end/self.t0

        #Prepare useful variables for stabilization
        self.switcher_parameter = {self.stab_method: None}
        if(self.stab_method == 'IP'):
            self.alpha = self.Param["Stabilization_Parameter"]
            #Auxiliary dictionary in order to set the proper parameter for stabilization
            self.switcher_parameter['IP'] = self.alpha
            #Share interior facets
            if(MPI.size(self.comm) > 1):
                parameters["ghost_mode"] = "shared_facet"
        elif(self.stab_method == 'SUPG'):
            self.scaling = self.Param["Stabilization_Parameter"]
            #Auxiliary dictionary in order to set the proper parameter for stabilization
            self.switcher_parameter['SUPG'] = self.scaling

        #Convert useful constants to constant FENICS functions
        self.DT = Constant(self.dt)

        #Set parameter for standard output (only rank 0 will print)
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
            if(self.rank == 0):
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

        #Define trial and test functions
        if(self.NS_sol_method == 'Standard'):
            (self.u, self.p) = TrialFunctions(self.W)
            (self.v, self.q) = TestFunctions(self.W)
        elif(self.NS_sol_method == 'ICT'):
            self.u = TrialFunction(self.V)
            self.v = TestFunction(self.V)
            self.p = TrialFunction(self.P)
            self.q = TestFunction(self.P)
        self.phi = TrialFunction(self.Q)
        self.l   = TestFunction(self.Q)

        #Define functions to store solution
        self.u_curr   = Function(self.V)
        self.u_old    = Function(self.V)
        self.p_curr   = Function(self.P)
        self.p_old    = Function(self.P)
        if(self.NS_sol_method == 'Standard'):
            self.w_curr = Function(self.W)
        self.phi_curr = Function(self.Q)
        self.phi_old  = Function(self.Q)

        #Define useful functions for reinitialization
        self.phi0 = Function(self.Q)
        self.phi_intermediate = Function(self.Q) #This is fundamental in case on 'non-conservative'
                                                 #reinitialization and it is also useful for clearness

        #Define function and vector for saving purposes
        self.rho_interp = Function(self.Q)

        #Declare function for normal vector (in case of conservative level-set method)
        if(self.reinit_method == 'Conservative'):
            self.Q2 = VectorFunctionSpace(self.mesh, "CG", 1)
            self.n = Function(self.Q2)


        #Parameters for reinitialization steps
        if(self.reinit_method == 'Non_Conservative_Hyperbolic'):
            hmin = MPI.min(self.comm, self.mesh.hmin())
            self.eps = self.Param["Interface_Thickness"]
            if(self.eps < DOLFIN_EPS):
                raise ValueError("Non-Positive value for the interface thickness")
            self.gamma_reinit = Constant(hmin)
            self.beta_reinit = Constant(0.0625*hmin)
            self.dt_reinit = Constant(np.minimum(0.0001, 0.5*hmin)) #We choose an explicit treatment to keep the linearity
                                                                    #and so a very small step is needed

            #Prepare useful dictionary in order to avoid too many ifs:
            #Dictionary for reinitialization weak form
            self.switcher_reinit_varf = {'Non_Conservative_Hyperbolic': self.NCLSM_hyperbolic_weak_form}
            self.switcher_arguments_reinit_varf = {'Non_Conservative_Hyperbolic': \
                                                   (self.phi, self.l, self.phi0, self.phi_curr, self.dt_reinit, \
                                                    self.gamma_reinit, self.beta_reinit)}

            #Dictionary for reinitialization solution
            self.switcher_reinit_solve = {'Non_Conservative_Hyperbolic': self.NC_Levelset_hyperbolic_reinit}
            self.switcher_arguments_reinit_solve = {'Non_Conservative_Hyperbolic': \
                                                    (self.phi_curr, self.phi_intermediate, self.phi0, self.dt_reinit, \
                                                     self.max_subiters, self.tol_recon)}
        elif(self.reinit_method == 'Conservative'):
            hmin = MPI.min(self.comm, self.mesh.hmin())
            self.dt_reinit = Constant(0.5*hmin**(1.1))
            self.eps = Constant(0.5*hmin**(0.9))

            #Prepare useful dictionary in order to avoid too many ifs:
            #Dictionary for reinitialization weak form
            self.switcher_reinit_varf = {'Conservative': self.CLSM_weak_form}
            self.switcher_arguments_reinit_varf = {'Conservative': \
                                                   (self.phi_intermediate, self.l, self.phi0, self.n, self.dt_reinit, self.eps)}

            #Dictionary for reinitialization solution
            self.switcher_reinit_solve = {'Conservative': self.C_Levelset_reinit}
            self.switcher_arguments_reinit_solve = {'Conservative': \
                                                    (self.phi_curr, self.phi_intermediate, self.phi0, self.dt_reinit, \
                                                     self.max_subiters, self.tol_recon)}


    """Set the proper initial condition"""
    def set_initial_condition(self):
        #Assign initial condition
        self.u_old.assign(interpolate(Constant((0.0,0.0)), self.V))
        self.p_old.assign(interpolate(Constant(0.0), self.P))

        #Check type of initial perturbation
        Interface_Perturbation_RT = self.Param["Interface_Perturbation_RT"]
        if(Interface_Perturbation_RT not in {'Cos', 'Tanh'}):
            raise ValueError("Invalid parameter for initial perturbation for RT instability")

        #Assign proper initial condition according to perturbation choice
        if(self.reinit_method == 'Non_Conservative_Hyperbolic'):
            if(Interface_Perturbation_RT == 'Tanh'):
                f = Expression("tanh((x[1] - A - 0.1*cos(2*pi*x[0]))/(0.01*sqrt(2.0)))", A = self.height/2.0, degree = 8)
            else:
                f = Expression("x[1] - A - 0.01*B*cos(2*pi*x[0])", A = self.height/2.0, B = self.height, degree = 8)
            self.phi_old.assign(interpolate(f,self.Q))
        elif(self.reinit_method == 'Conservative'):
            if(Interface_Perturbation_RT == 'Tanh'):
                f = Expression("1.0/(1.0 + exp(-tanh((x[1] - A - 0.1*cos(2*pi*x[0]))/(0.01*sqrt(2.0)))/eps))", \
                                A = self.height/2.0, eps = self.eps, degree = 8)
            else:
                f = Expression("1.0/(1.0 + exp(-(x[1] - A - 0.01*B*cos(2*pi*x[0]))/eps))", \
                                A = self.height/2.0, B = self.height, eps = self.eps, degree = 8)
            self.phi_old.assign(interpolate(f, self.Q))


    """Assemble boundary condition"""
    def assembleBC(self):
        if(self.NS_sol_method == 'Standard'):
            self.bcs = [DirichletBC(self.W.sub(0), Constant((0.0,0.0)),  NoSlip_Boundary(self.height)), \
                        DirichletBC(self.W.sub(0).sub(0), Constant(0.0), FreeSlip_Boundary(self.base))]

            #Useful dictionary for solver in order to avoid too many ifs
            self.switcher_NS_solve = {'Standard': self.solve_Standard_NS_system}
            self.switcher_arguments_NS_solve = {'Standard': (self.bcs, self.w_curr)}
        elif(self.NS_sol_method == 'ICT'):
            self.bcs = [DirichletBC(self.V, Constant((0.0,0.0)),  NoSlip_Boundary(self.height)), \
                        DirichletBC(self.V.sub(0), Constant(0.0), FreeSlip_Boundary(self.base))]

            #Useful dictionary for solver in order to avoid too many ifs
            self.switcher_NS_solve = {'ICT': self.solve_ICT_NS_systems}
            self.switcher_arguments_NS_solve = {'ICT': (self.bcs, self.u_curr, self.p_curr)}


    """Auxiliary function to select proper Heavised approximation"""
    def Appr_Heaviside(self, x, eps):
        if(self.reinit_method == 'Non_Conservative_Hyperbolic'):
            return CHeaviside(x, eps)
        elif(self.reinit_method == 'Conservative'):
            return x


    """Auxiliary function to select proper Dirac's delta approximation"""
    def Appr_Delta(self, x, eps):
        if(self.reinit_method == 'Non_Conservative_Hyperbolic'):
            return CDelta(x, eps)
        elif(self.reinit_method == 'Conservative'):
            return 1.0


    """Auxiliary function to compute density"""
    def rho(self, x, eps):
        return self.rho2_rho1*self.Appr_Heaviside(x,eps) + (1.0 - self.Appr_Heaviside(x,eps))


    """Auxiliary function to compute viscosity"""
    def mu(self, x, eps):
        return self.mu2_mu1*self.Appr_Heaviside(x,eps) + (1.0 - self.Appr_Heaviside(x,eps))


    """Set weak formulations"""
    def set_weak_forms(self):
        try:
            #Set variational problem for step 1 (Level-set)
            self.LS_weak_form(self.phi, self.l, self.phi_old, self.u_old, self.DT, self.mesh, \
                              self.stab_method, self.switcher_parameter[self.stab_method])

            #Set variational problem for reinitialization
            self.switcher_reinit_varf[self.reinit_method](*self.switcher_arguments_reinit_varf[self.reinit_method])

            #Set variational problem for step 2 (Navier-Stokes)
            if(self.NS_sol_method == 'Standard'):
                self.NS_weak_form(self.u, self.p, self.v, self.q, self.u_old, self.DT, self.rho, self.mu, \
                                  self.phi_curr, self.phi_old, self.eps, Re = self.Re, Fr = np.sqrt(self.At), We = 0.0)
            elif(self.NS_sol_method == 'ICT'):
                self.ICT_weak_form_1(self.u, self.v, self.u_old, self.p_old, self.DT, self.rho, self.mu, \
                                     self.phi_curr, self.phi_old, self.eps, Re = self.Re, Fr = np.sqrt(self.At), We = 0.0)
                self.ICT_weak_form_2(self.p, self.q, self.DT, self.p_old, self.u_curr, self.rho, self.phi_curr, self.eps)
                self.ICT_weak_form_3(self.u, self.v, self.DT, self.u_curr, self.p_curr, self.p_old, self.rho, self.phi_curr, self.eps)
        except ValueError as e:
            if(self.rank == 0):
                print(str(e))
                print("Aborting simulation...")
            exit(1)


    """Save the actual state for post-processing"""
    def plot_and_save(self):
        #Save the actual state for visualization
        self.vtkfile_u << (self.u_old, self.t*self.t0)
        self.rho_interp.assign(project(self.rho(self.phi_old,self.eps), self.Q))
        self.vtkfile_rho << (self.rho_interp, self.t*self.t0)


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
        reinit_iters = self.Param["Reinitialization_Frequency"]
        save_iters = self.Param["Saving_Frequency"]

        #File for plotting
        self.vtkfile_u = File(os.getcwd() + '/' + self.Param["Saving_Directory"] + '/u.pvd')
        self.vtkfile_rho = File(os.getcwd() + '/' + self.Param["Saving_Directory"] + '/rho.pvd')

        #Save initial state and start loop
        self.plot_and_save()
        self.t += self.dt
        while self.t <= self.t_stop:
            begin(int(LogLevel.INFO) + 1,"t = " + str(self.t*self.t0) + " s")
            self.n_iter += 1

            #Solve level-set
            begin(int(LogLevel.INFO) + 1,"Solving Level-set")
            self.solve_Levelset_system(self.phi_curr)
            end()

            #Solve Level-set reinit
            if(self.n_iter % reinit_iters == 0):
                try:
                    begin(int(LogLevel.INFO) + 1,"Solving reinitialization")
                    if(self.reinit_method == 'Conservative'):
                        self.n.assign(project(grad(self.phi_curr)/mgrad(self.phi_curr), self.Q2))
                    self.switcher_reinit_solve[self.reinit_method](*self.switcher_arguments_reinit_solve[self.reinit_method])
                    end()
                except Exception as e:
                    if(self.rank == 0):
                        print(str(e))
                        print("Aborting simulation...")
                    exit(1)

            #Solve Navier-Stokes
            begin(int(LogLevel.INFO) + 1,"Solving Navier-Stokes")
            self.switcher_NS_solve[self.NS_sol_method](*self.switcher_arguments_NS_solve[self.NS_sol_method])
            if(self.NS_sol_method == 'Standard'):
                (self.u_curr, self.p_curr) = self.w_curr.split(True)
            end()

            #Prepare to next step assign previous-step solution
            self.u_old.assign(self.u_curr)
            self.p_old.assign(self.p_curr)
            self.phi_old.assign(self.phi_curr)

            #Save and compute benchmark quantities
            if(self.n_iter % save_iters == 0):
                begin(int(LogLevel.INFO) + 1,"Saving data")
                self.plot_and_save()
                end()

            end()

            self.t = self.t + self.dt if self.t + self.dt <= self.t_stop or abs(self.t - self.t_stop) < DOLFIN_EPS else self.t_stop

        #Save the final state
        if(self.n_iter % save_iters != 0):
            self.plot_and_save()
