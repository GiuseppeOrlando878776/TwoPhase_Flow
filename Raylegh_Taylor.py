from TwoPhaseFlows import *
from My_Parameters import My_Parameters

from sys import exit

class RayleghTaylor(TwoPhaseFlows):
    """Class constructor"""
    def __init__(self, param_name):
        """
        Param --- class Parameters to store desired configuration
        rho1  --- Lighter density
        rho2  --- Heavier density
        mu1   --- Smaller viscosity
        mu2   --- Larger viscosity
        dt    --- Specified time step
        t_end --- End time of the simulation
        deg   --- Polynomial degree
        """

        #Call the base class constructor
        super(RayleghTaylor, self).__init__()

        #Start with the specific problem settings
        self.Param = My_Parameters(param_name).get_param()

        try:
            self.rho1  = float(self.Param["Lighter_density"])
            self.rho2  = float(self.Param["Heavier_density"])
            self.mu1   = float(self.Param["Smaller_viscosity"])
            self.mu2   = float(self.Param["Larger_viscosity"])
            self.g     = float(self.Param["Gravity"])
            self.dt    = float(self.Param["Time_step"])
            self.t_end = float(self.Param["End_time"])
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")
            exit(1)

        if(self.rho2 < self.rho1):
            warnings.warn("The heavier density is not greater than the lighter one")
        if(self.mu2 < self.mu1):
            warning.warn("The larger viscosity is not greter than the smaller one")

        #Since this parameters are more related to the numeric part
        #rather than physics we set a default value
        #and so they are present for sure
        self.deg = self.Param["Polynomial_degree"]
        self.reinit_method = self.Param["Reinit_Type"]
        self.stab_method   = self.Param["Stabilization_Type"]
        self.NS_sol_method = self.Param["NS_Procedure"]

        #Check correctness of Level-set method
        assert self.stab_method in self.stab_dict, "Stabilization method not available"

        #Check correctness of Navier-Stokes method
        assert self.NS_sol_method in self.NS_sol_dict, "Solution method for Navier-Stokes not available"

        #Check correctness of reinitialization method
        assert self.reinit_method in self.reinit_method_dict, "Reinitialization method not available"

        #Check coerence of dimensional choice
        assert self.Param["Reference_Dimensionalization"] == 'Non_Dimensional', \
        "The problem 'RayleghTaylor' works in a non-dimensional framework"

        #Compute the Atwood number
        self.At = (self.rho2 - self.rho1)/(self.rho2 + self.rho1)

        #Compute Reynolds number
        self.L0 = 1.0
        self.Re = self.rho1*self.L0*np.sqrt(self.At*self.L0*self.g)/self.mu1

        #Compute density and viscosity ratio
        self.rho2_rho1 = self.rho2/self.rho1
        self.mu2_mu1   = self.mu2/self.mu1

        #Compute reference time
        self.t0 = np.sqrt(self.L0*self.g)

        #Prepare useful variables for stabilization
        self.switcher_parameter = {self.stab_method: None}
        if(self.stab_method == 'IP'):
            self.alpha = self.Param["Stabilization_Parameter"]
            #Auxiliary dictionary in order to set the proper parameter for stabilization
            self.switcher_parameter['IP'] = self.alpha
        elif(self.stab_method == 'SUPG'):
            #Auxiliary dictionary in order to set the proper parameter for stabilization
            self.switcher_parameter['SUPG'] = self.Re

        #Convert useful constants to constant FENICS functions
        self.DT = Constant(self.dt)

        #Set parameter for standard output
        set_log_level(self.Param["Log_Level"])

        #Detect properties for reconstrution step
        self.tol_recon = self.Param["Tolerance_recon"]
        self.max_subiters = self.Param["Maximum_subiters_recon"]


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

        #Parameters for reinitialization steps
        if(self.reinit_method == 'Non_Conservative_Hyperbolic'):
            hmin = self.mesh.hmin()
            self.eps = self.Param["Interface_Thickness"]
            self.gamma_reinit = Constant(hmin)
            self.beta_reinit = Constant(0.0625*hmin)
            self.dt_reinit = Constant(np.minimum(0.0001, 0.5*hmin)) #We choose an explicit treatment to keep the linearity
                                                                    #and so a very small step is needed
        elif(self.reinit_method == 'Non_Conservative_Elliptic'):
            self.eps = self.Param["Interface_Thickness"]
            self.beta_reinit = Constant(self.Param["Penalization_Reconstruction"])
            self.gamma_reinit = None
            self.dt_reinit = None
        elif(self.reinit_method == 'Conservative'):
            hmin = self.mesh.hmin()
            d = self.Param["Extra_Power_Conservative_LevSet"]
            self.dt_reinit = Constant(0.5*hmin**(1.0 + d))
            self.eps = Constant(0.5*hmin**(1.0 - d))
            self.gamma_reinit = None
            self.beta_reinit = None

        #Define FE spaces
        Velem = VectorElement("Lagrange", self.mesh.ufl_cell(), self.deg + 1)
        Pelem = FiniteElement("Lagrange" if self.deg > 0 else "DG", self.mesh.ufl_cell(), self.deg)
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

        #Define function and vector for plotting level-set and computing volume
        self.rho_interp = Function(self.Q)

        #Declare function for normal vector
        self.n = Function(self.Q2)

        #Prepare useful dictionaries in order to avoid too many ifs
        self.switcher_reinit_varf = {'Non_Conservative_Hyperbolic': self.NCLSM_hyperbolic_weak_form, \
                                     'Non_Conservative_Elliptic': self.NCLSM_elliptic_weak_form, \
                                     'Conservative': self.CLSM_weak_form}
        self.switcher_arguments_reinit_varf = {'Non_Conservative_Hyperbolic': \
                                               (self.phi, self.l, self.phi0, self.phi_curr, self.dt_reinit, self.gamma_reinit, self.beta_reinit), \
                                               'Non_Conservative_Elliptic': \
                                                (self.phi, self.l, self.phi0, self.phi_curr, self.Appr_Delta, self.eps, self.beta_reinit), \
                                               'Conservative': \
                                               (self.phi_intermediate, self.l, self.phi0, self.n, self.dt_reinit, self.eps)}
        self.switcher_reinit_solve = {'Non_Conservative_Hyperbolic': self.NC_Levelset_hyperbolic_reinit, \
                                      'Non_Conservative_Elliptic': self.NC_Levelset_elliptic_reinit, \
                                      'Conservative': self.C_Levelset_reinit}
        self.switcher_arguments_reinit_solve = {'Non_Conservative_Hyperbolic': \
                                                (self.phi_curr, self.phi_intermediate, self.phi0, self.dt_reinit, \
                                                 self.n, self.Q2, self.max_subiters, self.tol_recon), \
                                                'Non_Conservative_Elliptic': \
                                                (self.phi_curr, self.phi_intermediate, self.phi0, \
                                                 self.n, self.Q2, self.max_subiters, self.tol_recon), \
                                                'Conservative': \
                                                (self.phi_curr, self.phi_intermediate, self.phi0, self.dt_reinit,
                                                 self.n, self.Q2, self.max_subiters, self.tol_recon)}

    """Weak formulation for Navier-Stokes"""
    def NS_weak_form(self):
        #Set weak formulation
        F2 = (1.0/self.DT)*inner(self.rho(self.phi_curr, self.eps)*self.u - self.rho(self.phi_old, self.eps)*self.u_old, self.v)*dx \
           + inner(self.rho(self.phi_curr, self.eps)*dot(self.u_old, nabla_grad(self.u)), self.v)*dx \
           + (2.0/self.Re)*inner(self.mu(self.phi_curr, self.eps)*D(self.u), D(self.v))*dx \
           - self.p*div(self.v)*dx \
           + div(self.u)*self.q*dx \
           + (1.0/self.At)*inner(self.rho(self.phi_curr, self.eps)*self.e2, self.v)*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()


    """Weak formulation for tentative velocity"""
    def ICT_weak_form_1(self):
        #Define variational formulation for step 1
        F2 = (1.0/self.DT)*inner(self.rho(self.phi_curr, self.eps)*self.u - self.rho(self.phi_old, self.eps)*self.u_old, self.v)*dx \
           + inner(self.rho(self.phi_curr, self.eps)*dot(self.u_old, nabla_grad(self.u)), self.v)*dx \
           + (2.0/self.Re)*inner(self.mu(self.phi_curr, eps)*D(self.u), D(self.v))*dx \
           - self.p_old*div(self.v)*dx \
           + (1.0/self.At)*inner(self.rho(self.phi_curr, self.eps)*self.e2, self.v)*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()


    """Set the proper initial condition"""
    def set_initial_condition(self):
        #Assign initial condition
        self.u_old.assign(interpolate(Constant((0.0,0.0)), self.V))
        self.p_old.assign(interpolate(Constant(0.0), self.P))
        if(self.reinit_method == 'Non_Conservative_Elliptic' or self.reinit_method == 'Non_Conservative_Hyperbolic'):
            f = Expression("tanh((x[1] - 2.0 - 0.1*cos(2*pi*x[0]))/(0.01*sqrt(2.0)))", degree = 8)
            self.phi_old.assign(interpolate(f,self.Q))
        elif(self.reinit_method == 'Conservative'):
            f = Expression("1.0/(1.0 + exp(tanh((x[1] - 2.0 - 0.1*cos(2*pi*x[0]))/(0.01*sqrt(2.0)))/eps))", eps = self.eps, degree = 8)
            self.phi_old.assign(interpolate(f, self.Q))


    """Assemble boundary condition"""
    def assembleBC(self):
        if(self.NS_sol_method == 'Standard'):
            self.bcs = [DirichletBC(self.W.sub(0), Constant((0.0,0.0)),  'near(x[1], 0.0) || near(x[1], 4.0)'), \
                        DirichletBC(self.W.sub(0).sub(0), Constant(0.0), 'near(x[0], 0.0) || near(x[0], 1.0)')]
        elif(self.NS_sol_method == 'ICT'):
            self.bcs = [DirichletBC(self.V, Constant((0.0,0.0)),  'near(x[1], 0.0) || near(x[1], 4.0)'), \
                        DirichletBC(self.V.sub(0), Constant(0.0), 'near(x[0], 0.0) || near(x[0], 1.0)')]

        #Useful dictionaries in order to avoid too many ifs
        self.switcher_NS_solve = {'Standard': self.solve_Standard_NS_system, \
                                  'ICT': self.solve_ICT_NS_systems}
        self.switcher_arguments_NS_solve = {'Standard': (self.bcs, self.w_curr, self.u_curr, self.p_curr), \
                                            'ICT': (self.bcs, self.u_curr, self.p_curr)}


    """Auxiliary function to select proper Heavised approximation"""
    def Appr_Heaviside(self, x, eps):
        if(self.reinit_method == 'Non_Conservative_Hyperbolic' or self.reinit_method == 'Non_Conservative_Elliptic'):
            return CHeaviside(x, eps)
        elif(self.reinit_method == 'Conservative'):
            return x


    """Auxiliary function to select proper Dirac's delta approximation"""
    def Appr_Delta(self, x, eps):
        if(self.reinit_method == 'Non_Conservative_Hyperbolic' or self.reinit_method == 'Non_Conservative_Elliptic'):
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
                self.NS_weak_form()
            elif(self.NS_sol_method == 'ICT'):
                self.ICT_weak_form_1()
                self.ICT_weak_form_2(self.p, self.q, self.DT, self.u_curr, self.rho, self.phi_curr, self.eps)
                self.ICT_weak_form_3(self.u, self.v, self.DT, self.u_curr, self.p_curr, self.p_old, self.rho, self.phi_curr, self.eps)
        except ValueError as e:
            print(str(e))
            print("Aborting simulation...")
            exit(1)


    """Plot the level-set function and compute the volume"""
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

        #Time-stepping loop
        self.t = 0.0
        self.n_iter = 0

        #File for plotting
        save_iters = self.Param["Saving_Frequency"]
        self.vtkfile_u = File('/u/archive/laureandi/orlando/Sim115/u.pvd')
        self.vtkfile_rho = File('/u/archive/laureandi/orlando/Sim115/rho.pvd')

        #Save initial state and start loop
        self.plot_and_save()
        self.t += self.dt
        while self.t <= self.t_end:
            begin(int(LogLevel.INFO) + 1,"t = " + str(self.t*self.t0) + " s")
            self.n_iter += 1

            #Solve level-set
            begin(int(LogLevel.INFO) + 1,"Solving Level-set")
            self.solve_Levelset_system(self.phi_curr, self.n, self.Q2)
            end()

            #Solve Level-set reinit
            try:
                begin(int(LogLevel.INFO) + 1,"Solving reinitialization")
                self.switcher_reinit_solve[self.reinit_method](*self.switcher_arguments_reinit_solve[self.reinit_method])
                end()
            except Exception as e:
                print(str(e))
                print("Aborting simulation...")
                exit(1)

            #Solve Navier-Stokes
            begin(int(LogLevel.INFO) + 1,"Solving Navier-Stokes")
            self.switcher_NS_solve[self.NS_sol_method](*self.switcher_arguments_NS_solve[self.NS_sol_method])
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

            self.t += self.dt if self.t + self.dt <= self.t_end or abs(self.t - self.t_end) < DOLFIN_EPS else self.t_end
