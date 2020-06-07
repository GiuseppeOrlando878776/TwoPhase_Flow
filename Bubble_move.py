from TwoPhaseFlows import *
from My_Parameters import My_Parameters

from sys import exit

class BubbleMove(TwoPhaseFlows):
    """Class constructor"""
    def __init__(self, param_name):
        """
        Param --- class Parameters to store desired configuration
        Re    --- Reynolds number
        At    --- Atwood number
        Bo    --- Bond number
        rho1  --- Lighter density
        rho2  --- Heavier density
        mu1   --- Smaller viscosity
        mu2   --- Larger viscosity
        dt    --- Specified time step
        t_end --- End time of the simulation
        deg   --- Polynomial degree
        """

        #Call the base class constructor
        super(BubbleMove, self).__init__()

        #Start with the specific problem settings
        self.Param = My_Parameters(param_name).get_param()

        try:
            self.Bo            = float(self.Param["Bond_number"])
            self.rho1          = float(self.Param["Lighter_density"])
            self.rho2          = float(self.Param["Heavier_density"])
            self.mu1           = float(self.Param["Smaller_viscosity"])
            self.mu2           = float(self.Param["Larger_viscosity"])
            self.base          = float(self.Param["Base"])
            self.height        = float(self.Param["Height"])
            self.dt            = float(self.Param["Time_step"])
            self.t_end         = float(self.Param["End_time"])
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")
            exit(1)

        assert self.rho2 > self.rho1, "The heavier density is not greater than the lighter"

        #Since this parameters are more related to the numeric part
        #rather than physics we set a default value
        #and so they are present for sure
        self.deg = self.Param["Polynomial_degree"]
        self.reinit_method = self.Param["Reinit_Type"]
        self.stab_method   = self.Param["Stabilization_Type"]
        self.NS_sol_method = self.Param["NS_Procedure"]

        #Define an auxiliary dictionary to set proper solution procedure for Navier-Stokes
        try:
            self.switcher_NS = {'Standard':self.solve_Standard_NS_system, \
                                'ICT':self.solve_ICT_NS_systems}
        except NameError as e:
            print("Solution procedure for solving Navier-Stokes " + str(e).split("'")[1] + \
                  " declared but not implemented")
            exit(1)
        assert self.NS_sol_method in self.switcher_NS, \
               "Solution method for Navier-Stokes not available"

        #Check correctness of reinitialization method (in order to avoid typos in particular
        #since, contrarly to stabilization, it is more difficult to think to other choises)
        #through auxiliary functions which can be eaily updated in future if needed
        try:
            self.switcher_reinit_solver = {'Non_Conservative':self.NC_Levelset_reinit, \
                                           'Conservative':self.C_Levelset_reinit}
        except NameError as e:
            print("Reinitialization method " + str(e).split("'")[1] + " declared but not implemented")
            exit(1)
        assert self.reinit_method in self.switcher_reinit_solver, \
               "Reinitialization method not available"

        #Compute density and viscosity ratio
        self.rho1_rho2 = self.rho1/self.rho2
        self.mu1_mu2 = self.mu1/self.mu2
        self.g = 0.98
        self.sigma = 1.96

        #Compute the reference length (we need it in order to adimensionalize the level-set):
        #since density and viscosity are physical properties it is reasonable to compute it from the
        #Reynolds number
        try:
            self.Re = float(self.Param["Reynolds_number"])
            assert self.Re > 1.0, "Invalid Reynolds number specified"
            self.L0 = (self.mu2*self.Re/(self.rho2*np.sqrt(self.g)))**(2/3)
        except RuntimeError as e:
            #In case Reynolds number is not present set the computational width of the box
            print("Setting reference length equal to the computational width of the box")
            self.L0 = self.base
            self.Re = self.rho2*self.L0*np.sqrt(self.L0*0.98)/self.mu1

        #Add reference time, velocity and pressure
        self.t0 = np.sqrt(self.L0/self.g)
        self.U0 = np.sqrt(self.g*self.L0)
        self.p0 = self.mu2/self.t0

        #Convert useful constants to constant FENICS functions
        self.DT = Constant(self.dt)

        #Set parameter for standard output
        set_log_level(self.Param["Log_Level"])


    """Build the mesh for the simulation"""
    def build_mesh(self):
        #Generate mesh
        self.mesh = RectangleMesh(Point(0.0, 0.0), Point(self.base, self.height), \
                                  self.Param["Number_vertices_x"], self.Param["Number_vertices_y"])

        #Prepare useful variables for stabilization
        if(self.stab_method == 'IP'):
            self.alpha = self.Param["Stabilization_Parameter"]

        #Parameters for reinitialization steps
        hmin = self.mesh.hmin()
        if(self.reinit_method == 'Non_Conservative'):
            self.eps = self.Param["Interface_Thickness"]
            self.gamma_reinit = Constant(hmin)
            self.beta_reinit = Constant(0.0625*hmin)
            self.dt_reinit = Constant(np.minimum(0.0001, 0.5*hmin)) #We choose an explicit treatment to keep the linearity
                                                                    #and so a very small step is needed
        elif(self.reinit_method == 'Conservative'):
            self.dt_reinit = Constant(0.5*hmin**(1.1))
            self.eps_reinit = Constant(0.5*hmin**(0.9))
            self.eps = self.eps_reinit

        #Define FE spaces
        self.V  = VectorFunctionSpace(self.mesh, "CG", self.deg + 1)
        self.P  = FunctionSpace(self.mesh, "CG" if self.deg > 0 else "DG", self.deg)
        if(self.NS_sol_method == 'Standard'):
            Velem = VectorElement("Lagrange", self.mesh.ufl_cell(), self.deg + 1)
            Pelem = FiniteElement("Lagrange" if self.deg > 0 else "DG", self.mesh.ufl_cell(), self.deg)
            self.W  = FunctionSpace(self.mesh, Velem*Pelem)
        self.Q  = FunctionSpace(self.mesh, "CG", 2)
        self.Q2 = VectorFunctionSpace(self.mesh, "CG", 1)

        #Define trial and test functions
        if(self.NS_sol_method == 'Standard'):
            (self.u, self.p) = TrialFunctions(self.W)
            (self.v, self.q) = TestFunctions(self.W)
        elif(self.NS_sol_method == 'ICT'):
            self.u   = TrialFunction(self.V)
            self.v   = TestFunction(self.V)
            self.p   = TrialFunction(self.P)
            self.q   = TestFunction(self.P)
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

        #Define function to store the normal
        self.n = Function(self.Q2)

        #Define useful functions for reinitialization
        self.phi0 = Function(self.Q)
        self.phi_intermediate = Function(self.Q) #This is fundamental in case on 'non-conservative'
                                                 #reinitialization and it is also useful for clearness

        #Define function and vector for plotting level-set and computing volume
        self.rho_interp = Function(self.Q)


    """Set the proper initial condition"""
    def set_initial_condition(self):
        #Read from configuration file center and radius
        try:
            center = Point(float(self.Param["x_center"]), float(self.Param["y_center"]))
            radius = float(self.Param["Radius"])
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")
            exit(1)

        #Check geoemtric limits
        assert center[0] - radius > 0.0 and center[0] + radius < self.base and \
               center[1] - radius > 0.0 and center[1] + radius < self.height,\
               "Initial condition of interface goes outside the domain"

        #Assign initial condition
        self.u_old.assign(interpolate(Constant((0.0,0.0)), self.V))
        self.p_old.assign(interpolate(Constant(0.0), self.P))
        if(self.reinit_method == 'Non_Conservative'):
            f = Expression("sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B)) - r",
                            A = center[0], B = center[1], r = radius, degree = 2)
            self.phi_old.assign(interpolate(f,self.Q))
        elif(self.reinit_method == 'Conservative'):
            f = Expression("1.0/(1.0 + exp((r - sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B)))/eps))",
                            A = center[0], B = center[1], r = radius, eps = self.eps, degree = 2)
            self.phi_old.assign(interpolate(f, self.Q))


    """Assemble boundary condition"""
    def assembleBC(self):
        if(self.NS_sol_method == 'Standard'):
            self.bcs = [DirichletBC(self.W.sub(0), Constant((0.0,0.0)),  'near(x[1], 0.0) || near(x[1], 2.0)'), \
                        DirichletBC(self.W.sub(0).sub(0), Constant(0.0), 'near(x[0], 0.0) || near(x[0], 1.0)')]
        elif(self.NS_sol_method == 'ICT'):
            self.bcs = [DirichletBC(self.V, Constant((0.0,0.0)),  'near(x[1], 0.0) || near(x[1], 2.0)'), \
                        DirichletBC(self.V.sub(0), Constant(0.0), 'near(x[0], 0.0) || near(x[0], 1.0)')]


    """Auxiliary function to compute density"""
    def rho(self, x, eps):
        if(self.reinit_method == 'Non_Conservative'):
            return self.rho2*CHeaviside(x,eps) + self.rho1*(1.0 - CHeaviside(x,eps))
        elif(self.reinit_method == 'Conservative'):
            return self.rho2*x + self.rho1*(1.0 - x)


    """Auxiliary function to compute viscosity"""
    def mu(self, x, eps):
        if(self.reinit_method == 'Non_Conservative'):
            return self.mu2*CHeaviside(x,eps) + self.mu1*(1.0 - CHeaviside(x,eps))
        elif(self.reinit_method == 'Conservative'):
            return self.mu2*x + self.mu1*(1.0 - x)


    """Set weak formulations"""
    def set_weak_forms(self):
        #Set variational problem for step 1 (Level-set)
        if(self.stab_method == 'SUPG'):
            self.LS_weak_form(self.phi, self.l, self.phi_old, self.u_old, self.DT, self.mesh, self.stab_method, self.Re)
        elif(self.stab_method == 'IP'):
            self.LS_weak_form(self.phi, self.l, self.phi_old, self.u_old, self.DT, self.mesh, self.stab_method, self.alpha)
        elif(self.stab_method == 'None'):
            self.LS_weak_form(self.phi, self.l, self.phi_old, self.u_old, self.DT, self.mesh, self.stab_method)

        #Set variational problem for reinitialization
        if(self.reinit_method == 'Non_Conservative'):
            self.NCLSM_hyperbolic_weak_form(self.phi, self.l, self.phi0, self.phi_curr, self.dt_reinit, self.gamma_reinit, self.beta_reinit)
        elif(self.reinit_method == 'Conservative'):
            self.CLSM_weak_form(self.phi_intermediate, self.l, self.phi0, self.n, self.dt_reinit, self.eps_reinit)

        #Set variational problem for step 2 (Navier-Stokes)
        if(self.NS_sol_method == 'Standard'):
            if(self.reinit_method == 'Non_Conservative'):
                self.NS_weak_form(self.u, self.p, self.v, self.q, self.u_old, self.DT, self.rho, self.mu, \
                                  self.phi_curr, self.phi_old, self.eps, self.g, self.sigma, self.n, CDelta)
            elif(self.reinit_method == 'Conservative'):
                self.NS_weak_form(self.u, self.p, self.v, self.q, self.u_old, self.DT, self.rho, self.mu, \
                                  self.phi_curr, self.phi_old, self.eps, self.g, self.sigma, self.n)
        elif(self.NS_sol_method == 'ICT'):
            if(self.stab_method == 'Non_Conservative'):
                self.ICT_weak_form_1(self.u, self.v, self.u_old, self.DT, self.rho, self.mu, self.phi_curr, self.phi_old, \
                                     self.eps, self.g, self.sigma, self.n, CDelta)
            elif(self.stab_method == 'Conservative'):
                self.ICT_weak_form_1(self.u, self.v, self.u_old, self.DT, self.rho, self.mu, self.phi_curr, self.phi_old, \
                                     self.eps, self.g, self.sigma, self.n)
            self.ICT_weak_form_2(self.p, self.q, self.DT, self.u_curr, self.rho, self.phi_curr, self.eps)
            self.ICT_weak_form_3(self.u, self.v, self.DT, self.u_curr, self.p_curr, self.p_old, self.rho, self.phi_curr, self.eps)


    """Build the system for Level set simulation"""
    def solve_Levelset_system(self):
        # Assemble matrix and right-hand side
        assemble(self.a1, tensor = self.A1)
        assemble(self.L1, tensor = self.b1)

        #Solve the level-set system
        solve(self.A1, self.phi_curr.vector(), self.b1, "gmres", "default")

        #Compute normal vector (in case we avoid reconstrution)
        grad_phi = grad(self.phi_curr)
        self.n.assign(project(grad_phi/sqrt(inner(grad_phi, grad_phi)), self.Q2))


    """Build and solve the system for Level set reinitialization (non-conservative)"""
    def NC_Levelset_reinit(self):
        #Assign current solution and buil the matrix
        self.phi0.assign(self.phi_curr)

        E_old = 1e10
        for n in range(100):
            #Assemble and solve the system
            assemble(self.L1_reinit, tensor = self.b1_reinit)
            solve(self.A1_reinit, self.phi_intermediate.vector(), self.b1_reinit, "cg" , "icc")

            #Compute the error and check no divergence
            error = (((self.phi_intermediate - self.phi0)/self.dt_reinit)**2)*dx
            E = sqrt(assemble(error))

            if(E_old < E):
                raise RuntimeError("Divergence at the reinitialization level (iteration " + str(n + 1) + ")")
            elif(E < 1e-4):
                break

            #Set previous step solution
            self.phi0.assign(self.phi_intermediate)

        #Assign the reinitialized level-set to the current solution and
        #update normal vector to the interface (for Navier-Stokes)
        self.phi_curr.assign(self.phi_intermediate)
        grad_phi = grad(self.phi_curr)
        self.n.assign(project(grad_phi/sqrt(inner(grad_phi, grad_phi)), self.Q2))


    """Build and solve the system for Level set reinitialization (conservative)"""
    def C_Levelset_reinit(self):
        self.phi0.assign(self.phi_curr)

        for n in range(10):
            #Solve the system
            solve(self.F1_reinit == 0, self.phi_intermediate, \
                  solver_parameters={"newton_solver": {'linear_solver': 'gmres', "preconditioner": "default"}}, \
                  form_compiler_parameters={"optimize": True})

            #Check if convergence has been reached
            if(sqrt(assemble((((self.phi_intermediate - self.phi0)/self.dt_reinit)**2)*dx)) < 1e-4):
                break

            #Prepare for next iteration
            self.phi0.assign(self.phi_intermediate)

        #Assign the reinitialized level-set to the current solution and
        #update normal vector to the interface (for Navier-Stokes)
        self.phi_curr.assign(self.phi_intermediate)
        grad_phi = grad(self.phi_curr)
        self.n.assign(project(grad_phi/sqrt(inner(grad_phi, grad_phi)), self.Q2))


    """Build and solve the system for Navier-Stokes simulation"""
    def solve_Standard_NS_system(self):
        # Assemble matrices and right-hand sides
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        # Apply boundary conditions
        for bc in self.bcs:
            bc.apply(self.A2)
            bc.apply(self.b2)

        #Solve the system
        solve(self.A2, self.w_curr.vector(), self.b2, "umfpack")
        (self.u_curr, self.p_curr) = self.w_curr.split(True)


    """Build and solve the system for Navier-Stokes simulation"""
    def solve_ICT_NS_systems(self):
        #Assemble matrices
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        #Apply boundary conditions
        for bc in self.bcs:
            bc.apply(self.A2)
            bc.apply(self.b2)

        #Solve the first system
        solve(self.A2, self.u_curr.vector(), self.b2, "gmres", "default")

        #Assemble and solve the second system
        assemble(self.a2_bis, tensor = self.A2_bis)
        assemble(self.L2_bis, tensor = self.b2_bis)
        solve(self.A2_bis, self.p_curr.vector(), self.b2_bis, "gmres", "default")

        #Assemble and solve the third system
        assemble(self.L2_tris, tensor = self.b2_tris)
        solve(self.A2_tris, self.u_curr.vector(), self.b2_tris, "cg", "default")


    """Plot the level-set function and compute the volume"""
    def plot_and_volume(self):
        #Save the actual state for visualization
        if(self.n_iter % 50 == 0):
            self.vtkfile_u << (self.u_old, self.t)
            self.rho_interp.assign(project(self.rho(self.phi_old,self.eps), self.Q))
            self.vtkfile_rho << (self.rho_interp, self.t)

        #Compute benchamrk quantities
        if(self.reinit_method == 'Non_Conservative'):
            Vol = assemble((1.0 - CHeaviside(self.phi_old,self.eps))*dx)
            Pa = 2.0*sqrt(np.pi*Vol)
            Pb = assemble(CDelta(self.phi_old,self.eps)*dx)
            Chi = Pa/Pb
            Xc = assemble(Expression("x[0]", degree=1)*(1.0 - CHeaviside(self.phi_old,self.eps))*dx)/Vol
            Yc = assemble(Expression("x[1]", degree=1)*(1.0 - CHeaviside(self.phi_old,self.eps))*dx)/Vol
            Uc = assemble(inner(self.u_old,self.e1)*(1.0 - CHeaviside(self.phi_old,self.eps))*dx)/Vol
            Vc = assemble(inner(self.u_old,self.e2)*(1.0 - CHeaviside(self.phi_old,self.eps))*dx)/Vol
            L2_gradphi = sqrt(assemble(inner(grad(self.phi_old),grad(self.phi_old))*dx)/(self.base*self.height))
            timeseries_vec = [self.t,Vol,Chi,Xc,Yc,Uc,Vc,L2_gradphi]
        elif(self.reinit_method == 'Conservative'):
            Vol = assemble((1.0 - self.phi_old)*dx)
            Pa = 2.0*sqrt(np.pi*Vol)
            Pb = assemble(sqrt(inner(grad(self.phi_old),grad(self.phi_old)))*dx)
            Chi = Pa/Pb
            Xc = assemble(Expression("x[0]", degree=1)*(1.0 - self.phi_old)*dx)/Vol
            Yc = assemble(Expression("x[1]", degree=1)*(1.0 - self.phi_old)*dx)/Vol
            Uc = assemble(inner(self.u_old,self.e1)*(1.0 - self.phi_old)*dx)/Vol
            Vc = assemble(inner(self.u_old,self.e2)*(1.0 - self.phi_old)*dx)/Vol
            timeseries_vec = [self.t,Vol,Chi,Xc,Yc,Uc,Vc]

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

        #Time-stepping loop
        self.t = 0.0
        self.n_iter = 0

        #File for plotting
        self.vtkfile_u = File('/u/archive/laureandi/orlando/Sim110/u.pvd')
        self.vtkfile_rho = File('/u/archive/laureandi/orlando/Sim110/rho.pvd')

        #File for benchamrk comparisons
        self.timeseries = open('/u/archive/laureandi/orlando/Sim110/benchmark_series.dat','ab')

        #Save initial state and start loop
        self.plot_and_volume()
        self.timeseries.close()
        self.t += self.dt
        while self.t <= self.t_end:
            begin(int(LogLevel.INFO) + 1,"t = " + str(self.t) + " s")
            self.n_iter += 1

            #Solve level-set
            begin(int(LogLevel.INFO) + 1,"Solving Level-set")
            self.solve_Levelset_system()
            end()

            #Solve Level-set reinit
            try:
                begin(int(LogLevel.INFO) + 1,"Solving reinitialization")
                self.switcher_reinit_solver[self.reinit_method]()
                end()
            except RuntimeError as e:
                print(e)
                print("Aborting simulation...")
                exit(1)

            #Solve Navier-Stokes
            begin(int(LogLevel.INFO) + 1,"Solving Navier-Stokes")
            self.switcher_NS[self.NS_sol_method]()
            end()

            #Prepare to next step assign previous-step solution
            self.u_old.assign(self.u_curr)
            self.p_old.assign(self.p_curr)
            self.phi_old.assign(self.phi_curr)

            #Save and compute benchmark quantities
            begin(int(LogLevel.INFO) + 1,"Computing benchmark quantities")
            self.timeseries = open('/u/archive/laureandi/orlando/Sim110/benchmark_series.dat','ab')
            self.plot_and_volume()
            self.timeseries.close()
            end()

            end()

            self.t += self.dt if self.t + self.dt <= self.t_end or abs(self.t - self.t_end) < DOLFIN_EPS else self.t_end
