from My_Parameters import My_Parameters
from Auxiliary_Functions import *

from sys import exit

class BubbleMove:
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
        self.eps = self.Param["Interface_Thickness"]
        self.alpha = self.Param["Stabilization_Parameter"]

        #Define an auxiliary dictionary to set proper stabilization
        try:
            self.switcher_stab = {'IP':self.IP, 'SUPG':self.SUPG, 'None':self.no_stab}
        except NameError as e:
            print("Stabilization method " + str(e).split("'")[1] + " declared but not implemented")
            exit(1)
        assert self.stab_method in self.switcher_stab, \
               "Stabilization method not available"

        #Check correctness of reinitialization method (in order to avoid typos in particular
        #since, contrarly to stabilization, it is more difficult to think to other choises)
        #through auxiliary functions which can be eaily updated in future if needed
        try:
            self.switcher_reinit_weak_form = {'Non_Conservative':self.NCLSM_weak_form, 'Conservative':self.CLSM_weak_form}
            self.switcher_reinit_solver    = {'Non_Conservative':self.NC_Levelset_reinit, \
                                              'Conservative':self.C_Levelset_reinit}
        except NameError as e:
            print("Reinitialization method " + str(e).split("'")[1] + " declared but not implemented")
            exit(1)
        assert self.reinit_method in self.switcher_reinit_weak_form and self.switcher_reinit_solver, \
               "Reinitialization method not available"

        #Compute the Atwood number according to the configuration file
        self.At = (self.rho2 - self.rho1)/(self.rho2 + self.rho1)

        #Compute density and viscosity ratio
        self.rho1_rho2 = self.rho1/self.rho2
        self.mu1_mu2 = self.mu1/self.mu2

        #Compute the reference length (we need it in order to adimensionalize the level-set):
        #since density and viscosity are physical properties it is reasonable to compute it from the
        #Reynolds number
        try:
            self.Re = float(self.Param["Reynolds_number"])
            assert self.Re > 1.0, "Invalid Reynolds number specified"
            self.L0 = (self.mu2*self.Re/(self.rho2*np.sqrt(self.At*0.98)))**(2/3)
        except RuntimeError as e:
            #In case Reynolds number is not present set the computational width of the box
            print("Setting reference length equal to the computational width of the box")
            self.L0 = self.base
            self.Re = self.rho2*self.L0*np.sqrt(self.At*self.L0*0.98)/self.mu1

        #Add reference time, velocity and pressure
        self.t0 = np.sqrt(self.L0/(self.At*0.98))
        self.U0 = np.sqrt(self.At*0.98*self.L0)
        self.p0 = self.mu2/self.t0

        #Convert useful constants to constant FENICS functions
        self.DT = Constant(self.dt)
        self.e2 = Constant((0.0,1.0))

        #Set parameter for standard output
        set_log_level(self.Param["Log_Level"])


    """Build the mesh for the simulation"""
    def build_mesh(self):
        #Generate mesh
        self.mesh = RectangleMesh(Point(0.0, 0.0), Point(self.base, self.height), \
                                  self.Param["Number_vertices_x"], self.Param["Number_vertices_y"])

        #Prepare useful variables for stabilization
        self.h = CellDiameter(self.mesh)
        if(self.stab_method == 'IP'):
            self.n_mesh = FacetNormal(self.mesh)
            self.h_avg  = (self.h('+') + self.h('-'))/2.0

        #Parameters for reinitialization steps
        hmin = self.mesh.hmin()
        if(self.reinit_method == 'Non_Conservative'):
            self.eps_reinit = Constant(hmin)
            self.alpha_reinit = Constant(0.0625*hmin)
            self.dt_reinit = Constant(np.minimum(0.0001, 0.5*hmin)) #We choose an explicit treatment to keep the linearity
                                                                    #and so a very small step is needed
        elif(self.reinit_method == 'Conservative'):
            self.dt_reinit = Constant(0.5*hmin**(1.1))
            self.eps_reinit = Constant(0.5*hmin**(0.9))

        #Define FE spaces
        self.V  = VectorFunctionSpace(self.mesh, "CG", self.deg + 1)
        self.P  = FunctionSpace(self.mesh, "CG" if self.deg > 0 else "DG", self.deg)
        self.Q  = FunctionSpace(self.mesh, "CG", 2)
        self.Q2 = VectorFunctionSpace(self.mesh, "CG", 1)

        #Define trial and test functions
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
        self.phi_curr = Function(self.Q)
        self.phi_old  = Function(self.Q)

        #Define function to store the normal
        self.n = Function(self.Q2)

        #Define useful functions for reinitialization
        self.phi0 = Function(self.Q)
        self.phi_intermediate = Function(self.Q) #This is fundamental in case on 'non-conservative'
                                                 #reinitialization and it is also useful for clearness

        #Define function and vector for plotting level-set and computing volume
        self.tmp = Function(self.Q)
        self.lev_set = np.empty_like(self.phi_old.vector().get_local())
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

        #Set initial condition of bubble and check geoemtric limits
        f = Expression("sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B)) - r",
                        A = center[0], B = center[1], r = radius, degree = 2)
        assert center[0] - radius > 0.0 and center[0] + radius < self.base and \
               center[1] - radius > 0.0 and center[1] + radius < self.height,\
               "Initial condition of interface goes outside the domain"

        #Assign initial condition
        self.u_old.assign(interpolate(Constant((0.0,0.0)),self.V))
        self.p_old.assign(interpolate(Constant(0.0),self.P))
        self.phi_old.assign(interpolate(f,self.Q))

        #Compute normal vector to the interface
        #self.grad_phi = project(grad(self.phi_old), self.Q2)
        #self.n = self.grad_phi/sqrt(inner(self.grad_phi, self.grad_phi))


    """Assemble boundary condition"""
    def assembleBC(self):
        self.bcs = [DirichletBC(self.V, Constant((0.0,0.0)),  'near(x[1], 0.0) || near(x[1], 2.0)'),
                    DirichletBC(self.V.sub(0), Constant(0.0), 'near(x[0], 0.0) || near(x[0], 1.0)')]


    """Auxiliary function to compute density"""
    def rho(self, x, eps):
        return CHeaviside(x,eps) + self.rho1_rho2*(1.0 - CHeaviside(x,eps))


    """Auxiliary function to compute viscosity"""
    def mu(self, x, eps):
        return CHeaviside(x,eps) + self.mu1_mu2*(1.0 - CHeaviside(x,eps))


    """No stabilization"""
    def no_stab(self, phi, l):
        return 0.0


    """Interior penalty method"""
    def IP(self, phi, l):
        r = self.alpha*self.h_avg*self.h_avg* \
            inner(jump(grad(phi),self.n_mesh), jump(grad(l),self.n_mesh))*dS
        return r


    """SUPG method"""
    def SUPG(self, phi, l):
        r = ((phi - self.phi_old)/self.DT + inner(self.u_old, grad(phi)))* \
            self.alpha*self.h/ufl.Max(2.0*sqrt(inner(self.u_old,self.u_old)), 4.0/(self.Re*self.h))*\
            inner(self.u_old,self.u_old)*inner(self.u_old, grad(l))*dx
        return r


    def IP(self, phi, l):
        r = self.alpha*self.h_avg*self.h_avg* \
            inner(jump(grad(phi),self.n_mesh), jump(grad(l),self.n_mesh))*dS
        return r


    """SUPG method"""
    def SUPG(self, phi, l):
        r = ((phi - self.phi_old)/self.DT + inner(self.u_old, grad(phi)))* \
            self.alpha*self.h/ufl.Max(2.0*sqrt(inner(self.u_old,self.u_old)), 4.0/(self.Re*self.h))*\
            inner(self.u_old,self.u_old)*inner(self.u_old, grad(l))*dx
        return r



    """Level-set weak formulation"""
    def LS_weak_form(self):
        F1 = (self.phi - self.phi_old)/self.DT*self.l*dx \
           + inner(self.u_old, grad(self.phi))*self.l*dx

        F1 += self.switcher_stab[self.stab_method](self.phi, self.l)

        #Save corresponding weak form and declare suitable matrix and vector
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        self.A1 = Matrix()
        self.b1 = Vector()


    """Weak form non-conservative reinitialization"""
    def NCLSM_weak_form(self):
        self.a1_reinit = self.phi/self.dt_reinit*self.l*dx
        self.L1_reinit = self.phi0/self.dt_reinit*self.l*dx + \
                         signp(self.phi_curr, self.eps_reinit)*(1.0 - mgrad(self.phi0))*self.l*dx -\
                         self.alpha_reinit*inner(grad(self.phi0), grad(self.l))*dx

        #Save the matrix that will not change and declare vector
        self.A1_reinit = assemble(self.a1_reinit)
        self.b1_reinit = Vector()


    """Weak form conservative reinitialization"""
    def CLSM_weak_form(self):
        #Save variational formulation
        self.F1_reinit = (self.phi_intermediate - self.phi0)/self.dt_reinit*self.l*dx \
                       - self.phi_intermediate*(1.0 - self.phi_intermediate)*inner(grad(self.l), self.n)*dx \
                       + self.eps_reinit*inner(grad(self.phi_intermediate), grad(self.l))*dx


    """Weak formulation for tentative velocity"""
    def ICT_weak_form_1(self):
        F2 = (1.0/self.DT)*inner(self.rho(self.phi_curr,self.eps)*self.u - self.rho(self.phi_old,self.eps)*self.u_old, self.v)*dx  \
           + inner(self.rho(self.phi_curr,self.eps)*dot(self.u_old,nabla_grad(self.u)), self.v)*dx \
           + 2.0/self.Re*inner(self.mu(self.phi_curr,self.eps)*D(self.u), D(self.v))*dx \
           - self.p_old*div(self.v)*dx \
           + inner(self.rho(self.phi_curr,self.eps)*self.e2, self.v)*dx \
           + 1.0/(self.Bo*self.At)*sqrt(inner(grad(self.phi_curr), grad(self.phi_curr)))*\
             inner(Identity(2) - outer(self.n, self.n), grad(self.v))*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()

    """Weak formulation for pressure correction"""
    def ICT_weak_form_2(self):
        #Define variational problem for step 2
        self.a2_bis = (1.0/self.rho(self.phi_curr,self.eps))*inner(grad(self.p), grad(self.q))*dx
        self.L2_bis = (1.0/self.rho(self.phi_curr,self.eps))*inner(grad(self.p_old), grad(self.q))*dx - \
                      (1.0/self.DT)*div(self.u_curr)*self.q*dx

        self.A2_bis = Matrix()
        self.b2_bis = Vector()

    """Weak formulation for velocity projection"""
    def ICT_weak_form_3(self):
        #Define variational problem for step 3
        self.a2_tris = inner(self.u, self.v)*dx
        self.L2_tris = inner(self.u_curr, self.v)*dx - \
                       self.DT*inner(grad(self.p_curr - self.p_old), self.v)/self.rho(self.phi_curr,self.eps)*dx

        self.A2_tris = assemble(self.a2_tris)
        self.b2_tris = Vector()


    """Set weak formulations"""
    def set_weak_forms(self):
        #Set variational problem for step 1 (Level-set)
        self.LS_weak_form()

        #Set variational problem for reinitialization after step 1
        self.switcher_reinit_weak_form[self.reinit_method]()

        #Set variational problem for step 2 (Navier-Stokes)
        self.ICT_weak_form_1()
        self.ICT_weak_form_2()
        self.ICT_weak_form_3()


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
        #Assign current solution
        self.phi0.assign(self.phi_curr)

        E_old = 1e10
        for n in range(4):
            #Assemble and solve the system
            assemble(self.L1_reinit, tensor = self.b1_reinit)
            solve(self.A1_reinit, self.phi_intermediate.vector(), self.b1_reinit, "cg" , "default")

            #Compute the error and check no divergence
            error = (((self.phi_intermediate - self.phi0)/self.dt_reinit)**2)*dx
            E = sqrt(abs(assemble(error)))

            if(E_old < E):
                raise RuntimeError("Divergence at the reinitialization level (iteration " + str(n + 1) + ")")
            elif(E < 1e-3):
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
        #Assign current solution
        self.phi0.assign(self.phi_curr)

        for n in range(10):
            #Solve the system
            solve(self.F1_reinit == 0, self.phi_intermediate, \
                  solver_parameters={"newton_solver": {'linear_solver': 'bicgstab', "preconditioner": "default"}}, \
                  form_compiler_parameters={"optimize": True})

            #Check if convergence has been reached
            if(norm(assemble(self.F1_reinit), "L2") < 1e-3):
                break

            #Set previous step solution
            self.phi0.assign(self.phi_intermediate)

        #Assign the reinitialized level-set to the current solution and
        #update normal vector to the interface (for Navier-Stokes)
        self.phi_curr.assign(self.phi_intermediate)
        grad_phi = grad(self.phi_curr)
        self.n.assign(project(grad_phi/sqrt(inner(grad_phi, grad_phi)), self.Q2))


    """Build and solve the system for Navier-Stokes simulation"""
    def solve_Standard_NS_system(self):
        #Assemble matrices
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        #Apply boundary conditions
        [bc.apply(self.A2) for bc in self.bcs]
        [bc.apply(self.b2) for bc in self.bcs]

        #Solve the first system
        solve(self.A2, self.u_curr.vector(), self.b2, "bicgstab", "default")

        #Assemble and solve the second system
        assemble(self.a2_bis, tensor = self.A2_bis)
        assemble(self.L2_bis, tensor = self.b2_bis)
        solve(self.A2_bis, self.p_curr.vector(), self.b2_bis, "bicgstab", "default")

        #Assemble and solve the third system
        assemble(self.L2_tris, tensor = self.b2_tris)
        solve(self.A2_tris, self.u_curr.vector(), self.b2_tris, "cg", "default")


    """Plot the level-set function and compute the volume"""
    def plot_and_volume(self):
        #Extract vector for FE function
        phi_old_vec = self.phi_old.vector().get_local()

        #Construct vector of ones inside the bubble
        self.lev_set = 1.0*(phi_old_vec < 0.0)

        #Assign vector to FE function
        self.tmp.vector().set_local(self.lev_set)

        #Plot the function just computed
        if(self.n_iter % 50 == 0):
            self.vtkfile_phi_draw << (self.tmp, self.t*self.t0)
            self.vtkfile_u << (self.u_old, self.t*self.t0)
            self.rho_interp.assign(project(self.rho(self.phi_old,self.eps), self.Q))
            self.vtkfile_rho << (self.rho_interp, self.t*self.t0)

        #Check volume consistency
        Vol = assemble(self.tmp*dx)
        begin(int(LogLevel.INFO) + 1,"Volume = " + str(Vol))
        end()


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
        self.t = self.dt
        self.n_iter = 0
        self.vtkfile_phi_draw = File('/u/archive/laureandi/orlando/Sim73/phi_draw.pvd')
        self.vtkfile_u = File('/u/archive/laureandi/orlando/Sim73/u.pvd')
        self.vtkfile_rho = File('/u/archive/laureandi/orlando/Sim73/rho.pvd')
        self.plot_and_volume()
        while self.t <= self.t_end:
            begin(int(LogLevel.INFO) + 1,"t = " + str(self.t*self.t0) + " s")

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
            self.solve_Standard_NS_system()
            end()

            #Prepare to next step assign previous-step solution
            self.u_old.assign(self.u_curr)
            self.p_old.assign(self.p_curr)
            self.phi_old.assign(self.phi_curr)

            #Save and compute volume
            begin(int(LogLevel.INFO) + 1,"Plotting and computing volume")
            self.n_iter += 1
            self.plot_and_volume()
            end()

            end()

            self.t += self.dt if self.t + self.dt <= self.t_end or abs(self.t - self.t_end) < DOLFIN_EPS else self.t_end
