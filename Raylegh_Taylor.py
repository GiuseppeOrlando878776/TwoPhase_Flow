from My_Parameters import My_Parameters
from Auxiliary_Functions import *
from Periodic_BC import WallBoundary

from sys import exit

class RayleghTaylor:
    """Class constructor"""
    def __init__(self, param_name):
        """
        Param --- class Parameters to store desired configuration
        Re    --- Reynolds number
        At    --- Atwood number
        dt    --- Specified time step
        t_end  --- End time of the simulation
        deg   --- Polynomial degree
        """

        self.Param = My_Parameters(param_name).get_param()

        try:
            self.Re            = self.Param["Reynolds_number"]
            self.At            = self.Param["Atwood_number"]
            self.surf_coeff    = self.Param["Surface_tension"]
            self.rho1          = self.Param["Lighter_density"]
            self.dt            = self.Param["Time_step"]
            self.t_end         = self.Param["End_time"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

        #Since this parameters are more related to the numeric part
        #rather than physics we set a default value
        #and so they are present for sure
        self.deg = int(self.Param["Polynomial_degree"])
        self.reinit_method = self.reinit_type(int(self.Param["Reinit_Type"]))
        self.stab_method   = self.stabilization_type(int(self.Param["Stabilization_Type"]))

        #Compute heavier density
        self.rho2 = self.rho1*(1.0 + self.At)/(1.0 - self.At)

        #Compute viscosity: the 'lighter' viscosity will be computed by
        #Reynolds number, while for the 'heavier' we choose to impose a constant
        #density-viscosity ratio (arbitrary choice)
        self.mu1 = self.rho1*np.sqrt(9.81*self.At)/self.Re
        self.mu2 = self.rho2*self.mu1/self.rho1

        #Convert useful constants to constant FENICS function
        self.DT = Constant(self.dt)
        self.g  = Constant(9.81)
        self.e2 = Constant((0.0,1.0))

        #Set parameters for standard output
        set_log_level(21)


    """Build the mesh for the simulation"""
    def build_mesh(self):
        #Generate mesh
        try:
            n_points = int(self.Param["Number_vertices"])
            self.mesh = RectangleMesh(Point(0.0, 0.0), Point(self.Param["Base"], self.Param["Height"]), \
                                      n_points, n_points)
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

        #Prepare useful variables for Interior Penalty
        self.n_mesh = FacetNormal(self.mesh)
        self.h      = CellDiameter(self.mesh)
        self.h_avg  = (self.h('+') + self.h('-'))/2.0

        #Parameter for interface thickness
        self.eps = 1.0e-4
        self.alpha = Constant(0.1) #Penalty parameter

        #Parameters for reinitialization steps
        hmin = self.mesh.hmin()
        if(self.reinit_type == 'Non_Conservative'):
            self.eps_reinit = Constant(hmin)
            self.alpha_reinit = Constant(0.0625*hmin)
            self.dt_reinit = Constant(0.0001) #We choose an explicit treatment to maintain the linearity
                                              #and so a very small step is neede
        elif(self.reinit_type == 'Conservative'):
            self.dt_reinit = Constant(0.5*hmin**(1.1))
            self.eps_reinit = Constant(0.5*hmin**(0.9))

        #Define function spaces
        Velem   = VectorElement("Lagrange", self.mesh.ufl_cell(), self.deg + 1)
        Qelem   = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.deg)
        Phielem = FiniteElement("Lagrange", self.mesh,ufl_cell(), 2)
        self.W  = FunctionSpace(self.mesh, Velem*Qelem)
        self.Q  = FunctionSpace(self.mesh, Phielem)
        self.Q2 = FunctionSpace(self.mesh, Phielem*Phielem)

        #Define trial and test functions
        (self.u, self.p) = TrialFunctions(self.W)
        self.phi         = TrialFunction(self.Q)
        (self.v, self.q) = TestFunctions(self.W)
        self.l           = TestFunction(self.Q)

        #Define functions for solutions at previous and current time steps
        self.w_old    = Function(self.W)
        self.phi_old  = Function(self.Q)
        self.w_curr   = Function(self.W)
        (self.u_curr, self.p_curr) = self.w_curr.split()
        self.phi_curr = Function(self.Q)

        #Define function for reinitialization
        self.phi0 = Function(self.Q)
        self.phi_intermediate = Function(self.Q) #This is fundamental in case on 'non-conservative'
                                                 #reinitialization and it is also useful for clearness

        #Define function for normal to the interface
        self.grad_phi = Function(self.Q2)
        self.n = Function(self.Q2)


    """Set the proper initial condition"""
    def set_initial_condition(self):
        #Read from configuration file center and radius
        try:
            center = Point(self.Param["x_center"], self.Param["y_center"])
            radius = self.Param["Radius"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

        #Set initial condition of bubble and check geoemtric limits
        f = Expression("sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B))-r",
                        A = center[0], B = center[1], r = radius, degree = 2)
        assert center[0] - radius > 0.0 and center[0] + radius < self.Param["Base"] and\
               center[1] - radius > 0.0 and center[1] + radius < self.Param["Height"],\
                "Initial condition of interface goes outside the domain"

        #Assign initial condition
        self.phi_old.assign(interpolate(f,self.Q))
        self.w_old.assign(interpolate(Constant((0.0,0.0,0.0)),self.W))
        (self.u_old, self.p_old) = self.w_old.split()

        #Compute normal vector to the surface
        self.grad_phi = project(grad(self.phi_old), self.Q2)
        self.n = self.grad_phi/sqrt(inner(self.grad_phi, self.grad_phi))


    """Interior penalty method"""
    def IP(self, phi, l):
        r = self.alpha*self.h_avg*self.h_avg* \
            inner(jump(grad(phi),self.n_mesh), jump(grad(l),self.n_mesh))*dS
        return r


    """SUPG method"""
    def SUPG(self, phi, l):
        r = ((phi - self.phi_old)/self.DT + inner(self.u_curr, grad(phi)))* \
            1.0/ufl.Max(2.0*sqrt(inner(self.u_curr,self.u_curr)),4.0/self.Re/self.h/self.h)*\
            inner(self.u_curr,self.u_curr)*inner(self.u_curr, grad(l))*dx
        return r


    """Set weak formulations"""
    def set_weak_forms(self):
        #Define variational problem for step 1 (Navier-Stokes)
        F1 = self.rho(self.phi_old,self.eps)*(inner((self.u - self.u_old) / self.DT, self.v) + \
                                          inner(dot(self.u_old, nabla_grad(self.u)), self.v))*dx \
           + 2.0/self.Re*self.mu(self.phi_old,self.eps)*inner(D(self.u), grad(self.v))*dx\
           - 1.0/self.Re*self.p*div(self.v)*dx\
           + div(self.u)*self.q*dx\
          # + 1.0/self.At*inner(self.rho_old*self.g*self.e2, self.v)*dx\
          # - 1.0/self.Re*inner(self.surf_coeff*div(self.n)*self.n*CDelta(self.phi_old, 1e-4), self.v)*dx(domain=self.mesh)

        #Save corresponding weak form and declare suitable matrix and vector
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        self.A1 = Matrix()
        self.b1 = Vector()

        #Define variational problem for step 2 (Level-set)
        F2 = (self.phi - self.phi_old) / self.DT * self.l*dx \
           + inner(self.u_curr, grad(self.phi))*self.l*dx

        if(self.stab_method == 'IP'):
           F2 += self.IP(self.phi, self.l)
       elif(self.stab_method == 'SUPG'):
           F2 += self.SUPG(self.phi, self.l)

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()

        #Set weak form for level-set reinitialization
        if(self.reinit_type == 'Non_Conservative'):
            self.a3 = self.phi/self.dt_reinit*self.l*dx
            self.L3 = self.phi0/self.dt_reinit*self.l*dx + \
                      signp(self.phi_curr, self.eps_reinit)*\
                      (1.0 - sqrt(inner(grad(self.phi0), grad(self.phi0))))*self.l*dx -\
                      self.alpha_reinit*inner(grad(self.phi0), grad(self.l))* dx
        elif(self.reinit_type == 'Conservative'):
            F3 = (self.phi - self.phi0)/self.dt_reinit*self.l*dx\
               - 0.5*(self.phi + self.phi0)*(1.0 - 0.5*(self.phi + self.phi0))* \
                 inner(self.n, grad(self.l))*dx \
               + self.eps_reinit*inner(self.n, grad((0.5*(self.phi + self.phi0))))* \
                 inner(self.n, grad(self.l))*dx
            self.a3 = lhs(F3)
            self.L3 = rhs(F3)


    """Assemble boundary condition"""
    def assembleBC(self):
        self.bcs = DirichletBC(self.W.sub(0), Constant((0.0,0.0)), WallBoundary())


    """Auxiliary function to compute density"""
    def rho(self, x, eps):
        return self.rho1*(1.0 - CHeaviside(x,eps)) + self.rho2*CHeaviside(x,eps)


    """Auxiliary function to compute viscosity"""
    def mu(self, x, eps):
        return self.mu1*(1.0 - CHeaviside(x,eps)) + self.mu2*CHeaviside(x,eps)


    """Build the system for Navier-Stokes simulation"""
    def assemble_NS_system(self):
        # Assemble matrices and right-hand sides
        assemble(self.a1, tensor = self.A1)
        assemble(self.L1, tensor = self.b1)

        # Apply boundary conditions
        self.bcs.apply(self.A1)
        self.bcs.apply(self.b1)


    """Build the system for Level set simulation"""
    def assemble_Levelset_system(self):
        # Assemble matrix and right-hand side
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)


    """Build the system for Level set reinitialization"""
    def Levelset_reinit(self):
        #Assign current solution and current normal vector to the interface
        self.phi0.assign(self.phi_curr)
        self.grad_phi = project(grad(self.phi_curr), self.Q2)
        self.n = self.grad_phi/sqrt(inner(self.grad_phi, self.grad_phi))

        E_old = 1e10
        for n in range(10):
            #Solve the system
            solve(self.a3 == self.L3, self.phi_intermediate, [])

            #Compute the error and check no divergence
            error = (((self.phi_intermediate - self.phi0)/self.dt_reinit)**2)*dx
            E = sqrt(abs(assemble(error)))

            if(E_old < E):
                raise RuntimeError("Divergence at the reinitialization level (iteration " + str(n + 1) + ")")

            E_old = E

            #Set previous step solution
            self.phi0.assign(self.phi_intermediate)

        #Assign the reinitialized level-set to the current solution and
        #update normal vector to the interface (for Navier-Stokes)
        self.phi_curr.assign(self.phi_intermediate)
        self.grad_phi = project(grad(self.phi_curr), self.Q2)
        self.n = self.grad_phi/sqrt(inner(self.grad_phi, self.grad_phi))


    """Execute simulation"""
    def run(self):
        #Build the mesh
        self.build_mesh()

        #Set the initial condition
        self.set_initial_condition()

        #Set weak formulations
        self.set_weak_forms()

        #Assemble boundary conditions
        self.assembleBC()

        #Time-stepping loop
        t = self.dt
        while t <= self.t_end:
            print("t = ",str(t))

            #Solve Navier-Stokes
            print("Solving Navier-Stokes")
            self.assemble_NS_system()
            solve(self.A1, self.w_curr.vector(), self.b1)
            (self.u_curr, self.p_curr) = self.w_curr.split()

            #Solve level-set
            print("Solving Level-set")
            self.assemble_Levelset_system()
            solve(self.A2, self.phi_curr.vector(), self.b2)
            #print(self.phi_curr.vector().get_local())

            #Apply reinitialization for level-set
            try:
                print("Solving reinitialization")
                self.Levelset_reinit()
            except RuntimeError as e:
                print(e)
                print("Aborting simulation...")
                exit(1)

            #Plot level-set solution
            plot(self.phi_curr, interactive = True)

            #Prepare to next step assign previous-step solution
            self.w_old.assign(self.w_curr)
            (self.u_old, self.p_old) = self.w_old.split()
            self.phi_old.assign(self.phi_curr)

            t = t + self.dt if t + self.dt <= self.t_end or abs(t - self.t_end) < DOLFIN_EPS else self.t_end
