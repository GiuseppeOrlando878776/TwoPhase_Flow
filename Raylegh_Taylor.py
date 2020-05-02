from My_Parameters import My_Parameters
from Auxiliary_Functions import *
from Periodic_BC import WallBoundary

from fenics import *

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
            self.Re         = self.Param["Reynolds_number"]
            self.At         = self.Param["Atwood_number"]
            self.surf_coeff = self.Param["Surface_tension"]
            self.rho1       = self.Param["Lighter_density"]
            self.dt         = self.Param["Time_step"]
            self.t_end      = self.Param["End_time"]
            self.deg        = self.Param["Polynomial_degree"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

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


    """Build the mesh for the simulation"""
    def build_mesh(self):
        #Generate mesh
        try:
            n_points = self.Param["Number_vertices"]
            self.mesh = RectangleMesh(Point(0.0, 0.0), Point(self.Param["Base"], self.Param["Height"]), n_points, n_points)
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

        #Prepare useful variables for Interior Penalty
        self.n = FacetNormal(self.mesh)
        self.h = CellDiameter(self.mesh)
        self.h_avg = (self.h('+') + self.h('-'))/2.0
        self.alpha = Constant(0.1)

        #Define function spaces
        Velem = VectorElement("Lagrange", self.mesh.ufl_cell(), self.deg + 1)
        Qelem = FiniteElement("Lagrange", self.mesh.ufl_cell(), self.deg)
        self.W = FunctionSpace(self.mesh, Velem*Qelem)
        self.Q = FunctionSpace(self.mesh, Qelem)

        #Define trial and test functions
        (self.u, self.p) = TrialFunctions(self.W)
        self.phi         = TrialFunction(self.Q)
        (self.v, self.q) = TestFunctions(self.W)
        self.l           = TestFunction(self.Q)

        #Define functions for solutions at previous and current time steps
        self.w_old    = Function(self.W)
        self.phi_old  = Function(self.Q)
        self.w_curr   = Function(self.W)
        self.phi_curr = Function(self.Q)


    """Set the proper initial condition"""
    def set_initial_condition(self):
        #Read from configuration file center and radius
        try:
            center = Point(self.param["x_center"], self.param["y_center"])
            radius = self.param["Radius"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

        #Set initial condition of bubble and check geoemtric limits
        f = Expression("sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B))-r",
                        A = center[0], B = center[1], r = radius)
        assert (center[0] - radius > 0.0 and center[0] + radius < self.Param["Base"] and
                center[1] - radius > 0.0 and center[1] + radius < self.Param["Height"]),
                "Initial condition of interface goes outside the domain"

        #Assign initial condition
        self.phi_old.assign(interpolate(f,self.Q))
        self.w_old.assign(interpolate(Constant((0.0,0.0,0.0)),self.W))
        (self.u_old, self.p_old) = self.w_old.split()

        self.rho_old = Function(self.Q)
        self.mu_old  = Function(self.Q)


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
        #Compute actual density and viscosity.
        self.rho_old.vector().set_local(self.rho(self.phi_old.vector().get_local(), 1e-4))
        self.mu_old.vector().set_local(self.mu(self.phi_old.vector().get_local(), 1e-4))

        # Define variational problem for step 1
        F1 = inner(self.rho_old*(self.u - self.u_old) / self.DT, self.v)*dx \
           + inner(self.rho_old*dot(self.u_old, nabla_grad(self.u)), self.v)*dx \
           + 1.0/self.Re*inner(sigma(self.mu_old, self.u, self.p_old), nabla_grad(self.v))*dx \
           + 1.0/self.At*dot(self.rho_old*self.g*self.e2, self.v)*dx
           #- 1.0/self.Re*inner(self.surf_coeff*div(self.n)*self.n*CDelta(self.phi_old, 1e-4), self.v)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Assemble matrices and right-hand sides
        self.A1 = assemble(a1)
        self.b1 = assemble(L1)

        # Apply boundary conditions
        self.bcs.apply(self.A1)
        self.bcs.apply(self.b1)


    """Interior penalty method"""
    def IP(self, phi, l):
        r = self.alpha*self.h_avg*self.h_avg*inner(jump(grad(phi),self.n), jump(grad(l),self.n))*dS
        return r


    """Build the system for Level set simulation"""
    def assemble_Levelset_system(self):
        F2 = (self.phi - self.phi_old) / self.DT * self.l*dx \
           - inner(self.phi*self.u_curr, grad(self.l))*dx \
           + self.IP(self.phi, self.l)
        a2 = lhs(F2)
        L2 = rhs(F2)

        # Assemble matrix and right-hand side
        self.A2 = assemble(a2)
        self.b2 = assemble(L2)


    """Execute simulation"""
    def run(self):
        #Build the mesh
        self.build_mesh()

        #Set the initial condition
        self.set_initial_condition()

        #Assemble boundary conditions
        self.assembleBC()

        #Time-stepping
        t = self.dt
        while t <= self.t_end:
            print("t = ",str(t))

            #Solve Navier-Stokes
            self.assemble_NS_system()
            solve(self.A1, self.w_curr.vector(), self.b1)
            (self.u_curr, self.p_curr) = self.w_curr.split()

            #Solve level-set
            self.assemble_Levelset_system()
            solve(self.A2, self.phi_curr.vector(), self.b2)

            #Plot level-set solution
            plot(self.phi_curr, interactive = True)

            #Prepare to next step
            self.w_old.assign(self.w_curr)
            (self.u_old, self.p_old) = self.w_old.split()
            self.phi_old.assign(self.phi_curr)

            t = t + self.dt if t + self.dt <= self.t_end else self.t_end
