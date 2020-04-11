from My_Parameters import My_Parameters
from Auxiliary_Functions import *
from Periodic_BC import *

from fenics import *

class RayleghTaylor:
    """Class constructor"""
    def __init__(self, param_name):
        """
        Param --- class Parameters to store desired configuration
        Re    --- Reynolds number
        At    --- Atwood number
        dt    --- Specified time step
        tend  --- End time of the simulation
        deg   --- Polynomial degree
        """

        self.Param = My_Parameters(param_name).get_param()

        try:
            self.Re         = self.Param["Reynolds_number"]
            self.At         = self.Param["Atwood_number"]
            self.surf_coeff = self.Param["Surface_tension"]
            self.rho1       = self.Param["Lighter_density"]
            self.dt         = self.Param["Time_step"]
            self.tend       = self.Param["End_time"]
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
        channel = Rectangle(Point(0, 0), Point(0.41, 2.22))
        #Generate mesh
        n_points = self.Param["Number_vertices"]
        self.mesh = generate_mesh(domain, n_points)

        #Prepare useful variables for Interior Penalty
        self.n = FacetNormal(self.mesh)
        self.h = CellSize(self.mesh)
        self.h_avg = (h('+') + h('-'))/2.0
        self.alpha = Constant(0.1)

        #Define function spaces
        Velem = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        Qelem = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        self.W = FunctionSpace(self.mesh, Velem*Qelem, constrained_domain = PeriodicBoundary())
        self.Q = FunctionSpace(self.mesh, Qelem,       constrained_domain = PeriodicBoundary())

        #Define trial and test functions
        (self.u, self.p) = TrialFunctions(W)
        self.phi         = TrialFunction(W)
        (self.v, self.q) = TestFunctions(W)
        self.l           = TestFunction(W)

        #Define functions for solutions at previous and current time steps
        self.w_old    = Function(W)
        self.phi_old  = Function(Q)
        self.w_curr   = Function(W)
        self.phi_curr = Function(Q)


    """Set the proper initial condition"""
    def set_initial_condition(self):
        self.phi_old.interpolate("tanh(x[1] - 2 - 0.1*cos(2*pi*x[0]))/(0.01*sqrt(2))")
        self.w_old.assign(interpolate(Constant((0.0,0.0,1.0)),self.W))
        (self.u_old, self.p_old) = self.w_old.split()


    """Auxiliary function to compute density"""
    def rho(self, x, eps):
        return self.rho1*(1.0 - CHeaviside(x,eps)) + self.rho2*CHeaviside(x,eps)


    """Auxiliary function to compute viscosity"""
    def mu(self, x, eps):
        return self.mu1*(1.0 - CHeaviside(x,eps)) + self.mu2*CHeaviside(x,eps)


    """No-slip boundary detection"""
    def WallBoundary(x, on_boundary):
        return x[1] < DOLFIN_EPS or x[1] - 2.22 > DOLFIN_EPS


    """Assemble boundary condition"""
    def assembleBC(self):
        self.bcs = DirichletBC(W.sub(0), Constant((0.0,0.0)), WallBoundary)


    """Build the system for Navier-Stokes simulation"""
    def assemble_NS_system(self):
        #Compute actual density and viscosity. The own functions should work
        #with class Function
        self.rho_old = self.rho(self.phi_old, 1e-4)
        self.mu_old  = self.mu(self.phi_old, 1e-4)

        # Define variational problem for step 1
        F1 = inner(self.rho_old*(self.u - self.u_old) / self.DT, self.v)*dx \
           + inner(self.rho_old*dot(self.u_old, nabla_grad(self.u)), self.v)*dx \
           + 1.0/self.Re*inner(sigma(self.mu_old, self.u, self.p_old), nabla_grad(self.v))*dx \
           + 1.0/self.At*dot(rho_old*self.g*self.e2, v)*dx\
           #- 1.0/self.Re*inner(sigma(mu_old, self.u, self.p)*n,v)*ds \
           - 1.0/self.Re*inner(self.surf_coeff*div(n)*n*CDelta(self.phi_old, 1e-4), self.v)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Assemble matrices and right-hand sides
        self.A1 = assemble(a1)
        self.b1 = assemble(L1)

        # Apply boundary conditions
        self.bcs.apply(self.A1)
        self.bcs.apply(self.b1)


    """Interior penalty method"""
    def IP(phi,l):
        r = self.alpha*self.h_avg*self.h_avg*inner(jump(grad(phi),self.n), jump(grad(l),n))*dS
        return r


    """Build the system for Level set simulation"""
    def assemble_Levelset_system(self):
        F2 = (self.phi - self.phi_old) / self.DT * self.l*dx \
           - inner(self.phi*self.u_curr, grad(self.l))*dx \
           + IP(self.phi, self.l)
        a2 = lhs(F2)
        L2 = rhs(F2)

        # Assemble matrix nad right-hand side
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

            t = conditional(gt(t + self.dt, self.t_end), self.t_end, t + self.dt)
