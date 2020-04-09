from My_Parameters import My_Parameters
from Auxiliary_Functions import *

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
            self.Re   = self.Param["Reynolds_number"]
            self.At   = self.Param["Atwood_number"]
            self.rho1 = self.Param["Lighter_density"]
            self.dt   = self.Param["Time_step"]
            self.tend = self.Param["End_time"]
            self.deg  = self.Param["Polynomial_degree"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

        #Compute heavier density
        self.rho2 = self.rho1*(1.0 + self.At)/(1.0 - self.At)

        #Compute viscosity: the 'lighter' viscosity will be computed by
        #Reynolds number, while for the 'heavier' we choose to impose a constant
        #density-viscosity ratio (arbitrary choice)
        self.mu1 = self.rho1*np.sqrt(9.81*self.At)/self.Re
        self.mu2 = self.rho2*self.mu1/self.rho1

        #Define function spaces
        self.V = VectorFunctionSpace(mesh, 'P', 2)
        self.Q = FunctionSpace(mesh, 'P', 1)

        #Define trial and test functions
        self.u   = TrialFunction(V)
        self.v   = TestFunction(V)
        self.p   = TrialFunction(Q)
        self.q   = TestFunction(Q)
        self.phi = TrialFunction(Q)
        self.l   = TestFunction(Q)

        #Define functions for solutions at previous and current time steps
        self.u_old    = Function(V)
        self.u_curr   = Function(V)
        self.p_old    = Function(Q)
        self.p_curr   = Function(Q)
        self.phi_old  = Function(Q)
        self.phi_curr = Function(Q)

        #Convert time step to constant FENICS function
        self.DT = Constant(self.dt)
        self.g  = Constant(9.81)
        self.e2 = Constant((0.0,1.0))

    """Build the mesh for the simulation"""
    def build_mesh(self):
        channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
        #Generate mesh
        n_points = self.Param["Number_vertices"]
        self.mesh = generate_mesh(domain, n_points)


    """Set the proper initial condition"""
    def set_initial_condition(self):
        self.phi_old.interpolate("tanh(x[1] - 2 - 0.1*cos(2*pi*x[0]))/(0.01*sqrt(2))")
        self.u_old.assign((0.0,0.0))


    """Auxiliary function to compute density"""
    def rho(self, q, eps):
        return self.rho1*(1.0 - CHeaviside(q,eps)) + self.rho2*CHeaviside(q,eps)


    """Auxiliary function to compute viscosity"""
    def mu(self, q, eps):
        return self.mu1*(1.0 - CHeaviside(q,eps)) + self.mu2*CHeaviside(q,eps)


    """Build the system for Navier-Stokes simulation"""
    def assemble_NS_system(self):
        #Compute actual density and viscosity. The own functions should work
        #with class Function
        rho_old = self.rho(self.phi_old, 1e-4)
        mu_old  = self.mu(self.phi_old, 1e-4)

        # Define variational problem for step 1
        F1 = inner(rho_old*(self.u - self.u_old) / self.DT, self.v)*dx \
           + inner(rho_old*dot(self.u_old, nabla_grad(self.u)), self.v)*dx \
           + 1.0/self.Re*inner(sigma(mu_old, self.u, self.p_old), nabla_grad(self.v))*dx \
           + 1.0/self.At*dot(rho_old*self.g*self.e2, v)*dx\
           - 1.0/self.Re*inner(sigma(mu_old, self.u, self.p_old)*n,v)*ds \
           - 1.0/self.Re*inner(self.surf_coeff*div(n)*n*CDelta(self.phi_old, 1e-4), self.v)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = 1.0/Re*inner(grad(self.p), grad(self.q))*dx
        L2 = inner(grad(self.p_old), grad(self.q))*dx \
           - (1/self.DT)*div(self.u_curr)*self.q*dx

        # Assemble matrices and right-hand sides
        self.A1 = assemble(a1)
        self.A2 = assemble(a2)

        self.b1 = assemble(L1)
        self.b2 = assemble(L2)


    """Interior penalty method"""
    def IP(phi,l):
        n = FacetNormal(self.mesh)
        h = CellSize(self.mesh)
        h_avg = (h('+') + h('-'))/2.0
        alpha = Constant(0.1)
        r = alpha('+')*h_avg*h_avg*inner(jump(grad(phi),n), jump(grad(l),n))*dS
        return r


    """Build the system for Level set simulation"""
    def assemble_Levelset_system(self):
        F3 = (self.phi - self.phi_old) / self.DT * self.l*dx \
           - inner(self.phi*self.u_curr, grad(self.l))*dx \
           + IP(self.phi, self.l)
        a3 = lhs(F3)
        L3 = rhs(F3)

        # Assemble matrices
        self.A3 = assemble(a3)
        self.b3 = assemble(L3)


    """Execute simulation"""
    def run(self):
        #Time-stepping
        t = self.dt
        while t < self.t_end:
            print("t = ",self.dt)

            solve(self.A1, self.u_curr.vector(), self.b1)
            solve(self.A2, self.p_curr.vector(), self.b2)
            self.u_curr = project(self.u_curr - self.DT*(self.p.curr - self.p.old), V)
            solve(self.A3, self.phi_curr.vector(), b3)

            plot(self.phi_curr, interactive=True)

            self.u_old.assign(self.u_curr)
            self.p_old.assign(self.p_curr)
            self.phi_old.assign(self.phi_curr)

            t += self.dt
