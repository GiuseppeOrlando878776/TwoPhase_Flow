from My_Parameters import My_Parameters
from Auxiliary_Functions import *

from fenics import *

class RayleghTaylor:
    """Class constructor"""
    def __init__(self, param_name):
        """
        Param --- class Parameters to store desired configuration
        Re    --- Reynolds number
        dt    --- Specified time step
        tend  --- End time of the simulation
        deg   --- Polynomial degree
        """

        self.Param = My_Parameters(param_name).get_param()

        try:
            self.Re   = self.Param["Reynolds_number"]
            self.At   = self.Param["Atwood_number"]
            self.rho1 = self.Param["Heavier_density"]
            self.mu1  = self.Param["Viscosity_heavier"]
            self.mu2  = self.Param["Viscosity_lighter"]
            self.dt   = self.Param["Time_step"]
            self.tend = self.Param["End_time"]
            self.deg  = self.Param["Polynomial_degree"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

        #Compute lighter density
        self.rho2 = self.rho1*(1.0 - self.At)/(1.0 + self.At)

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


    """Build the mesh for the simulation"""
    def build_mesh(self):
        channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
        #Generate mesh
        n_points = self.Param["Number_vertices"]
        self.mesh = generate_mesh(domain, n_points)


    """Set the proper initial condition"""
    def set_initial_condition(self):
        self.phi_old.interpolate("tanh(x[1] - 2 - 0.1*cos(2*pi*x[0]))/(0.01*sqrt(2))")
        self.u_old.assign(0.0,0.0)


    """Compute density"""
    def rho(self, q, eps):
        return self.rho1*(1.0 - CHeaviside(q,eps)) + self.rho2*CHeaviside(q,eps)

    """Compute viscosity"""
    def mu(self, q, eps):
        return self.mu1*(1.0 - CHeaviside(q,eps)) + self.mu2*CHeaviside(q,eps)


    """Build the system for Navier-Stokes simulation"""
    def assemble_NS_system(self):
        #Compute actual density and viscosity. The own functions should work
        #with class Function
        rho_old = self.rho(self.phi_old, 1e-4)
        mu_old  = self.mu(self.phi_old, 1e-4)

        # Define variational problem for step 1
        F1 = rho_old*dot((self.u - self.u_old) / self.DT, self.v)*dx \
           + rho_old*dot(dot(self.u_old, nabla_grad(self.u_old)), self.v)*dx \
           + inner(sigma(mu_old, U, self.p_old), D(v))*dx \
           + dot(self.p_old*n, v)*ds - dot(mu_old*nabla_grad(U)*n, v)*ds \
           - dot(f, v)*dx
        a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(self.p), nabla_grad(self.q))*dx
        self.L2 = dot(nabla_grad(self.p_old), nabla_grad(self.q))*dx - (1/DT)*div(self.u_curr)*self.q*dx

        # Assemble matrices
        self.A1 = assemble(a1)
        self.A2 = assemble(a2)


    """Build the system for Level set simulation"""
    def assemble_Levelset_system(self):
        F3 = (self.phi - self.phi_old) / self.DT * self.l*dx \
           + self.phi*dot(self.u_curr*nabla_grad(self.l))*dx
        a3 = lhs(F3)
        self.L3 = rhs(F3)

        # Assemble matrices
        self.A3 = assemble(a3)
