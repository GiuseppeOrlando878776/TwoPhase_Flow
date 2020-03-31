from My_Parameters import My_Parameters
from Auxiliary_Function import *

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
            self.dt   = self.Param["Time_step"]
            self.tend = self.Param["End_time"]
            self.deg  = self.Param["Polynomial_degree"]
        except RuntimeError as e:
            print(str(e) +  "\nPlease check configuration file")

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
        self.phi_old.interpolate("atan(x)*x[0]>1.0")

        #Define suitable representation for density. Since it is coupled with
        #interface evolution we choose the class 'Expression'
        self.rho = Expression(2.0*self.phi_old.eval(x)>0 + 1.0*self.phi_old.eval(x)<0)#to be set with interface and Heavised; it should be possible to be update


    """Build the system for Navier-Stokes simulation"""
    def assemble_NS_system(self):
        # Define variational problem for step 1
        F1 = self.rho*dot((self.u - self.u_old) / self.DT, self.v)*dx \
           + self.rho*dot(dot(self.u_old, nabla_grad(self.u_old)), self.v)*dx \
           + inner(sigma(U, p_n), epsilon(v))*dx \
           + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
           - dot(f, v)*dx
        a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        self.L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/DT)*div(u_)*q*dx

        # Assemble matrices
        self.A1 = assemble(a1)
        self.A2 = assemble(a2)
