from My_Parameters import My_Parameters
from Auxiliary_Functions import *
from Periodic_BC import WallBoundary

from fenics import *
import ufl, sys, math

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
            self.deg        = int(self.Param["Polynomial_degree"])
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
            n_points = int(self.Param["Number_vertices"])
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
        (self.u_curr, self.p_curr) = self.w_curr.split()
        self.phi_curr = Function(self.Q)

        #Define function for reinitialization
        self.phi0 = Function(self.Q)


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

        self.rho_old = Function(self.Q)
        self.mu_old  = Function(self.Q)
        self.signp   = Function(self.Q)


    """Interior penalty method"""
    def IP(self, phi, l):
        r = self.alpha*self.h_avg*self.h_avg*inner(jump(grad(phi),self.n), jump(grad(l),self.n))*dS
        return r


    """Set weak formulations"""
    def set_weak_forms(self):
        #Define variational problem for step 1 (Navier-Stokes)
        F1 = inner(self.rho_old*(self.u - self.u_old) / self.DT, self.v)*dx \
           + inner(self.rho_old*dot(self.u_old, nabla_grad(self.u)), self.v)*dx \
           + 1.0/self.Re*inner(sigma(self.mu_old, self.u, self.p_old), nabla_grad(self.v))*dx \
           + 1.0/self.At*dot(self.rho_old*self.g*self.e2, self.v)*dx
           #- 1.0/self.Re*inner(self.surf_coeff*div(self.n)*self.n*CDelta(self.phi_old, 1e-4), self.v)*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        self.A1 = Matrix()
        self.b1 = Vector()

        #Define variational problem for step 2 (Level-set)
        F2 = (self.phi - self.phi_old) / self.DT * self.l*dx \
           - inner(self.phi*self.u_curr, grad(self.l))*dx \
           + self.IP(self.phi, self.l)

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()

        #Set weak form for level-set reinitialization
        dt = Constant(0.0001)
        eps = Constant(1.0e-4)
        alpha = Constant(0.0625)

        self.a3 = self.phi/dt*self.l*dx
        self.L3 = self.phi0/dt*self.l*dx + self.signp*(1.0 - sqrt(dot(grad(self.phi0), grad(self.phi0))))*self.l*dx -\
                  alpha*inner(grad(self.phi0), grad(self.l))* dx


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
        self.signp = ufl.sign(self.phi)

        E_old = 1e10

        for n in range(10):
            #Set previous step solution
            self.phi0.assign(self.phi)

            #Solve the system
            solve(self.a3 == self.L3, self.phi, [])

            #Compute the error and check no divergence
            error = (((self.phi - self.phi0)/dt)**2)*dx
            E = sqrt(abs(assemble(error)))

            if(E_old < E):
                raise RuntimeError("Divergence at the reinitialization level (iteration )" + str(n + 1))

            E_old = E


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
            self.assemble_NS_system()
            solve(self.A1, self.w_curr.vector(), self.b1)
            (self.u_curr, self.p_curr) = self.w_curr.split()

            #Solve level-set
            self.assemble_Levelset_system()
            solve(self.A2, self.phi_curr.vector(), self.b2)

            #Apply reinitialization for level-set
            try:
                self.Levelset_reinit()
            except RuntimeError as e:
                print(e)
                print("Aborting simulation...")
                sys.exit(1)

            #Plot level-set solution
            plot(self.phi_curr, interactive = True)

            #Prepare to next step
            self.w_old.assign(self.w_curr)
            (self.u_old, self.p_old) = self.w_old.split()
            self.phi_old.assign(self.phi_curr)

            t = t + self.dt if t + self.dt <= self.t_end else self.t_end
