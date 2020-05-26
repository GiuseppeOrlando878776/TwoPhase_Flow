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

        #Define FE spaces
        self.V  = VectorFunctionSpace(self.mesh, "CG", self.deg + 1)
        self.P  = FunctionSpace(self.mesh, "CG" if self.deg > 0 else "DG", self.deg)
        self.Q  = FunctionSpace(self.mesh, "CG", 2)
        #self.Q2 = VectorFunctionSpace(self.mesh, "CG", 1)

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


    """Interior penalty method"""
    def IP(self, phi, l):
        r = self.alpha*self.h_avg*self.h_avg* \
            inner(jump(grad(phi),self.n_mesh), jump(grad(l),self.n_mesh))*dS
        return r


    """Weak formulation for Navier-Stokes"""
    def ICS_weak_form_1(self):
        F1 = (1.0/self.DT)*inner(self.rho(self.phi_curr,self.eps)*self.u - self.rho(self.phi_old,self.eps)*self.u_old, self.v)*dx  \
           - inner(dot(grad(self.v),self.u_old), self.rho(self.phi_curr,self.eps)*self.u)*dx \
           + 2.0/self.Re*inner(self.rho(self.phi_curr,self.eps)*D(self.u), grad(self.v))*dx \
           - self.p_old*div(self.v)*dx \
           + inner(self.rho(self.phi_curr,self.eps)*self.e2, self.v)*dx \
           #+ 1.0/(self.Bo*self.At)*div(self.n)*inner(self.n, self.v)*CDelta(self.phi_old, self.eps)*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        self.A1 = Matrix()
        self.b1 = Vector()

    def ICS_weak_form_2(self):
        #Define variational problem for step 2
        self.a1_bis = (1.0/self.rho(self.phi_curr,self.eps))*inner(grad(self.p), grad(self.q))*dx
        self.L1_bis = (1.0/self.rho(self.phi_curr,self.eps))*inner(grad(self.p_old), grad(self.q))*dx - \
                      (1.0/self.DT)*div(self.u_curr)*self.q*dx

        self.A1_bis = Matrix()
        self.b1_bis = Vector()

    def ICS_weak_form_3(self):
        #Define variational problem for step 3
        self.a1_tris = inner(self.u, self.v)*dx
        self.L1_tris = inner(self.u_curr, self.v)*dx - \
                       self.DT*inner(grad(self.p_curr - self.p_old), self.v)/self.rho(self.phi_curr,self.eps)*dx

        self.A1_tris = assemble(self.a1_tris)
        self.b1_tris = Vector()


    """Level-set weak formulation"""
    def LS_weak_form(self):
        F2 = (self.phi - self.phi_old)/self.DT*self.l*dx \
           + inner(self.u_old, grad(self.phi))*self.l*dx

        F2 += self.IP(self.phi, self.l)

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()


    """Set weak formulations"""
    def set_weak_forms(self):
        #Set variational problem for step 1 (Level-set)
        self.LS_weak_form()

        #Set variational problem for step 2 (Navier-Stokes)
        self.ICS_weak_form_1()
        self.ICS_weak_form_2()
        self.ICS_weak_form_3()


    """Build and solve the system for Navier-Stokes simulation"""
    def solve_Standard_NS_system(self):
        #Assemble matrices
        assemble(self.a1, tensor = self.A1)
        assemble(self.L1, tensor = self.b1)

        #Apply boundary conditions
        [bc.apply(self.A1) for bc in self.bcs]
        [bc.apply(self.b1) for bc in self.bcs]

        #Solve the first system
        solve(self.A1, self.u_curr.vector(), self.b1, "bicgstab", "default")

        #Assemble and solve the second system
        assemble(self.a1_bis, tensor = self.A1_bis)
        assemble(self.L1_bis, tensor = self.b1_bis)
        solve(self.A1_bis, self.p_curr.vector(), self.b1_bis, "bicgstab", "default")

        #Assemble and solve the third system
        assemble(self.L1_tris, tensor = self.b1_tris)
        solve(self.A1_tris, self.u_curr.vector(), self.b1_tris, "cg", "default")


    """Build the system for Level set simulation"""
    def solve_Levelset_system(self):
        # Assemble matrix and right-hand side
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        #Solve the level-set system
        solve(self.A2, self.phi_curr.vector(), self.b2, "gmres", "default")

        #Compute normal vector (in case we avoid reconstrution)
        #self.grad_phi.assign(project(grad(self.phi_curr), self.Q2))
        #self.n = self.grad_phi/sqrt(inner(self.grad_phi, self.grad_phi))


    """Plot the level-set function and compute the volume"""
    def plot_and_volume(self):
        #Extract vector for FE function
        self.phi_curr_vec = self.phi_curr.vector().get_local()

        #Construct vector of ones inside the bubble
        self.lev_set = 1.0*(self.phi_curr_vec < 0.0)

        #Assign vector to FE function
        self.tmp.vector().set_local(self.lev_set)

        #Plot the function just computed
        if(self.n_iter % 50 == 0):
            self.vtkfile_phi_draw << (self.tmp, self.t*self.t0)
            self.vtkfile_u << (self.u_curr, self.t*self.t0)
            self.rho_interp.assign(project(self.rho(self.phi_curr,self.eps), self.Q))
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
        self.vtkfile_phi_draw = File('/u/archive/laureandi/orlando/Sim69/phi_draw.pvd')
        self.vtkfile_u = File('/u/archive/laureandi/orlando/Sim69/u.pvd')
        self.vtkfile_rho = File('/u/archive/laureandi/orlando/Sim69/rho.pvd')
        while self.t <= self.t_end:
            begin(int(LogLevel.INFO) + 1,"t = " + str(self.t*self.t0) + " s")

            #Solve level-set
            begin(int(LogLevel.INFO) + 1,"Solving Level-set")
            self.solve_Levelset_system()
            end()

            #Solve Level-set reinit
            """
            try:
                begin(int(LogLevel.INFO) + 1,"Solving reinitialization")
                self.Levelset_reinit()
                end()
            except RuntimeError as e:
                print(e)
                print("Aborting simulation...")
                exit(1)
            """
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
