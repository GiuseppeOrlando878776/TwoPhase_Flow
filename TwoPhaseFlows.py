from dolfin import *

from Auxiliary_Functions import *

import warnings

class TwoPhaseFlows():
    """Default constructor"""
    def __init__(self):
        #Define auxiliary dictionaries to set proper stabilization,
        #solution method and reconstruction
        self.stab_dict = {'IP', 'SUPG', 'None'}
        self.NS_sol_dict = {'Standard', 'ICT'}
        self.reinit_method_dict = {'Non_Conservative_Hyperbolic', 'Non_Conservative_Elliptic', 'Conservative'}

        #Declare useful constant vector
        self.e1 = Constant((1.0, 0.0))
        self.e2 = Constant((0.0, 1.0))


    """Weak formulation for Navier-Stokes"""
    def NS_weak_form(self, u, p, v, q, u_old, dt, rho, mu, phi_curr, phi_old, eps, g, sigma, n_gamma, CDelta):
        #Check correctness of types
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")
        if(not callable(mu)):
            raise ValueError("The function to compute the viscosity must be a callable object")
        if(not callable(CDelta)):
            raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
        if(not isinstance(n_gamma, Function)):
            raise ValueError("n(the unit normal to the interface) must be an instance of Function")

        #Set weak formulation
        F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
           + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
           + 2.0*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
           - p*div(v)*dx \
           + div(u)*q*dx \
           + g*inner(rho(phi_curr, eps)*self.e2, v)*dx \
           + sigma*mgrad(phi_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(phi_curr, eps)*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()


    """Weak formulation for tentative velocity"""
    def ICT_weak_form_1(self, u, v, u_old, dt, rho, mu, phi_curr, phi_old, eps, g, sigma, n_gamma, CDelta):
        #Check the correctness of type
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")
        if(not callable(mu)):
            raise ValueError("The function to compute the viscosity must be a callable object")
        if(not callable(CDelta)):
            raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
        if(not isinstance(n_gamma, Function)):
            raise ValueError("n(the unit normal to the interface) must be an instance of Function")

        #Define variational formulation for step 1
        F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
           + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
           + 2.0*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
           - p_old*div(v)*dx \
           + g*inner(rho(phi_curr, eps)*self.e2, v)*dx \
           + sigma*mgrad(phi_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(phi_curr, eps)*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()


    """Weak formulation for pressure correction"""
    def ICT_weak_form_2(self, p, q, dt, u_curr, rho, phi_curr, eps):
        #Check the correctness of type
        if(not isinstance(u_curr, Function)):
            raise ValueError("u_curr must be an instance of Function")

        #Define variational problem for step 2
        self.a2_bis = (1.0/rho(phi_curr, eps))*inner(grad(p), grad(q))*dx
        self.L2_bis = (1.0/rho(phi_curr, eps))*inner(grad(p_old), grad(q))*dx - \
                      (1.0/dt)*div(u_curr)*q*dx

        self.A2_bis = Matrix()
        self.b2_bis = Vector()


    """Weak formulation for velocity projection"""
    def ICT_weak_form_3(self, u, v, dt, u_curr, p_curr, p_old, rho, phi_curr, eps):
        #Check the correctness of type
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")

        #Define variational problem for step 3
        self.a2_tris = inner(u, v)*dx
        self.L2_tris = inner(u_curr, v)*dx - \
                       dt*inner(grad(p_curr - p_old), v)/rho(phi_curr, eps)*dx

        self.A2_tris = assemble(self.a2_tris)
        self.b2_tris = Vector()


    """Interior penalty method"""
    def IP(self, phi, l, mesh, alpha = 0.1):
        #Extract cell diameter and facets's normal
        h = CellDiameter(mesh)
        n_mesh = FacetNormal(mesh)
        h_avg  = (h('+') + h('-'))/2.0

        #Compute the stabilization term
        r = alpha*h_avg*h_avg*inner(jump(grad(phi), n_mesh), jump(grad(l), n_mesh))*dS
        return r


    """SUPG method"""
    def SUPG(self, phi, l, phi_old, u_old, dt, mesh, Re):
        #Extract cell diameter
        h = CellDiameter(mesh)

        #Compute the stabilization term
        r = ((phi - phi_old)/dt + inner(u_old, grad(phi)))* \
            h/ufl.Max(2.0*sqrt(inner(u_old,u_old)), 4.0/(Re*h))*inner(u_old, grad(l))*dx
        return r


    """Level-set weak formulation"""
    def LS_weak_form(self, phi, l, phi_old, u_old, dt, mesh, method, param = None):
        #Check availability of the method before proceding
        assert method in self.stab_dict, "Stabilization method not available"

        #Check the correctness of type
        if(not isinstance(phi_old, Function)):
            raise ValueError("phi_old must be an instance of Function")

        #Save the dimension of the problem
        self.n_dim = mesh.geometry().dim()

        #Declare weak formulation
        F1 = ((phi - phi_old)/dt + inner(u_old, grad(phi)))*l*dx

        #Add stabilization term (if specified)
        if(method == 'SUPG'):
            #Check whether Reynolds number is really available
            assert param is not None, \
            "Reynolds number not available in order to use SUPG stabilization"

            #Add the stabilization term
            F1 += self.SUPG(phi, l, phi_old, u_old, dt, mesh, param)
        elif(method == 'IP'):
            #Check whether stabilization parameter is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use IP stabilization"

            #Add the stabilization term
            F1 += self.IP(phi, l, mesh, param)

        #Save corresponding weak forms and declare matrix and vector for solving
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        self.A1 = Matrix()
        self.b1 = Vector()


    """Weak form non-conservative reinitialization (hyperbolic version)"""
    def NCLSM_hyperbolic_weak_form(self, phi, l, phi0, phi_curr, dt_reinit, gamma_reinit, beta_reinit):
        #Check correctness of types
        if(not isinstance(phi0, Function)):
            raise ValueError("phi0 must be an instance of Function")

        #Declare weak formulation
        self.a1_reinit = phi/dt_reinit*l*dx
        self.L1_reinit = phi0/dt_reinit*l*dx + \
                         signp(phi_curr, gamma_reinit)*(1.0 - mgrad(phi0))*l*dx -\
                         beta_reinit*inner(grad(phi0), grad(l))*dx

        #Save the matrix that will not change and declare vector
        self.A1_reinit = assemble(self.a1_reinit)
        self.b1_reinit = Vector()


    """Weak form non-conservative reinitialization (elliptic version)"""
    def NCLSM_elliptic_weak_form(self, phi, l, phi0, phi_curr, CDelta, eps_reinit, beta_reinit = 1.0e3):
        #Check correctness of types
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")
        if(not callable(CDelta)):
            raise ValueError("CDelta must be a callable object")

        #Declare weak formulation
        self.a1_reinit = inner(grad(phi), grad(l))*dx + \
                         beta_reinit*phi*CDelta(phi_curr, eps_reinit)*dx
        self.L1_reinit = inner(grad(phi0)/mgrad(phi0), grad(l))*dx

        #Declare matrix and vector for solution
        self.A1_reinit = Matrix()
        self.b1_reinit = Vector()


    """Weak form conservative reinitialization"""
    def CLSM_weak_form(self, phi_intermediate, l, phi0, n_gamma, dt_reinit, eps_reinit):
        #Check correctness of types
        if(not isinstance(phi_intermediate, Function)):
            raise ValueError("phi_intermediate must be an instance of Function")

        #Save variational formulation
        self.F1_reinit = (phi_intermediate - phi0)/dt_reinit*l*dx \
                       - phi_intermediate*(1.0 - phi_intermediate)*inner(grad(l), n_gamma)*dx \
                       + eps_reinit*inner(grad(phi_intermediate), n_gamma)*inner(grad(l), n_gamma)*dx


    """Build the system for Level set simulation"""
    def solve_Levelset_system(self, phi_curr, n_gamma, Normal_Space):
        # Assemble matrix and right-hand side
        assemble(self.a1, tensor = self.A1)
        assemble(self.L1, tensor = self.b1)

        #Solve the level-set system
        solve(self.A1, phi_curr.vector(), self.b1, "gmres", "default")

        #Compute normal vector (in case we avoid reconstrution)
        n_gamma.assign(project(grad(phi_curr)/mgrad(phi_curr), Normal_Space))


    """Build and solve the system for Level set hyperbolic reinitialization (non-conservative)"""
    def NC_Levelset_hyperbolic_reinit(self, phi_curr, phi_intermediate, phi0, dt_reinit, n_gamma, Normal_Space, n_subiters = 10, tol = 1.0e-4):
        #Assign current solution
        phi0.assign(phi_curr)

        E_old = 1e10
        for n in range(n_subiters):
            #Assemble and solve the system
            assemble(self.L1_reinit, tensor = self.b1_reinit)
            solve(self.A1_reinit, phi_intermediate.vector(), self.b1_reinit, "cg" , "icc")

            #Compute the error and check no divergence
            error = (((phi_intermediate - phi0)/dt_reinit)**2)*dx
            E = sqrt(assemble(error))

            if(E_old < E):
                raise RuntimeError("Divergence at the reinitialization level (iteration " + str(n + 1) + ")")
            elif(E < tol):
                break

            #Set previous step solution
            phi0.assign(phi_intermediate)

        #Assign the reinitialized level-set to the current solution and
        #update normal vector to the interface (for Navier-Stokes)
        phi_curr.assign(phi_intermediate)
        n_gamma.assign(project(grad(phi_curr)/mgrad(phi_curr), Normal_Space))


    """Build and solve the system for Level set elliptic reinitialization (non-conservative)"""
    def NC_Levelset_elliptic_reinit(self, phi_curr, phi_intermediate, phi0, n_gamma, Normal_Space, n_subiters = 10, tol = 1.0e-4):
        #Assign current solution
        phi0.assign(phi_curr)

        #Build the matrix that will not change during the procedure
        assemble(self.a1_reinit, tensor = self.A1_reinit)

        E_old = 1e10
        for n in range(n_subiters):
            #Assemble and solve the system
            assemble(self.L1_reinit, tensor = self.b1_reinit)
            solve(self.A1_reinit, phi_intermediate.vector(), self.b1_reinit, "gmres" , "default")

            #Compute the error and check no divergence
            error = ((phi_intermediate - phi0)**2)*dx
            E = sqrt(assemble(error))

            if(E_old < E):
                raise RuntimeError("Divergence at the reinitialization level (iteration " + str(n + 1) + ")")
            elif(E < tol):
                break

            #Set previous step solution
            phi0.assign(phi_intermediate)

        #Assign the reinitialized level-set to the current solution and
        #update normal vector to the interface (for Navier-Stokes)
        phi_curr.assign(phi_intermediate)
        n_gamma.assign(project(grad(phi_curr)/mgrad(phi_curr), Normal_Space))


    """Build and solve the system for Level set reinitialization (conservative)"""
    def C_Levelset_reinit(self, phi_curr, phi_intermediate, phi0, dt_reinit, n_gamma, Normal_Space, n_subiters = 10, tol = 1.0e-4):
        #Assign the current solution
        phi0.assign(phi_curr)

        #Start the loop
        for n in range(n_subiters):
            #Solve the system
            solve(self.F1_reinit == 0, phi_intermediate, \
                  solver_parameters={"newton_solver": {'linear_solver': 'gmres', "preconditioner": "default"}}, \
                  form_compiler_parameters={"optimize": True})

            #Check if convergence has been reached
            error = (((phi_intermediate - phi0)/dt_reinit)**2)*dx
            E = sqrt(assemble(error))
            if(E < tol):
                break

            #Prepare for next iteration
            phi0.assign(phi_intermediate)

        #Assign the reinitialized level-set to the current solution and
        #update normal vector to the interface (for Navier-Stokes)
        phi_curr.assign(phi_intermediate)
        n_gamma.assign(project(grad(phi_curr)/mgrad(phi_curr), Normal_Space))


    """Build and solve the system for Navier-Stokes simulation"""
    def solve_Standard_NS_system(self, bcs, w_curr, u_curr, p_curr):
        # Assemble matrices and right-hand sides
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        # Apply boundary conditions
        for bc in self.bcs:
            bc.apply(self.A2)
            bc.apply(self.b2)

        #Solve the system
        solve(self.A2, w_curr.vector(), self.b2, "umfpack")

        #Save in seprated functions
        (self.u_curr, self.p_curr) = self.w_curr.split(True)


    """Build and solve the system for Navier-Stokes simulation"""
    def solve_ICT_NS_systems(self, bcs, u_curr, p_curr):
        #Assemble matrices
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        #Apply boundary conditions
        for bc in self.bcs:
            bc.apply(self.A2)
            bc.apply(self.b2)

        #Solve the first system
        solve(self.A2, u_curr.vector(), self.b2, "gmres", "default")

        #Assemble and solve the second system
        assemble(self.a2_bis, tensor = self.A2_bis)
        assemble(self.L2_bis, tensor = self.b2_bis)
        solve(self.A2_bis, p_curr.vector(), self.b2_bis, "gmres", "default")

        #Assemble and solve the third system
        assemble(self.L2_tris, tensor = self.b2_tris)
        solve(self.A2_tris, u_curr.vector(), self.b2_tris, "cg", "default")
