from Auxiliary_Functions import *

import warnings

class TwoPhaseFlows():
    """Default constructor"""
    def __init__(self):
        #Define auxiliary dictionaries to set proper stabilization,
        #solution method for Navier-Stokes and reinitialization
        self.LS_sol_dict = {'DG', 'Continuous'}
        self.stab_dict = {'IP', 'SUPG', 'None'}
        self.NS_sol_dict = {'Standard', 'ICT'}
        self.reinit_method_dict = {'Non_Conservative_Hyperbolic', 'Conservative'}

        #Save solvers and preconditioners settings; in this way we prepare ourselves
        #in case the option to pass it through configuration file will be added in a future version
        self.solver_Levset = "gmres"
        self.precon_Levset = "default"
        self.solver_recon = "gmres"
        self.precon_recon = "default"
        self.solver_Standard_NS = "mumps"
        self.precon_Standard_NS = "default"
        self.solver_ICT_1 = "gmres"
        self.precon_ICT_1 = "default"
        self.solver_ICT_2 = "gmres"
        self.precon_ICT_2 = "default"
        self.solver_ICT_3 = "gmres"
        self.precon_ICT_3 = "default"

        #Declare useful constant vectors
        self.e1 = Constant((1.0, 0.0))
        self.e2 = Constant((0.0, 1.0))


    """Weak formulation for Navier-Stokes"""
    def NS_weak_form(self, u, p, v, q, u_old, dt, rho, mu, phi_curr, phi_old, eps, n_gamma = None, CDelta = None, **kwargs):
        #Check correctness of types
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")
        if(not callable(mu)):
            raise ValueError("The function to compute the viscosity must be a callable object")

        #Set weak formulation
        if(len(kwargs) == 2):
            assert 'g' in kwargs, "Error in the parameters for dimensional version of NS: 'g' not found (check function call)"
            assert 'sigma' in kwargs, "Error in the parameters for dimensional version of NS: 'sigma' not found (check function call)"
            g = kwargs.get('g')
            sigma = kwargs.get('sigma')
            F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
               + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
               + 2.0*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
               - p*div(v)*dx \
               + div(u)*q*dx \
               + g*inner(rho(phi_curr, eps)*self.e2, v)*dx
            if(sigma > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += sigma*mgrad(phi_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(phi_curr, eps)*dx
        elif(len(kwargs) == 3):
            assert 'Re' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Re' not found (check function call)"
            assert 'Fr' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Fr' not found (check function call)"
            assert 'We' in kwargs, "Error in the parameters for non-dimensional version of NS: 'We' not found (check function call)"
            Re = kwargs.get('Re')
            Fr = kwargs.get('Fr')
            We = kwargs.get('We')
            F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
               + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
               + 2.0/Re*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
               - p*div(v)*dx \
               + div(u)*q*dx \
               + 1.0/(Fr*Fr)*inner(rho(phi_curr, eps)*self.e2, v)*dx
            if(We > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += (1.0/We)*mgrad(phi_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(phi_curr, eps)*dx
        else:
            raise ValueError("Wrong number of arguments in Standard NS weak form setting (check function call)")

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = PETScMatrix()
        self.b2 = PETScVector()


    """Weak formulation for tentative velocity"""
    def ICT_weak_form_1(self, u, v, u_old, p_old, dt, rho, mu, phi_curr, phi_old, eps, n_gamma = None, CDelta = None, **kwargs):
        #Check the correctness of type
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")
        if(not callable(mu)):
            raise ValueError("The function to compute the viscosity must be a callable object")

        #Define variational formulation for step 1
        if(len(kwargs) == 2):
            assert 'g' in kwargs, "Error in the parameters for dimensional version of NS: 'g' not found (check function call)"
            assert 'sigma' in kwargs, "Error in the parameters for dimensional version of NS: 'sigma' not found (check function call)"
            g = kwargs.get('g')
            sigma = kwargs.get('sigma')
            F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
               + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
               + 2.0*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
               - p_old*div(v)*dx \
               + g*inner(rho(phi_curr, eps)*self.e2, v)*dx
            if(sigma > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += sigma*mgrad(phi_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(phi_curr, eps)*dx
        elif(len(kwargs) == 3):
            assert 'Re' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Re' not found (check function call)"
            assert 'Fr' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Fr' not found (check function call)"
            assert 'We' in kwargs, "Error in the parameters for non-dimensional version of NS: 'We' not found (check function call)"
            Re = kwargs.get('Re')
            Fr = kwargs.get('Fr')
            We = kwargs.get('We')
            F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
               + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
               + 2.0/Re*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
               - p_old*div(v)*dx \
               + 1.0/(Fr*Fr)*inner(rho(phi_curr, eps)*self.e2, v)*dx
            if(We > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += (1.0/We)*mgrad(phi_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(phi_curr, eps)*dx
        else:
            raise ValueError("Wrong number of arguments in ICT-Step 1 weak form setting (check function call)")

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = PETScMatrix()
        self.b2 = PETScVector()


    """Weak formulation for pressure correction"""
    def ICT_weak_form_2(self, p, q, dt, p_old, u_curr, rho, phi_curr, eps):
        #Check the correctness of type
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")
        if(not isinstance(u_curr, Function)):
            raise ValueError("u_curr must be an instance of Function")
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")

        #Define variational problem for step 2 of ICT
        self.a2_bis = (1.0/rho(phi_curr, eps))*inner(grad(p), grad(q))*dx
        self.L2_bis = (1.0/rho(phi_curr, eps))*inner(grad(p_old), grad(q))*dx - \
                      (1.0/dt)*div(u_curr)*q*dx

        #Declare matrix and vector for the linear system solution
        self.A2_bis = PETScMatrix()
        self.b2_bis = PETScVector()


    """Weak formulation for velocity projection"""
    def ICT_weak_form_3(self, u, v, dt, u_curr, p_curr, p_old, rho, phi_curr, eps):
        #Check the correctness of type
        if(not isinstance(u_curr, Function)):
            raise ValueError("u_curr must be an instance of Function")
        if(not isinstance(p_curr, Function)):
            raise ValueError("p_curr must be an instance of Function")
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")

        #Define variational problem for step 3 of ICT
        self.a2_tris = inner(u, v)*dx
        self.L2_tris = inner(u_curr, v)*dx - \
                       dt*inner(grad(p_curr - p_old), v)/rho(phi_curr, eps)*dx

        #Save matrix (that will not change during the computations) and declare vector
        self.A2_tris = assemble(self.a2_tris)
        self.b2_tris = PETScVector()


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
    def SUPG(self, phi, l, phi_old, u_old, dt, mesh, scaling):
        #Extract cell diameter
        h = CellDiameter(mesh)

        #Compute the stabilization term
        r = ((phi - phi_old)/dt + inner(u_old, grad(phi)))* \
            scaling*h/ufl.Max(2.0*norm(u_old,'L2'),1.0e-3/h)*inner(u_old, grad(l))*dx
        return r


    """Level-set weak formulation"""
    def LS_weak_form(self, phi, l, phi_old, u_old, dt, mesh, method, param = None):
        #Check availability of the method before proceding
        assert method in self.stab_dict, "Stabilization method(" + method + ") not available"

        #Check the correctness of type
        if(not isinstance(phi_old, Function)):
            raise ValueError("phi_old must be an instance of Function")
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")

        #Save the dimension of the problem
        self.n_dim = mesh.geometry().dim()

        #Declare weak formulation
        F1 = ((phi - phi_old)/dt + inner(u_old, grad(phi)))*l*dx

        #Add stabilization term (if specified)
        if(method == 'SUPG'):
            #Check whether Reynolds number is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use SUPG stabilization (check the call of the function)"

            #Add the stabilization term
            F1 += self.SUPG(phi, l, phi_old, u_old, dt, mesh, param)
        elif(method == 'IP'):
            #Check whether stabilization parameter is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use IP stabilization (check the call of the function)"

            #Add the stabilization term
            F1 += self.IP(phi, l, mesh, param)

        #Save corresponding weak forms
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        #Declare matrix and vector for solving
        self.A1 = PETScMatrix()
        self.b1 = PETScVector()


    """Weak form non-conservative reinitialization (hyperbolic version)"""
    def NCLSM_hyperbolic_weak_form(self, phi, l, phi0, phi_curr, dt_reinit, gamma_reinit, beta_reinit):
        #Check correctness of types
        if(not isinstance(phi0, Function)):
            raise ValueError("phi0 must be an instance of Function")
        if(not isinstance(phi_curr, Function)):
            raise ValueError("phi_curr must be an instance of Function")

        #Declare weak formulation
        self.a1_reinit = (phi/dt_reinit)*l*dx
        self.L1_reinit = (phi0/dt_reinit)*l*dx \
                       + signp(phi_curr, gamma_reinit)*(1.0 - mgrad(phi0))*l*dx \
                       - beta_reinit*inner(grad(phi0), grad(l))*dx

        #Save the matrix (that will not change during computations) and declare vector
        self.A1_reinit = assemble(self.a1_reinit)
        self.b1_reinit = PETScVector()


    """Weak form conservative reinitialization"""
    def CLSM_weak_form(self, phi_intermediate, l, phi0, n_gamma, dt_reinit, eps_reinit):
        #Check correctness of types
        if(not isinstance(phi_intermediate, Function)):
            raise ValueError("phi_intermediate must be an instance of Function")
        if(not isinstance(phi0, Function)):
            raise ValueError("phi0 must be an instance of Function")
        if(not isinstance(n_gamma, Function)):
            raise ValueError("n_gamma must be an instance of Function")

        #Save variational formulation
        self.F1_reinit = (phi_intermediate - phi0)/dt_reinit*l*dx \
                       - phi_intermediate*(1.0 - phi_intermediate)*inner(grad(l), n_gamma)*dx \
                       + eps_reinit*inner(grad(phi_intermediate), n_gamma)*inner(grad(l), n_gamma)*dx


    """Build and solve the system for Level set transport"""
    def solve_Levelset_system(self, phi_curr):
        # Assemble matrix and right-hand side
        assemble(self.a1, tensor = self.A1)
        assemble(self.L1, tensor = self.b1)

        #Solve the level-set system
        solve(self.A1, phi_curr.vector(), self.b1, self.solver_Levset, self.precon_Levset)


    """Build and solve the system for Level set hyperbolic reinitialization (non-conservative)"""
    def NC_Levelset_hyperbolic_reinit(self, phi_curr, phi_intermediate, phi0, dt_reinit, n_subiters = 10, tol = 1.0e-4):
        #Assign current solution
        phi0.assign(phi_curr)

        #Start loop
        E_old = 1e10
        for n in range(n_subiters):
            #Assemble and solve the system
            assemble(self.L1_reinit, tensor = self.b1_reinit)
            solve(self.A1_reinit, phi_intermediate.vector(), self.b1_reinit, self.solver_recon , self.precon_recon)

            #Compute the L2-error and check no divergence
            error = (((phi_intermediate - phi0)/dt_reinit)**2)*dx
            E = sqrt(assemble(error))

            if(E_old < E):
                raise RuntimeError("Divergence at the reinitialization level (iteration " + str(n + 1) + ")")
            elif(E < tol):
                break

            #Set previous step solution
            phi0.assign(phi_intermediate)

        #Assign the reinitialized level-set to the current solution
        phi_curr.assign(phi_intermediate)


    """Build and solve the system for Level set reinitialization (conservative)"""
    def C_Levelset_reinit(self, phi_curr, phi_intermediate, phi0, dt_reinit, n_subiters = 10, tol = 1.0e-4):
        #Assign the current solution
        phi0.assign(phi_curr)

        #Start the loop
        for n in range(n_subiters):
            #Solve the system
            solve(self.F1_reinit == 0, phi_intermediate, \
                  solver_parameters={"newton_solver": {"linear_solver": self.solver_recon, "preconditioner": self.precon_recon,\
                                     "maximum_iterations": 20, "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}}, \
                  form_compiler_parameters={"optimize": True})

            #Check if convergence has been reached
            error = (((phi_intermediate - phi0)/dt_reinit)**2)*dx
            E = sqrt(assemble(error))
            if(E < tol):
                break

            #Prepare for next iteration
            phi0.assign(phi_intermediate)

        #Assign the reinitialized level-set to the current solution
        phi_curr.assign(phi_intermediate)


    """Build and solve the system for Navier-Stokes part using Standard method"""
    def solve_Standard_NS_system(self, bcs, w_curr):
        # Assemble matrices and right-hand sides
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        # Apply boundary conditions
        for bc in self.bcs:
            bc.apply(self.A2)
            bc.apply(self.b2)

        #Solve the system
        solve(self.A2, w_curr.vector(), self.b2, self.solver_Standard_NS, self.precon_Standard_NS)


    """Build and solve the system for Navier-Stokes part using ICT method"""
    def solve_ICT_NS_systems(self, bcs, u_curr, p_curr):
        #Assemble matrix and right-hand side for the first step
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        #Apply boundary conditions
        for bc in self.bcs:
            bc.apply(self.A2)
            bc.apply(self.b2)

        #Solve the first system
        solve(self.A2, u_curr.vector(), self.b2, self.solver_ICT_1, self.precon_ICT_1)

        #Assemble and solve the second system
        assemble(self.a2_bis, tensor = self.A2_bis)
        assemble(self.L2_bis, tensor = self.b2_bis)
        solve(self.A2_bis, p_curr.vector(), self.b2_bis, self.solver_ICT_2, self.precon_ICT_2)

        #Assemble and solve the third system
        assemble(self.L2_tris, tensor = self.b2_tris)
        solve(self.A2_tris, u_curr.vector(), self.b2_tris, self.solver_ICT_3, self.precon_ICT_3)
