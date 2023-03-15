from Auxiliary_Functions import *

import warnings

class TwoPhaseFlows():
    """Default constructor"""
    def __init__(self):
        #Define auxiliary dictionaries to set proper volume fraction solver,
        #stabilization (if needed) and solution method for Navier-Stokes and
        self.VF_sol_dict = {'Continuous', 'DG'}
        self.stab_dict = {'IP', 'SUPG', 'None'}
        self.NS_sol_dict = {'Standard', 'ICT'}

        #Save solvers and preconditioners settings; in this way we prepare ourselves
        #in case the option to pass it through configuration file will be added in a future version
        self.solver_VF = "gmres"
        self.precon_VF = "default"
        self.solver_area = "gmres"
        self.precon_area = "default"
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
    def NS_weak_form(self, u, p, v, q, u_old, dt, rho, mu, alpha_curr, alpha_old, n_gamma = None, area_curr = None, **kwargs):
        #Check correctness of types
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")
        if(not isinstance(alpha_curr, Function)):
            raise ValueError("alpha_curr must be an instance of Function")
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
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_curr, eps)*u - rho(alpha_old, eps)*u_old, v)*dx \
               + inner(rho(alpha_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
               + Constant(2.0)*inner(mu(alpha_curr, eps)*D(u), D(v))*dx \
               - p*div(v)*dx \
               + div(u)*q*dx \
               + g*inner(rho(alpha_curr, eps)*self.e2, v)*dx
            if(sigma > DOLFIN_EPS):
                if(not isinstance(area_curr, Function)):
                    raise ValueError("The area approximation must be an instance of Function")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(sigma)*area_curr*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*dx
        elif(len(kwargs) == 3):
            assert 'Re' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Re' not found (check function call)"
            assert 'Fr' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Fr' not found (check function call)"
            assert 'We' in kwargs, "Error in the parameters for non-dimensional version of NS: 'We' not found (check function call)"
            Re = kwargs.get('Re')
            Fr = kwargs.get('Fr')
            We = kwargs.get('We')
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_curr, eps)*u - rho(alpha_old, eps)*u_old, v)*dx \
               + inner(rho(alpha_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
               + Constant(2.0/Re)*inner(mu(alpha_curr, eps)*D(u), D(v))*dx \
               - p*div(v)*dx \
               + div(u)*q*dx \
               + Constant(1.0/(Fr*Fr))*inner(rho(alpha_curr, eps)*self.e2, v)*dx
            if(We > DOLFIN_EPS):
                if(not isinstance(area_curr, Function)):
                    raise ValueError("The area approximation must be an instance of Function")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(1.0/We)*area_curr*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*dx
        else:
            raise ValueError("Wrong number of arguments in Standard NS weak form setting (check function call)")

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = PETScMatrix()
        self.b2 = PETScVector()


    """SUPG method for NS"""
    def SUPG_NS_ICT(self, u, v, u_old, p_old, dt, rho, mu, alpha_curr, alpha_old, eps, n_gamma, area_curr, g, sigma, \
                    mesh, scaling):
        #Extract cell diameter
        h = CellDiameter(mesh)

        #Compute the stabilization term
        r = scaling*h/ufl.Max(2.0*sqrt(inner(u_old, u_old)),1.0e-3/h)* \
            inner((Constant(1.0)/dt)*(rho(alpha_curr, eps)*u - rho(alpha_old, eps)*u_old) + \
                  rho(alpha_curr, eps)*dot(u_old, nabla_grad(u)) - \
                  div(Constant(2.0)*mu(alpha_curr, eps)*D(u)) + \
                  grad(p_old) + \
                  g*rho(alpha_curr, eps)*self.e2 + \
                  Constant(sigma)*div(area_curr*(Identity(self.n_dim) - outer(n_gamma, n_gamma))), \
                  rho(alpha_curr, eps)*dot(u_old, nabla_grad(v)))*dx

        return r


    """Weak formulation for tentative velocity"""
    def ICT_weak_form_1(self, u, v, u_old, u_k, p_old, dt, rho, mu, alpha_k, alpha_old, eps, mesh, scaling, \
                        n_gamma = None, area_k = None, **kwargs):
        #Check the correctness of type
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")
        if(not isinstance(u_k, Function)):
            raise ValueError("u_k must be an instance of Function")
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")
        if(not isinstance(alpha_k, Function)):
            raise ValueError("alpha_k must be an instance of Function")
        if(not isinstance(alpha_old, Function)):
            raise ValueError("alpha_old must be an instance of Function")
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
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_k, eps)*u - rho(alpha_old, eps)*u_old, v)*dx \
               + inner(rho(alpha_k, eps)*dot(u_k, nabla_grad(u)), v)*dx \
               + Constant(2.0)*inner(mu(alpha_k, eps)*D(u), D(v))*dx \
               - p_old*div(v)*dx \
               + g*inner(rho(alpha_k, eps)*self.e2, v)*dx
            if(sigma > DOLFIN_EPS):
                if(not isinstance(area_k, Function)):
                    raise ValueError("The area approximation must be an instance of Function")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(sigma)*area_k*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*dx
                #F2 += self.SUPG_NS_ICT(u, v, u_old, p_old, dt, rho, mu, alpha_curr, alpha_old, eps, n_gamma, area_curr, \
                #                       g, sigma, mesh, scaling)
        elif(len(kwargs) == 3):
            assert 'Re' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Re' not found (check function call)"
            assert 'Fr' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Fr' not found (check function call)"
            assert 'We' in kwargs, "Error in the parameters for non-dimensional version of NS: 'We' not found (check function call)"
            Re = kwargs.get('Re')
            Fr = kwargs.get('Fr')
            We = kwargs.get('We')
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_curr, eps)*u - rho(alpha_old, eps)*u_old, v)*dx \
               + inner(rho(alpha_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
               + Constant(2.0/Re)*inner(mu(alpha_curr, eps)*D(u), D(v))*dx \
               - p_old*div(v)*dx \
               + Constant(1.0/(Fr*Fr))*inner(rho(alpha_curr, eps)*self.e2, v)*dx
            if(We > DOLFIN_EPS):
                if(not isinstance(area_curr, Function)):
                    raise ValueError("The area approximation must be an instance of Function")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(1.0/We)*area_curr*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*dx
        else:
            raise ValueError("Wrong number of arguments in ICT-Step 1 weak form setting (check function call)")

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = PETScMatrix()
        self.b2 = PETScVector()


    """Weak formulation for pressure correction"""
    def ICT_weak_form_2(self, p, q, dt, p_old, u_curr, rho, alpha_curr, eps):
        #Check the correctness of type
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")
        if(not isinstance(u_curr, Function)):
            raise ValueError("u_curr must be an instance of Function")
        if(not isinstance(alpha_curr, Function)):
            raise ValueError("alpha_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")

        #Define variational problem for step 2 of ICT
        self.a2_bis = (Constant(1.0)/rho(alpha_curr, eps))*inner(grad(p), grad(q))*dx
        self.L2_bis = (Constant(1.0)/rho(alpha_curr, eps))*inner(grad(p_old), grad(q))*dx - \
                      (Constant(1.0)/dt)*div(u_curr)*q*dx

        #Declare matrix and vector for the linear system solution
        self.A2_bis = PETScMatrix()
        self.b2_bis = PETScVector()


    """Weak formulation for velocity projection"""
    def ICT_weak_form_3(self, u, v, dt, u_curr, p_curr, p_old, rho, alpha_curr, eps):
        #Check the correctness of type
        if(not isinstance(u_curr, Function)):
            raise ValueError("u_curr must be an instance of Function")
        if(not isinstance(p_curr, Function)):
            raise ValueError("p_curr must be an instance of Function")
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")
        if(not isinstance(alpha_curr, Function)):
            raise ValueError("alpha_curr must be an instance of Function")
        if(not callable(rho)):
            raise ValueError("The function to compute the density must be a callable object")

        #Define variational problem for step 3 of ICT
        self.a2_tris = inner(u, v)*dx
        self.L2_tris = inner(u_curr, v)*dx - \
                       dt*inner(grad(p_curr - p_old), v)/rho(alpha_curr, eps)*dx

        #Save matrix (that will not change during the computations) and declare vector
        self.A2_tris = assemble(self.a2_tris)
        self.b2_tris = PETScVector()


    """Interior penalty method"""
    def IP(self, alpha, l, mesh, coeff = 0.1):
        #Extract cell diameter and facets's normal
        h = CellDiameter(mesh)
        n_mesh = FacetNormal(mesh)
        h_avg  = (h('+') + h('-'))/2.0

        #Compute the stabilization term
        r = coeff*h_avg*h_avg*inner(jump(grad(alpha), n_mesh), jump(grad(l), n_mesh))*dS
        return r


    """SUPG method"""
    def SUPG(self, alpha, l, alpha_old, u_k, dt, mesh, scaling):
        #Extract cell diameter
        h = CellDiameter(mesh)

        #Compute the stabilization term
        r = ((alpha - alpha_old)/dt + inner(u_k, grad(alpha)))* \
            scaling*h/ufl.Max(2.0*sqrt(inner(u_k, u_k)),1.0e-3/h)*inner(u_k, grad(l))*dx
        return r


    """SUPG method for area"""
    def SUPG_area(self, area, l, area_old, u_k, dt, mesh, n, scaling):
        #Extract cell diameter
        h = CellDiameter(mesh)

        #Compute the stabilization term
        r = ((area - area_old)/dt + inner(u_k, grad(area)) + area*inner(dot(n, nabla_grad(u_k)), n))* \
            scaling*h/ufl.Max(2.0*sqrt(inner(u_k, u_k)),1.0e-3/h)*inner(u_k, grad(l))*dx
        return r


    """Volume fraction advection weak formulation (continuous formulation)"""
    def VF_weak_form(self, alpha, l, alpha_old, u_k, dt, mesh, method, param = None):
        #Check availability of the method before proceding
        assert method in self.stab_dict, "Stabilization method(" + method + ") not available"

        #Check the correctness of type
        if(not isinstance(alpha_old, Function)):
            raise ValueError("alpha_old must be an instance of Function")
        if(not isinstance(u_k, Function)):
            raise ValueError("u_k must be an instance of Function")

        #Save the dimension of the problem
        self.n_dim = mesh.geometry().dim()

        #Declare weak formulation
        F1 = ((alpha - alpha_old)/dt + inner(u_k, grad(alpha)))*l*dx

        #Add stabilization term (if specified)
        if(method == 'SUPG'):
            #Check whether Reynolds number is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use SUPG stabilization (check the call of the function)"

            #Add the stabilization term
            F1 += self.SUPG(alpha, l, alpha_old, u_k, dt, mesh, param)
        elif(method == 'IP'):
            #Check whether stabilization parameter is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use IP stabilization (check the call of the function)"

            #Add the stabilization term
            F1 += self.IP(alpha, l, mesh, param)

        #Save corresponding weak forms
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        #Declare matrix and vector for solving
        self.A1 = PETScMatrix()
        self.b1 = PETScVector()


    """Volume fraction advection (DG discretization)"""
    def VF_weak_form_DG(self, alpha, l, alpha_old, u_old, dt, mesh, param):
        #Check the correctness of type
        if(not isinstance(alpha_old, Function)):
            raise ValueError("alpha_old must be an instance of Function")
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")

        #Save the dimension of the problem
        self.n_dim = mesh.geometry().dim()

        #Save the normal to internal facets
        n_mesh = FacetNormal(mesh)

        #Declare weak formulation
        F1 = (alpha - alpha_old)/dt*l*dx - alpha*inner(u_old, grad(l))*dx \
           + avg(alpha)*inner(u_old, jump(l,n_mesh))*dS \
           + 0.5*abs(inner(u_old, n_mesh('+')))*inner(jump(alpha,n_mesh), jump(l,n_mesh))*dS \
           + inner(u_old,n_mesh)*alpha*l*ds \
        #F1 = (alpha - alpha_old)/dt*l*dx - alpha*inner(u_old, grad(l))*dx \
        #   + avg(alpha)*inner(u_old, jump(l,n_mesh))*dS \
        #   + avg(l)*inner(u_old, jump(alpha,n_mesh))*dS \
        #   + param*inner(jump(alpha,n_mesh), jump(l,n_mesh))*dS

        #Save corresponding weak forms
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        #Declare matrix and vector for solving
        self.A1 = PETScMatrix()
        self.b1 = PETScVector()


    """Weak form conservative reinitialization"""
    def CLSM_weak_form(self, alpha_intermediate, l, alpha0, n_gamma, dt_reinit, eps_reinit):
        #Check correctness of types
        if(not isinstance(alpha_intermediate, Function)):
            raise ValueError("alpha_intermediate must be an instance of Function")
        if(not isinstance(alpha0, Function)):
            raise ValueError("alpha0 must be an instance of Function")
        if(not isinstance(n_gamma, Function)):
            raise ValueError("n_gamma must be an instance of Function")

        #Save variational formulation
        self.F1_reinit = (alpha_intermediate - alpha0)/dt_reinit*l*dx \
                       - alpha_intermediate*(1.0 - alpha_intermediate)*inner(grad(l), n_gamma)*dx \
                       + eps_reinit*inner(grad(alpha_intermediate), n_gamma)*inner(grad(l), n_gamma)*dx


    """Area advection weak formulation"""
    def area_weak_form(self, area, l, area_old, u_k, dt, mesh, n, method, param = None):
        #Check availability of the method before proceding
        assert method in self.stab_dict, "Stabilization method(" + method + ") not available"

        #Check the correctness of type
        if(not isinstance(area_old, Function)):
            raise ValueError("area_old must be an instance of Function")
        if(not isinstance(u_k, Function)):
            raise ValueError("u_k must be an instance of Function")

        #Declare weak formulation
        F1 = ((area - area_old)/dt + inner(u_k, grad(area)))*l*dx \
           + area*inner(dot(n, nabla_grad(u_k)), n)*l*dx


        #Add stabilization term (if specified)
        if(method == 'SUPG'):
            #Check whether Reynolds number is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use SUPG stabilization (check the call of the function)"

            #Add the stabilization term
            F1 += self.SUPG_area(area, l, area_old, u_k, dt, mesh, n, param)
        elif(method == 'IP'):
            #Check whether stabilization parameter is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use IP stabilization (check the call of the function)"

            #Add the stabilization term
            F1 += self.IP(area, l, mesh, param)

        #Save corresponding weak forms
        self.a1_area = lhs(F1)
        self.L1_area = rhs(F1)

        #Declare matrix and vector for solving
        self.A1_area = PETScMatrix()
        self.b1_area = PETScVector()


    """Build and solve the system for volume fraction transport"""
    def solve_VolumeFraction_system(self, alpha_curr):
        #Assemble matrix and right-hand side
        assemble(self.a1, tensor = self.A1)
        assemble(self.L1, tensor = self.b1)

        #Solve the level-set system
        solve(self.A1, alpha_curr.vector(), self.b1, self.solver_VF, self.precon_VF)


    """Build and solve the system for area evolution"""
    def solve_Area_system(self, area_curr):
        #Assemble matrix and right-hand side
        assemble(self.a1_area, tensor = self.A1_area)
        assemble(self.L1_area, tensor = self.b1_area)

        #Solve the level-set system
        solve(self.A1_area, area_curr.vector(), self.b1_area, self.solver_area, self.precon_area)


    """Build and solve the system for Volume fraction reinitialization (conservative)"""
    def C_VolumeFraction_reinit(self, alpha_curr, alpha_intermediate, alpha0, dt_reinit, n_subiters = 10, tol = 1.0e-4):
        #Assign the current solution
        alpha0.assign(alpha_curr)

        #Start the loop
        for n in range(n_subiters):
            #Solve the system
            solve(self.F1_reinit == 0, alpha_intermediate, \
                  solver_parameters={"newton_solver": {"linear_solver": self.solver_recon, "preconditioner": self.precon_recon,\
                                     "maximum_iterations": 20, "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6}}, \
                  form_compiler_parameters={"optimize": True})

            #Check if convergence has been reached
            error = (((alpha_intermediate - alpha0)/dt_reinit)**2)*dx
            E = sqrt(assemble(error))
            if(E < tol):
                break

            #Prepare for next iteration
            alpha0.assign(alpha_intermediate)

        #Assign the reinitialized level-set to the current solution
        alpha_curr.assign(alpha_intermediate)


    """Build and solve the system for Navier-Stokes part using Standard method"""
    def solve_Standard_NS_system(self, bcs, w_curr):
        #Assemble matrices and right-hand sides
        assemble(self.a2, tensor = self.A2)
        assemble(self.L2, tensor = self.b2)

        #Apply boundary conditions
        for bc in bcs:
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
        for bc in bcs:
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
