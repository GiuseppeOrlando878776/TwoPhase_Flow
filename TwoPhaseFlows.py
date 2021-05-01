from Auxiliary_Functions import *

import warnings

class TwoPhaseFlows():
    """Default constructor"""
    def __init__(self):
        #Define auxiliary dictionaries to set proper stabilization and
        #solution method for Navier-Stokes and
        self.stab_dict = {'IP', 'SUPG', 'None'}
        self.NS_sol_dict = {'Standard', 'ICT'}

        #Save solvers and preconditioners settings; in this way we prepare ourselves
        #in case the option to pass it through configuration file will be added in a future version
        self.solver_VF = "gmres"
        self.precon_VF = "default"
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
    def NS_weak_form(self, u, p, v, q, u_old, dt, rho, mu, alpha_curr, alpha_old, n_gamma = None, CDelta = None, **kwargs):
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
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_curr)*u - rho(alpha_old)*u_old, v)*dx \
               + inner(rho(alpha_curr)*dot(u_old, nabla_grad(u)), v)*dx \
               + Constant(2.0)*inner(mu(alpha_curr)*D(u), D(v))*dx \
               - p*div(v)*dx \
               + div(u)*q*dx \
               + g*inner(rho(alpha_curr)*self.e2, v)*dx
            if(sigma > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(sigma)*mgrad(alpha_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(alpha_curr)*dx
        elif(len(kwargs) == 3):
            assert 'Re' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Re' not found (check function call)"
            assert 'Fr' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Fr' not found (check function call)"
            assert 'We' in kwargs, "Error in the parameters for non-dimensional version of NS: 'We' not found (check function call)"
            Re = kwargs.get('Re')
            Fr = kwargs.get('Fr')
            We = kwargs.get('We')
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_curr)*u - rho(alpha_old)*u_old, v)*dx \
               + inner(rho(alpha_curr)*dot(u_old, nabla_grad(u)), v)*dx \
               + Constant(2.0/Re)*inner(mu(alpha_curr)*D(u), D(v))*dx \
               - p*div(v)*dx \
               + div(u)*q*dx \
               + Constant(1.0/(Fr*Fr))*inner(rho(alpha_curr)*self.e2, v)*dx
            if(We > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(1.0/We)*mgrad(alpha_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(alpha_curr)*dx
        else:
            raise ValueError("Wrong number of arguments in Standard NS weak form setting (check function call)")

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = PETScMatrix()
        self.b2 = PETScVector()


    """Weak formulation for tentative velocity"""
    def ICT_weak_form_1(self, u, v, u_old, p_old, dt, rho, mu, alpha_curr, alpha_old, n_gamma = None, CDelta = None, **kwargs):
        #Check the correctness of type
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")
        if(not isinstance(p_old, Function)):
            raise ValueError("p_old must be an instance of Function")
        if(not isinstance(alpha_curr, Function)):
            raise ValueError("alpha_curr must be an instance of Function")
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
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_curr)*u - rho(alpha_old)*u_old, v)*dx \
               + inner(rho(alpha_curr)*dot(u_old, nabla_grad(u)), v)*dx \
               + Constant(2.0)*inner(mu(alpha_curr)*D(u), D(v))*dx \
               - p_old*div(v)*dx \
               + g*inner(rho(alpha_curr)*self.e2, v)*dx
            if(sigma > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(sigma)*mgrad(alpha_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(alpha_curr)*dx
        elif(len(kwargs) == 3):
            assert 'Re' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Re' not found (check function call)"
            assert 'Fr' in kwargs, "Error in the parameters for non-dimensional version of NS: 'Fr' not found (check function call)"
            assert 'We' in kwargs, "Error in the parameters for non-dimensional version of NS: 'We' not found (check function call)"
            Re = kwargs.get('Re')
            Fr = kwargs.get('Fr')
            We = kwargs.get('We')
            F2 = (Constant(1.0)/dt)*inner(rho(alpha_curr)*u - rho(alpha_old)*u_old, v)*dx \
               + inner(rho(alpha_curr)*dot(u_old, nabla_grad(u)), v)*dx \
               + Constant(2.0/Re)*inner(mu(alpha_curr)*D(u), D(v))*dx \
               - p_old*div(v)*dx \
               + Constant(1.0/(Fr*Fr))*inner(rho(alpha_curr)*self.e2, v)*dx
            if(We > DOLFIN_EPS):
                if(not callable(CDelta)):
                    raise ValueError("The function to compute the approximation of Dirac's delta must be a callable object")
                if(not isinstance(n_gamma, Function)):
                    raise ValueError("n(the unit normal to the interface) must be an instance of Function")
                F2 += Constant(1.0/We)*mgrad(alpha_curr)*inner((Identity(self.n_dim) - outer(n_gamma, n_gamma)), D(v))*CDelta(alpha_curr)*dx
        else:
            raise ValueError("Wrong number of arguments in ICT-Step 1 weak form setting (check function call)")

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = PETScMatrix()
        self.b2 = PETScVector()


    """Weak formulation for pressure correction"""
    def ICT_weak_form_2(self, p, q, dt, p_old, u_curr, rho, alpha_curr):
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
        self.a2_bis = (Constant(1.0)/rho(alpha_curr))*inner(grad(p), grad(q))*dx
        self.L2_bis = (Constant(1.0)/rho(alpha_curr))*inner(grad(p_old), grad(q))*dx - \
                      (Constant(1.0)/dt)*div(u_curr)*q*dx

        #Declare matrix and vector for the linear system solution
        self.A2_bis = PETScMatrix()
        self.b2_bis = PETScVector()


    """Weak formulation for velocity projection"""
    def ICT_weak_form_3(self, u, v, dt, u_curr, p_curr, p_old, rho, alpha_curr):
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
                       dt*inner(grad(p_curr - p_old), v)/rho(alpha_curr)*dx

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
    def SUPG(self, alpha, l, alpha_old, u_old, dt, mesh, scaling):
        #Extract cell diameter
        h = CellDiameter(mesh)

        #Compute the stabilization term
        r = ((alpha - alpha_old)/dt + inner(u_old, grad(alpha)))* \
            scaling*h/ufl.Max(2.0*sqrt(inner(u_old, u_old)),1.0e-3/h)*inner(u_old, grad(l))*dx
        return r


    """Volume fraction advection weak formulation"""
    def VF_weak_form(self, alpha, l, alpha_old, u_old, dt, mesh, method, param = None):
        #Check availability of the method before proceding
        assert method in self.stab_dict, "Stabilization method(" + method + ") not available"

        #Check the correctness of type
        if(not isinstance(alpha_old, Function)):
            raise ValueError("alpha_old must be an instance of Function")
        if(not isinstance(u_old, Function)):
            raise ValueError("u_old must be an instance of Function")

        #Save the dimension of the problem
        self.n_dim = mesh.geometry().dim()

        #Declare weak formulation
        F1 = ((alpha - alpha_old)/dt + inner(u_old, grad(alpha)))*l*dx

        #Add stabilization term (if specified)
        if(method == 'SUPG'):
            #Check whether Reynolds number is really available
            assert param is not None, \
            "Stabilization parameter not available in order to use SUPG stabilization (check the call of the function)"

            #Add the stabilization term
            F1 += self.SUPG(alpha, l, alpha_old, u_old, dt, mesh, param)
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


    """Build and solve the system for volume fraction transport"""
    def solve_VolumeFraction_system(self, alpha_curr):
        #Assemble matrix and right-hand side
        assemble(self.a1, tensor = self.A1)
        assemble(self.L1, tensor = self.b1)

        #Solve the level-set system
        solve(self.A1, alpha_curr.vector(), self.b1, self.solver_VF, self.precon_VF)


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
