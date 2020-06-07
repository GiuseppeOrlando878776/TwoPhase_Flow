from dolfin import *

from Auxiliary_Functions import *

class TwoPhaseFlows:
    """Default constructor"""
    def __init__(self):
        #Define an auxiliary dictionary to set proper stabilization
        self.stab_dict = {'IP', 'SUPG', 'None'}

        #Declare useful constant vector
        self.e1 = Constant((1.0, 0.0))
        self.e2 = Constant((0.0, 1.0))


    """Weak formulation for Navier-Stokes"""
    def NS_weak_form(self, u, p, v, q, u_old, dt, rho, mu, phi_curr, phi_old, eps, g, sigma, n, CDelta = None):
        F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
           + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
           + 2.0*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
           - p*div(v)*dx \
           + div(u)*q*dx \
           + g*inner(rho(phi_curr, eps)*self.e2, v)*dx \
           + sigma*mgrad(phi_curr)*inner((Identity(2) - outer(n, n)), D(v))*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()


    """Weak formulation for tentative velocity"""
    def ICT_weak_form_1(self, u, v, u_old, dt, rho, mu, phi_curr, phi_old, eps, g, sigma, n, CDelta = None):
        F2 = (1.0/dt)*inner(rho(phi_curr, eps)*u - rho(phi_old, eps)*u_old, v)*dx \
           + inner(rho(phi_curr, eps)*dot(u_old, nabla_grad(u)), v)*dx \
           + 2.0*inner(mu(phi_curr, eps)*D(u), D(v))*dx \
           - p_old*div(v)*dx \
           + g*inner(rho(phi_curr, eps)*self.e2, v)*dx \
           + sigma*mgrad(phi_curr)*inner((Identity(2) - outer(n, n)), D(v))*dx

        #Save corresponding weak form and declare suitable matrix and vector
        self.a2 = lhs(F2)
        self.L2 = rhs(F2)

        self.A2 = Matrix()
        self.b2 = Vector()


    """Weak formulation for pressure correction"""
    def ICT_weak_form_2(self, p, q, dt, u_curr, rho, phi_curr, eps):
        #Define variational problem for step 2
        self.a2_bis = (1.0/rho(phi_curr, eps))*inner(grad(p), grad(q))*dx
        self.L2_bis = (1.0/rho(phi_curr, eps))*inner(grad(p_old), grad(q))*dx - \
                      (1.0/dt)*div(u_curr)*q*dx

        self.A2_bis = Matrix()
        self.b2_bis = Vector()


    """Weak formulation for velocity projection"""
    def ICT_weak_form_3(self, u, v, dt, u_curr, p_curr, p_old, rho, phi_curr, eps):
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
        #Check availability of the method
        assert method in self.stab_dict, "Stabilization method not available"

        #Declare weak formulation
        F1 = ((phi - phi_old)/dt + inner(u_old, grad(phi)))*l*dx 

        #Add stabilization term (if specified)
        if(method == 'SUPG'):
            #Check whether Reynolds number is available
            assert param is not None, \
            "Reynolds number not available in order to use SUPG stabilization"

            #Add the stabilization term
            F1 += self.SUPG(phi, l, phi_old, u_old, dt, mesh, param)
        elif(method == 'IP'):
            #Check whether stabilization parameter is available
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
        self.a1_reinit = phi/dt_reinit*l*dx
        self.L1_reinit = phi0/dt_reinit*l*dx + \
                         signp(phi_curr, gamma_reinit)*(1.0 - mgrad(phi0))*l*dx -\
                         beta_reinit*inner(grad(phi0), grad(l))*dx

        #Save the matrix that will not change and declare vector
        self.A1_reinit = assemble(self.a1_reinit)
        self.b1_reinit = Vector()


    """Weak form non-conservative reinitialization (elliptic version)"""
    def NCLSM_elliptic_weak_form(self, phi, l, phi0, phi_curr, dt_reinit, CDelta, eps_reinit, beta_reinit = 1.0e3):
        self.a1_reinit = inner(grad(phi), grad(l))*dx + \
                         beta_reinit*phi*CDelta(phi_curr, eps_reinit)*dx
        self.L1_reinit = inner(grad(phi0)/mgrad(phi0), grad(l))*dx

        #Declare matrix and vector for solution
        self.A1_reinit = Matrix()
        self.b1_reinit = Vector()


    """Weak form conservative reinitialization"""
    def CLSM_weak_form(self, phi_intermediate, l, phi0, n, dt_reinit, eps_reinit):
        #Save variational formulation
        self.F1_reinit = (phi_intermediate - phi0)/dt_reinit*l*dx \
                       - phi_intermediate*(1.0 - phi_intermediate)*inner(grad(l), n)*dx \
                       + eps_reinit*inner(grad(phi_intermediate), n)*inner(grad(l), n)*dx
