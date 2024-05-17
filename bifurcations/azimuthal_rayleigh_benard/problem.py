from pyoomph import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.advection_diffusion import *

# Class for the problem definition
class RBCylindrical(Problem):
    def __init__(self):
        super ().__init__ ()
        # Aspect ratio, Rayleigh and Prandtl number with defaults
        self.Gamma=self.define_global_parameter(Gamma = 1)
        self.Ra, self.Pr=self.define_global_parameter(Ra = 1, Pr = 1)

    def define_problem(self):
        # Axisymmetric coordinate system
        self.set_coordinate_system(axisymmetric) 

        # Scale radial coordinate with aspect ratio parameter
        self.set_scaling(coordinate_x=self.Gamma)

        # Axisymmetric cross-section as mesh
        # We use R=1 and H=1, but due to the radial scaling, 
        # we can modify the effective radius
        self += RectangularQuadMesh(size = [1, 1], N = 20)

        RaPr = self.Ra * self.Pr # Shortcut for Ra*Pr
        # Equations: Navier-Stokes. We scale the pressure also with 
        # RaPr, so that the hydrostatic pressure due to the bulk-force 
        # is independent on the value of Ra*Pr
        NS = NavierStokesEquations(mass_density = 1, dynamic_viscosity = self.Pr, bulkforce = RaPr*var("T") * vector(0,1), pressure_factor = RaPr)
        # Since u*n is set at all walls, we have a nullspace in the pressure.
        # This offset is fixed by e.g. an integral constraint <p>=0.
        eqs=NS.with_pressure_integral_constraint(self, integral_value = 0, set_zero_on_angular_eigensolve = True)
        # And advection-diffusion for temperature
        eqs += AdvectionDiffusionEquations(fieldnames = "T", diffusivity = 1, space = "C1")

        # Boundary conditions
        eqs += DirichletBC(T = 1)@"bottom"
        eqs += DirichletBC(T = 0)@"top"
        # The NoSlipBC will actually also set velocity_phi=0 automatically
        eqs += NoSlipBC()@["top", "right", "bottom"]
        # Here, the magic happens regarding the m-dependent boundary conditions
        eqs += AxisymmetryBC()@"left"

        # Output
        eqs += MeshFileOutput()

        # Add the system to the problem
        self += eqs@"domain"
