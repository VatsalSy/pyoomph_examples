# A comparison with oomph-lib's example case
# https://oomph-lib.github.io/oomph-lib/doc/multi_physics/b_convection/html/index.html
# oomph-lib driver code: https://oomph-lib.github.io/oomph-lib/demo_drivers/multi_physics/boussinesq_convection/
# However, we set the number of elements to 150x150

from pyoomph import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.advection_diffusion import *

class BoussinesqProblem(Problem):
    def __init__(self):
        super().__init__()
        self.lx=3.0
        self.ly=1.0
        self.Nx=150
        self.Ny=150
        self.Peclet=1
        self.Inverse_Prandtl=1.0
        self.Rayleigh = 1800.0
        self.g=vector(0,-1)
        self.max_refinement_level=0

    def define_problem(self):
        self+=RectangularQuadMesh(size=[self.lx,self.ly],N=[self.Nx,self.Ny])
        eqs=MeshFileOutput()
        eqs+=NavierStokesEquations(mode="CR",bulkforce=-self.g*var("T")*self.Rayleigh,mass_density=self.Inverse_Prandtl).with_pressure_fixation()
        eqs+=AdvectionDiffusionEquations("T",diffusivity=1/self.Peclet)
        eqs+=DirichletBC(velocity_x=0)@["left","right"]
        eqs+=DirichletBC(velocity_x=0,velocity_y=0)@"bottom"
        
        
        epsilon=0.01
        uypert = sin(2.0*pi*var("coordinate_x")/3.0)*epsilon*var("time")*exp(-var("time"))
        eqs+=DirichletBC(velocity_x=0,velocity_y=uypert)@"top"
        
        eqs+=DirichletBC(T=0.5)@"bottom"
        eqs+=DirichletBC(T=-0.5)@"top"
        self+=eqs@"domain"
        
with BoussinesqProblem() as problem:
    problem.set_c_compiler("system").optimize_for_max_speed() 
    problem.initialise()
    dt=0.1
    problem.run(endtime=200*dt,outstep=dt)
