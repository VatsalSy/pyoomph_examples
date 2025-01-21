#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================



# A comparison with oomph-lib's example case
# https://oomph-lib.github.io/oomph-lib/doc/navier_stokes/single_layer_free_surface/html/index.html
# oomph-lib driver code: https://oomph-lib.github.io/oomph-lib/demo_drivers/navier_stokes/single_layer_free_surface/
# For assembly performance testing, increase the number of elements (N) to e.g. 300 in both implementations

from pyoomph import *  # Load pyoomph
from pyoomph.equations.navier_stokes import * # Import the Navier-Stokes equation
from pyoomph.equations.ALE import *  # same for moving mesh equations


# Problem class for a deformed Navier-Stokes interface relaxing to flat equilibrium
class SingleLayerProblem(Problem):
    def __init__(self):
        super().__init__()
        # Default parameters
        self.Re = 5  # Rayleigh number
        self.Ca = 0.01  # Capillary number
        self.Nu = 0.1  # Parameter for the pseudo-elastic mesh
        self.G = vector(0, -1)  # Gravity
        self.epsilon = 0.1  # initial deformation
        self.N = 12  # number of nodes per direction. 
        # self.N = 300 # Increase this in both oomph-lib and here to assess the assembly performance

    def define_problem(self):
        # Add a rectangular mesh
        self += RectangularQuadMesh(N=[self.N, self.N], size=[1, 1])

        # Equation system: Paraview output, Navier-Stokes and mesh motion
        eqs = MeshFileOutput()
        eqs += NavierStokesEquations(dynamic_viscosity=1,
                                     mass_density=self.Re, gravity=self.G, mode="CR")
        eqs += PseudoElasticMesh(nu=self.Nu)
        # set lagragian=Eulerian each time step
        eqs += SetLagrangianToEulerianAfterSolve()

        # Periodic BC in x
        eqs += PeriodicBC("right",offset=[1,0])@"left"

        # Initial condition: Deform the mesh, don't degrade time stepping to first order in first step
        X, Y = var(["lagrangian_x", "lagrangian_y"])
        eqs += InitialCondition(mesh_y=Y+(1-absolute(1-Y))
                                * self.epsilon*cos(2*pi*X), degraded_start=False)
        eqs += InitialCondition(velocity_x=0,
                                velocity_y=0, degraded_start=False)

        # Dirichlet conditions, fixed mesh positions and velocity conditions
        # fix the entire x-coordinate of the mesh
        eqs += DirichletBC(mesh_x=True)
        eqs += DirichletBC(velocity_x=0)@"left" # no outflow left and right
        eqs += DirichletBC(velocity_x=0)@"right" # and a no-slip with a fixed y-coordinate at the bottom
        eqs += DirichletBC(velocity_x=0, velocity_y=0, mesh_y=0)@"bottom"

        # Free surface
        eqs += NavierStokesFreeSurface(surface_tension=1/self.Ca)@"top"

        # Output of the left corner height
        eqs += (IntegralObservables(h=var("mesh_y")) +
                IntegralObservableOutput())@"top/left"

        # Add the system to the problem
        self += eqs@"domain"



with SingleLayerProblem() as problem:
    # This corresponds to -O3 --ffast-math (leave optimize_for_max_speed out for -O2 without fast-math)
    # Compile oomph-lib with the same flags for a fair comparison
    problem.set_c_compiler("system").optimize_for_max_speed()
    # Since oomph-lib's case is also without spatial adaptivity, we boost our code to remove if-statements for adaptivity here as well 
    problem+=EquationCompilationFlags(with_adaptivity=False)@["domain","domain/top"]
    problem.max_refinement_level=0
    
    problem.run(0.6,0.005)
	
