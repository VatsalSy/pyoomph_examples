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


# A comparison with oomph-lib's 2d Poisson equation case
# https://oomph-lib.github.io/oomph-lib/doc/poisson/two_d_poisson/html/
# oomph-lib driver code: https://oomph-lib.github.io/oomph-lib/demo_drivers/poisson/two_d_poisson/
# However, we use 1000x1000 elements

from pyoomph import *  # Load pyoomph and the Poisson equation
from pyoomph.equations.poisson import *


# Problem class for a deformed Navier-Stokes interface relaxing to flat equilibrium
class PoissonProblem(Problem):
    def __init__(self):
        super().__init__()
        self.N = 1000  # number of nodes per direction
        Alpha=self.define_global_parameter(Alpha=1)
        TanPhi=1
        x=var("coordinate")
        self.source=  2*tanh(-1+Alpha*(TanPhi*x[0]-x[1]))*(1-(tanh(-1.0+Alpha*(TanPhi*x[0]-x[1])))**2)*Alpha*Alpha*TanPhi*TanPhi+2*tanh(-1+Alpha*(TanPhi*x[0]-x[1]))*(1-(tanh(-1+Alpha*(TanPhi*x[0]-x[1])))**2)*Alpha*Alpha
        self.source=subexpression(self.source)

    def define_problem(self):
        # Add a rectangular mesh
        self += RectangularQuadMesh(N=[self.N, self.N], size=[1, 2])

        # Equation system:
        eqs = MeshFileOutput()
        eqs += PoissonEquation(source=self.source)
        eqs += DirichletBC(u=0)@["top","left","right","bottom"]        
        # Add the system to the problem
        self += eqs@"domain"

import time
with PoissonProblem() as problem:

    # Compile with maximum performance
    problem.set_c_compiler("system").optimize_for_max_speed()
    
    # Add analytical_position_jacobian=False,analytical_jacobian=False for pure FD
    problem+=EquationCompilationFlags(with_adaptivity=False)@["domain"]
    #problem+=SpatialIntegrationOrder(1)@"domain"
    problem.max_refinement_level=0
    problem.initialise()
    t0=time.time()
    problem.assemble_jacobian() # will also assemble the residuals by default
    t1=time.time()
    print(t1-t0)
    
