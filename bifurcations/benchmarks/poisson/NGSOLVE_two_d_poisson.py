
#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2024  Christian Diddens & Duarte Rocha
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
# This code is using NGSolve (https://ngsolve.org/)

from ngsolve import *
from netgen.geom2d import unit_square

ngsglobals.msg_level = 1

# generate a triangular mesh of mesh-size 0.2
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
import ngsolve.meshes as ngm
mesh = ngm.MakeStructured2DMesh(nx=1000,ny=1000,secondorder=True)

# H1-conforming finite element space
fes = H1(mesh, order=2, dirichlet=[1,2,3,4])

# define trial- and test-functions
u = fes.TrialFunction()
v = fes.TestFunction()

# the right hand side
f = LinearForm(fes)
#f = CF(0)
#f += 32 * (y*(1-y)+x*(1-x)) * v * dx
Alpha=Parameter(1) # TODO: MAke param
TanPhi=1
tanh=lambda x : sinh(x)/cosh(x)
f += (2*tanh(-1+Alpha*(TanPhi*x-y))*(1-(tanh(-1.0+Alpha*(TanPhi*x-y)))**2)*Alpha*Alpha*TanPhi*TanPhi+2*tanh(-1+Alpha*(TanPhi*x-y))*(1-(tanh(-1+Alpha*(TanPhi*x-y)))**2)*Alpha*Alpha)*v*dx

# the bilinear-form 
a = BilinearForm(fes, symmetric=False)
a += grad(u)*grad(v)*dx

import time
t0=time.time()
a.Assemble()
R=f.Assemble()
t1=time.time()
print(t1-t0,len(R.vec),fes.ndof)

# the solution field 
gfu = GridFunction(fes)
#gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
# print (u.vec)


# plot the solution (netgen-gui only)
#Draw (gfu)
#Draw (-grad(gfu), mesh, "Flux")

