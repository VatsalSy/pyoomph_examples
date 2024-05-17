# A comparison with oomph-lib's 2d Poisson equation case
# https://oomph-lib.github.io/oomph-lib/doc/poisson/two_d_poisson/html/
# oomph-lib driver code: https://oomph-lib.github.io/oomph-lib/demo_drivers/poisson/two_d_poisson/
# However, we use 1000x1000 elements
# This code is using FEniCS (https://fenicsproject.org/)

from dolfin import *


N=1000
mesh=RectangleMesh.create([Point(0,0),Point(1,2)],[N,N],CellType.Type.quadrilateral) 
V=FunctionSpace(mesh,"Lagrange",2)

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS or x[1] > 2.0 - DOLFIN_EPS

u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

u = TrialFunction(V)
v = TestFunction(V)
source = Expression("2.0*tanh(-1.0+Alpha*(1*x[0]-x[1]))*(1.0-pow(tanh(-1.0+Alpha*(1*x[0]-x[1])),2.0))*Alpha*Alpha*1*1+2.0*tanh(-1.0+Alpha*(1*x[0]-x[1]))*(1.0-pow(tanh(-1.0+Alpha*(1*x[0]-x[1])),2.0))*Alpha*Alpha",Alpha=Constant(1.0),element=V.ufl_element())
a = inner(grad(u), grad(v))*dx
L = source*v*dx 

parameters["form_compiler"]["cpp_optimize_flags"]="-O3 -ffast-math"

# Assemble once (might invoke just-in-time compilation)
import time
t0=time.time()
R=assemble(L)
assemble(a)
t1=time.time()
print(t1-t0,len(R))

# Assemble a second time to measure the pure assembly speed
t0=time.time()
R=assemble(L)
assemble(a)
t1=time.time()
print(t1-t0,len(R))

#u=Function(V)
#f=XDMFFile("fenics.xdmf")
#solve(a==L,u,bcs=bc)
#f.write(u)
#plot(u)

