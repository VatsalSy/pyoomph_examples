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
# This code is using nutils (https://nutils.org/)

from nutils import mesh, function, solver, export, testing,function


def main(nelems: int = 1000):
    topo, x = mesh.unitsquare(nelems, etype='square')
    u = function.dotarg('u', topo.basis('std', degree=1))
    g = u.grad(x)
    J = function.J(x)

    sqr = topo.boundary.integral(u**2 * J, degree=2)
    cons = solver.optimize(('u',), sqr, droptol=1e-12)
    
    def get_f(xv):
       Alpha=1
       TanPhi=1
       tanh=function.tanh
       x=xv[0]
       y=xv[1]
       return (2*tanh(-1+Alpha*(TanPhi*x-y))*(1-(tanh(-1.0+Alpha*(TanPhi*x-y)))**2)*Alpha*Alpha*TanPhi*TanPhi+2*tanh(-1+Alpha*(TanPhi*x-y))*(1-(tanh(-1+Alpha*(TanPhi*x-y)))**2)*Alpha*Alpha)
    f=get_f(x)
    energy = topo.integral((g @ g / 2 - u*f) * J, degree=2)
    args = solver.optimize(('u',), energy, constrain=cons)

    bezier = topo.sample('bezier', 3)
    x, u = bezier.eval([x, u], **args)
    export.triplot('u.png', x, u, tri=bezier.tri, cmap='jet')

    return args


if __name__ == '__main__':
    from nutils import cli
    cli.run(main)



