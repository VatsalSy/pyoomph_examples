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

# Import problem and 3d eigen output
from problem import *
from pyoomph.meshes.meshdatacache import MeshDataCombineWithEigenfunction,MeshDataRotationalExtrusion

with DropletProblem() as problem:
    # Automatically derive the azimuthal stability eigensystem from the weak forms. Also calculate the second order derivatives for bifurcation tracking
    problem.setup_for_stability_analysis(azimuthal_stability=True,analytic_hessian=True)   

    # Take parameters close to criticality from the paper
    problem.contact_angle=15*degree
    MaC=122   
    mc=1

    # Add 3d output with critical eigenfunction
    out_operator=MeshDataCombineWithEigenfunction(0)+MeshDataRotationalExtrusion(100)
    problem+=MeshFileOutput(filetrunk="eigenmerge",operator=out_operator)@"droplet"

    # Start with a small Marangoni number and make a stationary solve
    problem.set_MaH(10)
    problem.solve()
    # Increase it more
    problem.set_MaH(20)
    problem.solve()
    # Use the critical Marangoni number from the paper
    problem.set_MaH(MaC)
    # Transient solve with adaptive time steps to converge towards a stable stationary solution
    problem.run(1000,startstep=0.001,temporal_error=1,outstep=False,do_not_set_IC=True,out_initially=False)
    # Solve for the stationary solution
    problem.solve()
    # Solve the azimuthal eigenproblem to get a guess for the eigenvector
    problem.solve_eigenproblem(50,azimuthal_m=mc)
    # Jump on the bifurcation by adjusting Re
    problem.activate_bifurcation_tracking("Re")
    problem.solve()
    # Full output
    problem.output_at_increased_time()
    print("MaC/Imag_Eigen",problem.get_MaH(),numpy.imag(problem.get_last_eigenvalues()[0]))
