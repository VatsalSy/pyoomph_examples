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



from pyoomph import *
from pyoomph.expressions import *
from pyoomph.utils.num_text_out import NumericalTextOutputFile
from pyoomph.equations.ALE import *
from pyoomph.equations.navier_stokes import *
from pyoomph.meshes.remesher import Remesher2d        
        
class LiquidBridgeProblem(Problem):
    def __init__(self):
        super(LiquidBridgeProblem, self).__init__()
        # Add the global parameters, namely the length and the normalized volume
        self.L,self.Vhat=self.define_global_parameter(L=1,Vhat=1)
                
    def define_problem(self):
        # Axisymmetric coordinate system. Only axisymmetric contributions will be expanded for azimuthal stability analysis
        self.set_coordinate_system("axisymmetric")        
        # Attach the mesh with a remesher
        mesh=RectangularQuadMesh(size=[1,self.L],N=[10,10]) # Rather coarse mesh here
        mesh.remesher=Remesher2d(mesh)
        self+=mesh                
        
        # Bulk equations: Just Stokes flow on a Laplace-smoothed mesh
        eqs=StokesEquations()
        eqs+=LaplaceSmoothedMesh()    
                                        
        # Top and bottom: No-slip, fixed x-coordinate and y=0 (bottom) and enforce y=L (top)
        eqs+=(NoSlipBC()+DirichletBC(mesh_x=True)+EnforcedDirichlet(mesh_y=self.L))@"top" 
        eqs+=(NoSlipBC()+DirichletBC(mesh_y=0,mesh_x=True))@"bottom"        
        
        # Side boundary conditions
        eqs+=AxisymmetryBC()@"left"
        eqs+=NavierStokesFreeSurface(surface_tension=1)@"right"        
        
        # Adjust liquid pressure by the volume constraint
        Vdest=pi*self.L*self.Vhat
        P,Ptest=self.add_global_dof("P",equation_contribution=-Vdest,initial_condition=1)            
        eqs+=WeakContribution(1,Ptest)
        eqs+=AverageConstraint(_kin_bc=P)@"right"+ DirichletBC(_kin_bc=0)@"right/top"
        
        eqs+=IntegralObservables(Pint=var("pressure"),Pavg=lambda Pint: Pint/Vdest)
        
        
        eqs+=RemeshWhen(RemeshingOptions())
        
                
        self+=eqs@"domain"
        

with LiquidBridgeProblem() as problem:
    from pyoomph.solvers.petsc import *
    problem.set_eigensolver("slepc").use_mumps()    
    problem.set_c_compiler("system").optimize_for_max_speed() 
    
    problem.setup_for_stability_analysis(azimuthal_stability=True)
    
    from pyoomph.meshes.meshdatacache import MeshDataCombineWithEigenfunction    
    problem+=MeshFileOutput(operator=MeshDataCombineWithEigenfunction(0))@"domain"
    
    # Solve and go close to the bifurcation point
    
    problem.Vhat.value=1.05
    problem.solve()     
    problem.go_to_param(Vhat=1.5)        
    problem.force_remesh()    
    problem.go_to_param(Vhat=1.86)    
    problem.force_remesh()    
    problem.solve()
            
    problem.solve_eigenproblem(azimuthal_m=1,n=6,report_accuracy=True) # Get any guess
    # Switch on bifurcation tracking
    problem.activate_bifurcation_tracking("Vhat","azimuthal",azimuthal_mode=1)
    problem.solve()        
    # Scan the curve
    outfile=NumericalTextOutputFile(problem.get_output_directory("steiner.txt"),header=["L","Vhat","p"])
    outfile.add_row(problem.L,problem.Vhat,problem.get_mesh("domain").evaluate_observable("Pavg"))                
    problem.output_at_increased_time()
    dL=0.01    
    while problem.L.value<5:
        dL=problem.arclength_continuation("L",dL)         
        problem.remesh_handler_during_continuation(resolve_max_newton_steps=50,resolve_globally_convergent_newton=True)
        outfile.add_row(problem.L,problem.Vhat,problem.get_mesh("domain").evaluate_observable("Pavg"))                
        problem.output_at_increased_time()        
    
