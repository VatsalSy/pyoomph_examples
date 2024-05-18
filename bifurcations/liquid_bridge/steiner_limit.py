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



from pyoomph import *
from pyoomph.expressions import *
from pyoomph.utils.num_text_out import NumericalTextOutputFile

class YoungLaplaceEquation(Equations):
    def __init__(self,p):
        super().__init__()
        self.p=p # Store the liquid pressure
        
    def define_fields(self):
        # Moving mesh
        self.activate_coordinates_as_dofs("C2")
        # Projected normal
        self.define_vector_field("proj_n","C2")
        # Curvature
        self.define_scalar_field("kappa","C2")
        # Normalized arclength for tangnetial nodal placements
        self.define_scalar_field("normalized_s","C2")
        
    def define_residuals(self):
        # Project the element normal to a continuous normal field
        pn,pntest=var_and_test("proj_n")
        n=var("normal")
        self.add_weak(pn-n,pntest)
        # We need initial conditions, otherwise the following normalization will initially fail
        self.set_initial_condition("proj_n_x",-1)
        self.set_initial_condition("proj_n_y",0)
        pn=pn/square_root(dot(pn,pn))
        # The curvature is the divergence of the projected normal
        kappa,kappatest=var_and_test("kappa")
        self.add_weak(kappa+div(pn),kappatest)
        # Young-Laplace equation by normally shifting the nodes
        self.add_weak(kappa-self.p,-dot(pn,testfunction("mesh")))
        # Tangentially, we shift the nodes so that the position with respect to the normalized arclength is preserved
        s,stest=var_and_test("normalized_s")                
        self.add_weak(grad(s)/var("coordinate_x"),grad(stest),coordinate_system=cartesian)
        t=vector(-pn[1],pn[0])
        self.add_weak(s-var("lagrangian_x") , dot(t,testfunction("mesh")))
        
        
        
        
class LiquidBridgeProblem(Problem):
    def __init__(self):
        super(LiquidBridgeProblem, self).__init__()
        # Add the global parameters, namely the length and the normalized volume
        self.L,self.Vhat=self.define_global_parameter(L=1,Vhat=1)
                
    def define_problem(self):
        # Axisymmetric coordinate system. Only axisymmetric contributions will be expanded for azimuthal stability analysis
        self.set_coordinate_system("axisymmetric")
        
        # A line mesh embedded in a 2d domain. Will be rotated by the InitialCondition
        self+=LineMesh(N=60,size=1,nodal_dimension=2,name="interface",left_name="bottom",right_name="top")       
                
        # Adjust liquid pressure by the volume constraint
        Vdest=(pi*self.L)*(self.Vhat-1/3)        
        p,ptest=self.add_global_dof("p",equation_contribution=-Vdest,initial_condition=1)
                
        eqs=YoungLaplaceEquation(p)
        # Initial condition: Rotate the mesh to range from (1,0) to (1,L), and set the curvature to 1 and initialize the arclength
        eqs+=InitialCondition(mesh_x=1,mesh_y=var("lagrangian_x")*self.L,kappa=1,normalized_s=var("lagrangian_x"))
        # Output
        eqs+=TextFileOutput()
        
        # Boundary conditions: Fix the contact line, the bottom and set the top height to L. Also fix the normalized arclength
        eqs+=DirichletBC(mesh_x=1)@["top","bottom"]                
        eqs+=DirichletBC(normalized_s=0,mesh_y=0)@"bottom"
        eqs+=(DirichletBC(normalized_s=1)+EnforcedBC(mesh_y=self.L-var("mesh_y")))@"top"
        
        # Volume calculation
        eqs+=WeakContribution(-1/3*dot(var("coordinate"),var("normal")),ptest)
            
        
        # Add some temporal dynamics for the mass matrix
        eqs+=WeakContribution(-partial_t(var("mesh")),testfunction("mesh"))
        
        
        self+=eqs@"interface"
        

with LiquidBridgeProblem() as problem:
    # The generated code is quite large (>1 MB), takes some time for compilation
    # Also, GiNaC does not always give the same term order. Normally, it is not an issue, but here it can be
    problem.set_c_compiler("system") # .optimize_for_max_speed() # So we do not use ffast-math
    #problem.set_c_compiler("tcc") # Use tcc for faster compilation, but slower execution
    
    # Activate azimuthal stability analysis
    problem.setup_for_stability_analysis(azimuthal_stability=True)
    
    problem+=TextFileOutput()@"interface/bottom"
    
    # Solve and go close to the bifurcation point
    problem.solve()        
    problem.go_to_param(Vhat=1.86)    
    
    # All eigensolvers fail quite badly for this problem, so we use scipy, which at least gives any starting guess
    problem.set_eigensolver("scipy")        
    problem.solve_eigenproblem(azimuthal_m=1,n=6)
    # Jump on the bifurcation
    problem.activate_bifurcation_tracking("Vhat","azimuthal",azimuthal_mode=1)
    problem.solve(max_newton_iterations=40)
    
    # Scan the curve
    outfile=NumericalTextOutputFile(problem.get_output_directory("steiner.txt"),header=["L","Vhat","p"])
    dL=0.01
    outfile.add_row(problem.L,problem.Vhat,problem.get_ode("globals").get_value("p"))
    problem.set_arc_length_parameter(Desired_newton_iterations_ds=8)
    while problem.L.value<3:
        dL=problem.arclength_continuation("L",dL)        
        problem.output()
        outfile.add_row(problem.L,problem.Vhat,problem.get_ode("globals").get_value("p"))        
        # Reset bifurcation tracking to update the normalization vector
        #problem.reset_arc_length_parameters()
        #problem.deactivate_bifurcation_tracking()
        #problem.activate_bifurcation_tracking("Vhat","azimuthal",azimuthal_mode=1)
    
