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
from pyoomph.utils.num_text_out import NumericalTextOutputFile

# Generic base class for both full bulk and Young-Laplace implementation
class DropletDetachmentProblem(Problem):
    def __init__(self):
        super().__init__()
        # define parameters with initial values
        self.Bo, self.R = self.define_global_parameter(Bo=0, R=1)
        self.fold_out = None  # output file, will be create on the first output

    # Override output to also write the critical curve
    def output(self, stage="", quiet=None) -> None:
        # Create the file initially
        if self.fold_out is None:
            filename = self.get_output_directory("critical_curve.txt")
            self.fold_out = NumericalTextOutputFile(filename, header=["R", "Bo"])
        # Write the current data file
        self.fold_out.add_row(self.R.value, self.Bo.value)
        return super().output(stage, quiet)  # call all other output


### SKIP ##
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.utils.dropgeom import DropletGeometry

class DropletMesh(GmshTemplate):
    def __init__(self, loaded_from_mesh_file: str | None = None):
        super().__init__(loaded_from_mesh_file)
        self.remesher = Remesher2d(self)
        self.mesh_mode = "tris"
        self.default_resolution = 0.035

    def define_geometry(self):
        pr = self.get_problem()
        geom = DropletGeometry(volume=1, base_radius=pr.R.value)
        p00 = self.point(0, 0)
        p0h = self.point(0, -geom.apex_height)
        pR0 = self.point(pr.R.value, 0)
        self.circle_arc(
            pR0, p0h, through_point=(-pr.R.value, 0), name="interface")
        self.create_lines(p0h, "axis", p00, "wall", pR0)
        self.plane_surface("axis", "wall", "interface", name="liquid")
###

from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.navier_stokes import *

# Droplet detachment with the full bulk system
class DropletDetachmentWithBulk(DropletDetachmentProblem):
    def define_problem(self):
        # Mesh and axisymmetric coordinate system
        self += DropletMesh()  # add the mesh (class skipped for brevity)
        self.set_coordinate_system("axisymmetric")

        # Assemble equation system:
        # Stokes equation with gravity
        bulkforce=self.Bo*vector(0, -1)
        eqs = StokesEquations(dynamic_viscosity=1,bulkforce=bulkforce)
        eqs += LaplaceSmoothedMesh()  # add Laplace smoothed mesh
        eqs += RemeshWhen(RemeshingOptions(max_expansion=1.7,
                          min_expansion=0.6))  # control remeshing

        # Boundary conditions
        eqs += AxisymmetryBC()@"axis"  # axis of symmetry
        # free surface
        eqs += NavierStokesFreeSurface(surface_tension=1,
                                       kinbc_name="kin_bc")@"interface"
        # Wall at the top, fixed y coordinate, no velocity
        eqs += DirichletBC(mesh_y=0, velocity_x=0, velocity_y=0)@"wall"
        # impose r=R at the contact line
        cl_condition = var("mesh_x")-self.R
        # Adjust the kinematic boundary condition so that mesh_x=R holds
        eqs += EnforcedBC(kin_bc=cl_condition)@"interface/wall"

        # Global Lagrange multiplier to solve V=integral 1*dx == V0
        # We define it, subtract V0=1 from the equation and add it to the globals
        self += GlobalLagrangeMultiplier(p_ref=-1)@"globals"
        p_ref = var("p_ref", domain="globals")  # Bind it
        # Then we add the volume integral to the equation
        eqs += WeakContribution(dot(var("normal"), var("mesh"))/3, testfunction(p_ref))@"interface"
        # Feed back to the average pressure, 
        eqs += WeakContribution(p_ref,testfunction("pressure"))

        # Finally: Add the equation system to the problemn
        self += eqs@"liquid"


class YoungLaplaceEquation(Equations):
    def __init__(self, p_ref, additional_pressure):
        super(YoungLaplaceEquation, self).__init__()
        self.p_ref = p_ref  # reference pressure
        self.additional_pressure = additional_pressure  # e.g. gravity

    def define_fields(self):
        # Projected normal to smooth the normal
        self.define_vector_field("projected_normal", "C2")
        self.define_scalar_field("curvature", "C2")  # curvature
        # arclength for tangential placing
        self.define_scalar_field("normalized_arclength", "C2")
        # moving mesh, i.e. mesh_x and mesh_y are unknowns
        self.activate_coordinates_as_dofs()

    def define_residuals(self):
        n_elem = var("normal")
        n_proj, n_proj_test = var_and_test("projected_normal")

        # project the normal, smoothed across the elements
        self.add_weak(n_proj-n_elem,n_proj_test, coordinate_system=cartesian)
        # The projected normal is not necessarily normalized, do it now
        norm = n_proj / square_root(dot(n_proj, n_proj))

        # project the curvature from curv=-div(n_p)
        curv, curv_test = var_and_test("curvature")
        self.add_weak(curv+div(norm), curv_test)

        # In normal direction of the mesh position, the YL-equation is solved
        xtest = testfunction("mesh")
        self.add_weak(curv+self.additional_pressure -self.p_ref, dot(n_elem, xtest))

        # We solve the normalized arclength by a Laplace equation along the curve
        s, stest = var_and_test("normalized_arclength")
        self.add_weak(grad(s, coordsys=cartesian), grad(
            stest, coordsys=cartesian), coordinate_system=cartesian)

        # And we shift the nodes tangentially to maintain them equidistant
        # The desired location is the initial arclength, i.e. Lagrangian coord.
        sdest = var("lagrangian_x")
        tangent = vector(n_elem[1], -n_elem[0])
        # move the nodes tangentially so that they are equidistant
        self.add_weak(s-sdest, dot(xtest, tangent))


class DropletDetachmentWithYoungLaplace(DropletDetachmentProblem):
    def define_problem(self):
        # Axisymmetric coordinate system. Make a line mesh and bend it to a circle segment
        self.set_coordinate_system("axisymmetric")
        self += LineMesh(N=500, nodal_dimension=2, size=1, name="interface")

        # Add global dof for the reference pressure. Add +1 to the equation
        self += GlobalLagrangeMultiplier(p_ref=1)@"globals"
        p_ref = var("p_ref", domain="globals")  # bind it

        # Lagrangian arclength value around the interface
        geom = DropletGeometry(base_radius=self.R.value, volume=1)
        phi = var("lagrangian_x") * geom.contact_angle
        zcenter = geom.apex_height - geom.curv_radius  # center of the circle

        # Equations:
        eqs = YoungLaplaceEquation(p_ref, self.Bo*var("mesh_y"))

        # ICs: curve the mesh to a hanging droplet at Bo=0, set further ICs
        eqs += InitialCondition(mesh_x=geom.curv_radius * sin(phi),
                                mesh_y=-(zcenter+geom.curv_radius * cos(phi)))
        eqs += InitialCondition(curvature=2 / geom.curv_radius,
                                normalized_arclength=var("lagrangian_x"))
        eqs += InitialCondition(projected_normal_x=sin(phi),
                                projected_normal_y=-cos(phi))

        # apex and contact line boundary conditions
        eqs += DirichletBC(mesh_x=0, normalized_arclength=0)@"left"
        eqs += DirichletBC(mesh_y=0, normalized_arclength=1)@"right"
        eqs += EnforcedBC(mesh_x=var("mesh_x")-self.R)@"right"

        # contribute to the volume integral for the p_ref equation
        eqs += WeakContribution(1 / 3 * dot(var("mesh"),
                                var("normal")), testfunction(p_ref))
        
        # add some time derivative to prevent zero mass matrix 
        eqs += WeakContribution(partial_t("mesh"),testfunction("mesh"))

        # add the equation system to the mesh
        self += eqs@"interface"


# Create a problem, can used DropletDetachmentWithYoungLaplace() instead
with DropletDetachmentWithBulk() as problem:
    # Generate and compile also C code for the Hessian
    problem.setup_for_stability_analysis(analytic_hessian=True)

    # Go close to a guess of the bifurcation 
    problem.go_to_param(Bo=6.563)
    # force a mesh reconstruction
    problem.remesh_handler_during_continuation(force=True)
    problem.solve() # solve once more for the stationary solution

    problem.solve_eigenproblem(1) # solve 1 eigenvector as a guess

    # activate fold tracking, using the eigenfunction as guess    
    problem.activate_bifurcation_tracking("Bo","fold")
    problem.solve() # Solve the augmented system
    problem.output() # write output

    # Arclength continuation along the fold in terms of the radius
    ds=-0.01 # can also go to higher radii here by a +
    while problem.R.value>0.01:
        ds=problem.arclength_continuation(problem.R,ds)
        problem.output()
        problem.remesh_handler_during_continuation()



