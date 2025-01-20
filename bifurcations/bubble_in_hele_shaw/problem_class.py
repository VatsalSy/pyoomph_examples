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


# Import pyoomph, expressions like grad, Laplace-smoothed mesh and remeshing possibilities
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.equations.ALE import *
from pyoomph.meshes.remesher import Remesher2d


# Bulk equation class
class HeleShawFlowEquations(Equations):
    def __init__(self, b):
        super().__init__()
        self.b = b  # Store the cell height expression

    def define_fields(self):
        # Define the pressure field (second order, continuous)
        self.define_scalar_field("p", "C2")

    def define_residuals(self):
        # Solve the equation div(b^3 grad(p))=0
        p, ptest = var_and_test("p")
        self.add_weak(self.b**3 * grad(p), grad(ptest))


# Interface equation class
class BubbleInterface(InterfaceEquations):
    def __init__(self, Q, alpha, b, Ux, p_bubble):
        super().__init__()
        # Store the passed parameters
        self.Q, self.alpha, self.b, self.Ux, self.p_bubble = Q, alpha, b, Ux, p_bubble

    def define_fields(self):
        # Define the projected normal, the curvature and the Lagrange multiplier enforcing the dynamic BC.
        self.define_vector_field("proj_n", "C2")
        self.define_scalar_field("kappa", "C2")
        self.define_scalar_field("dynbc", "C2")

    def define_residuals(self):
        # Bind the unknowns and, if required their test functions
        n = var("normal")
        x, x_test = var_and_test("mesh")
        dynbc, dynbc_test = var_and_test("dynbc")
        pout, pouttest = var_and_test("p")
        np, np_test = var_and_test("proj_n")
        kappa, kappa_test = var_and_test("kappa")
        # Project the normal (since we cannot take div() on the normal directly)
        self.add_weak(np - n, np_test)
        # The curvature is calculated from the projected normal
        self.add_weak(kappa + div(np / square_root(dot(np, np))), kappa_test)
        # Jump in pressure
        jump_p = 1 / (3 * self.alpha * self.Q) * (1 / self.b + kappa / self.alpha)
        # Enforce the pressure jump by shifting the mesh accordingly
        self.add_weak(pout - self.p_bubble + jump_p, dynbc_test)
        self.add_weak(dynbc, dot(x_test, n))
        # Kinematic BC via a Neumann condition for the bulk pressure
        neumann_out = self.b * (self.Ux * var("normal_x") + dot(partial_t(x), n))
        self.add_weak(neumann_out, pouttest)


class MeshWithBubble(GmshTemplate):
    def define_geometry(self):
        self.mesh_mode = "tris"  # Triangualar mesh
        self.use_macro_elements = False  # We cannot use macro elements if we mirror the mesh
        # Add the points on the axis of symmetry
        pr = cast(DropletInCellProblem, self.get_problem())
        pinflow0 = self.point(-pr.L, 0, size=pr.far_resolution)
        poutflow0 = self.point(pr.L, 0, size=pr.far_resolution)
        pbubble_west = self.point(-pr.R, 0, size=pr.bubble_resolution)
        pbubble_east = self.point(pr.R, 0, size=pr.bubble_resolution)

        # Create upper (ysign=1) or lower (ysign=-1) half of the domain
        def create_half_of_the_domain(ysign):
            pinflow1 = self.point(-pr.L, ysign, size=pr.far_resolution)
            poutflow1 = self.point(pr.L, ysign, size=pr.far_resolution)
            self.create_lines(pinflow0, "inflow", pinflow1, "wall", poutflow1, "outflow", poutflow0)
            pbubble = self.point(0, ysign * pr.R, size=pr.bubble_resolution)
            self.circle_arc(pbubble_west, pbubble, through_point=pbubble_east, name="interface")
            self.circle_arc(pbubble_east, pbubble, through_point=pbubble_west, name="interface")

        # Either create only the half mesh (will be mirrored) or the full mesh (potentially not symmetric)
        create_half_of_the_domain(1)
        if pr.symmetric_mesh:
            lsymm1, lsymm2 = self.line(poutflow0, pbubble_east),self.line(pinflow0, pbubble_west)
            self.plane_surface(lsymm1, lsymm2, "wall", "outflow", "inflow", "interface", name="outer")
            self.mirror_mesh = "mirror_y"  # We construct only half of the mesh and mirror it afterwards
        else:
            create_half_of_the_domain(-1)  # We mesh the bottom half manually 
            self.plane_surface("wall", "outflow", "inflow", holes=[["interface"]], name="outer")

        # add remesher and inform about the hole
        self.remesher = Remesher2d(self)
        self.remesher.set_holes("outer", [["interface"]])


class DropletInCellProblem(Problem):
    def __init__(self):
        super().__init__()
        # Default constant parameters
        self.w, self.s, self.h = 0.25, 40, 0.024
        self.R, self.L, self.alpha = 0.46, 4, 40
        # Flow rate as variable parameter
        self.Q = self.define_global_parameter(Q=0.055)
        # Remeshing options and resolution
        self.remeshing = RemeshingOptions(max_expansion=1.75, min_expansion=0.6)
        self.far_resolution = 0.1
        self.bubble_resolution = 0.02
        self.symmetric_mesh = True

    # Shape of the constriction
    def get_b(self, y: Expression):
        b = 1 - self.h / 2 * (tanh(self.s * (y + self.w)) - tanh(self.s * (y - self.w)))
        # b is a complicated expressions. We just wrap it as subexpression.
        # Thereby, its value and derivatives will be calculated to dedicated double values
        return subexpression(b)

    def define_problem(self):
        self += MeshWithBubble()

        b = self.get_b(var("coordinate_y"))

        eqs = HeleShawFlowEquations(b)

        # Mesh dynamics: Laplace-smoothed with remeshing
        eqs += LaplaceSmoothedMesh()  # or eqs+=PseudoElasticMesh()
        eqs += RemeshWhen(self.remeshing)

        # Boundary conditions. Fixed pressure at the outflow, pressure gradient at the inflow, fixed mesh positions
        eqs += DirichletBC(p=0, mesh_x=self.L, mesh_y=True) @ "outflow"
        eqs += ( NeumannBC(p=-(b**3)) + DirichletBC(mesh_x=-self.L, mesh_y=True) ) @ "inflow"
        eqs += DirichletBC(mesh_y=True) @ "wall"

        # The bubble velocity Ux is determined such that the bubble is in the center
        Ux, Ux_test = self.add_global_dof("Ux", initial_condition=1)
        eqs += WeakContribution(var("coordinate_x"), Ux_test)@"interface"

        # The bubble pressure is determined by enforcing the volume
        pB, pB_test = self.add_global_dof("p_bubble", equation_contribution=-pi * self.R**2 )
        eqs += WeakContribution(1 / 2 * b * dot(var("coordinate"), -var("normal")), pB_test) @ "interface"

        # Adding the bubble
        eqs += BubbleInterface(Q=self.Q, alpha=self.alpha, b=b, Ux=Ux, p_bubble=pB) @ "interface"
        # We must initialize the projected normal to prevent division by 0
        eqs += InitialCondition(proj_n_x=var("normal_x"), proj_n_y=var("normal_y")) @ "interface"
        
        # Output, Paraview for the bulk, text file for the interface
        eqs += MeshFileOutput()
        eqs += TextFileOutput() @ "interface"

        # Add the equations to the problem
        self += eqs @ "outer"
