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


""" 
Use the Bifurcation GUI to make the bifurcation diagram 

A window will pop up, where you can use the mouse to select points and perform continuation with eigenvalue analysis.

  Keybindings:
    - Space: Perform a arclength step (or select point in "reorder branch" mode)
    - Shift+Space: Continue to the end of the current plotting range
    - PgUp/PgDown: Cycle through the points on the current branch
    - Home/End: Jump to first or last point on branch (in "reorder branch" mode: Move point there)
    - Enter: Load selected point (click or select by PgUp/PgDown first)
    - Backspace/Delete: Delete point
    - Numpad+/-: Increase/Decrease arclength step
    - Numpad *,/: Flip arclength direction
    - m: Reorder branch mode. You can select points by clicking and move them with PgUp/PgDown
    - b: Jump on closest bifurcation
    - 0-9: Mark point for full output (state files to be loaded by problem.load_state)
    - o: Write output
    - i: Toggle branch interpolation
    - t: Leave the branch by transient integration until a new stable one is found
    - y: Toggle the y-axis
    - Escape/q: Close program (will load the current diagram on restart)
 """

from problem_class import *
from pyoomph.utils.bifurcation_gui import *
import pyoomph.solvers.petsc
# Load the problem and the SLEPc eigensolver

with DropletInCellProblem() as problem:
    # Use a symmetric mesh or not, symmetric converges better!
    problem.symmetric_mesh = True      

    # Use the SLEPc eigensolver with the MUMPS backend for inversion during eigensolve
    problem.set_eigensolver("slepc").use_mumps()
    # Requires slepc4py and petsc4py
    
    # Replace the constant h by a global parameter for continuation later on
    problem.h=problem.define_global_parameter(h=0.024)

    
    # When we do not use a symmetric mesh, we take the weak symmetry contraint
    problem.setup_for_stability_analysis(
        analytic_hessian=True,
        use_hessian_symmetry=True
    )

    # Use a parameter for s. We first solve it with a smooth transition, s=10, for better initial convergence
    problem.s = problem.define_global_parameter(s=10)

    # Create the GUI before solving anything. Select "Q" as the parameter for the bifurcation diagram
    gui=BifurcationGUI(problem,"Q")
    
    # Take the best C compiler, activate -O3 -march=native -ffast-math
    problem.set_c_compiler("system").optimize_for_max_speed()

    problem.solve()
    # Make the transition in b sharper by continuation
    problem.go_to_param(s=40)

    # Start the GUI with some initial arclength step.
    gui.start(0.01)
