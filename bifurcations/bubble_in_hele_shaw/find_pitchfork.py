# Load the problem and the SLEPc eigensolver
from problem_class import *

# PETSc and SLEPc are required for a good eigenfunction guess
import pyoomph.solvers.petsc

from pyoomph.utils.num_text_out import NumericalTextOutputFile

with DropletInCellProblem() as problem:
    # Use a symmetric mesh or not, symmetric converges better!
    problem.symmetric_mesh = True      
    
    # Replace the constant h by a global parameter for continuation later on
    problem.h=problem.define_global_parameter(h=0.024)

    # Use the SLEPc eigensolver with the MUMPS backend for inversion during eigensolve
    problem.set_eigensolver("slepc").use_mumps()
    # Requires slepc4py and petsc4py
    
    # When we do not use a symmetric mesh, we take the weak symmetry contraint
    problem.setup_for_stability_analysis(
        analytic_hessian=True,
        use_hessian_symmetry=True,
        improve_pitchfork_on_unstructured_mesh=not problem.symmetric_mesh # Use our improved symmetry constraint on unsymmetric meshes
    )

    # Use a parameter for s. We first solve it with a smooth transition, s=10, for better initial convergence
    problem.s = problem.define_global_parameter(s=10)

    # Take the best C compiler, activate -O3 -march=native -ffast-math
    problem.set_c_compiler("system").optimize_for_max_speed()

    problem.solve()
    # Make the transition in b sharper by continuation
    problem.go_to_param(s=40)
    # Increase the flow rate, go close to the pitchfork
    problem.go_to_param(Q=0.07)
    # Get a guess for the eigenvector, which also serves as symmetry constraint vector
    problem.solve_eigenproblem(10)
    problem.activate_bifurcation_tracking("Q", "pitchfork")
    problem.solve(max_newton_iterations=50)
    print("Found Pitchfork at Q=", problem.Q.value)
    
    bifurcation_curve=NumericalTextOutputFile(problem.get_output_directory("bifurcation_curve.txt"),header=["h","Q"])
    bifurcation_curve.add_row(problem.h, problem.Q)
    
    # Continuation in h
    dh=0.001
    while problem.h.value<0.06:
        dh=problem.arclength_continuation("h", dh)
        problem.output_at_increased_time()
        bifurcation_curve.add_row(problem.Q, problem.h)
    
