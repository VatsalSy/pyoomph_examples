from sympy.matrices.expressions.matmul import any_zeros
from problem import RBCylindrical

import pathlib

with RBCylindrical() as problem:
    # Magic function: it will perform all necessary adjustments, i.e.:
    # -expand fields and test functions with exp(i*m*phi)
    # -consider phi-components in vector fields, i.e. here velocity
    # -incorporate phi-derivatives in grad and div
    # -generate the base residual, Jacobian, mass matrix and Hessian, 
    #  but also the corresponding versions for the azimuthal mode m!=0
    problem.setup_for_stability_analysis(analytic_hessian=True, azimuthal_stability=True)
    
    # Write results to file
    pathlib.Path(problem.get_output_directory()).mkdir(exist_ok=True)
    outfile = open(problem.get_output_directory("bifurcation_file.txt"), "w")

    # Loop over azimuthal modes m=0,1,2
    for m in [0,1,2]:
        
        # Set initial parameters
        problem.Gamma.value=0.5
        problem.Ra.value=1600

        # Solve once to get initial solution
        problem.solve(max_newton_iterations=100)
        
        # Find the critical Rayleigh number for the given azimuthal mode, 
        # via tweeking Ra to get eigenvalue zero. Due to the scaling of the pressure, we do not need to resolve for each Ra
        for parameter in problem.find_bifurcation_via_eigenvalues("Ra", initstep=1000, azimuthal_m=m, do_solve=False,epsilon=1e-2):
            print(parameter)

        # Activate bifurcation tracking, i.e. augment the system for the given azimuthal mode
        problem.activate_bifurcation_tracking('Ra',bifurcation_type="pitchfork" if m==0 else "azimuthal")
        problem.solve() # And jump on the bifurcation

        # Increase Gamma through arclength continuation from 0.5 to 3
        # with a maximum step of 0.02
        ds = 0.01
        while problem.Gamma.value<3:
            ds = problem.arclength_continuation('Gamma', ds, max_ds=0.02)
            problem.reset_arc_length_parameters()

            outfile.write(str(problem.Gamma.value) + "\t" + str(problem.Ra.value) + "\t" + str(m) + "\n")
            outfile.flush()
        
        # Deactivate bifurcation tracking for next solve
        problem.deactivate_bifurcation_tracking()

