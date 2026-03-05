from problem_def import *
from problem_def.plotters import JetPlotter

# Initialize the jet problem with realistic nozzle geometry
with JetWithRealisticNozzle() as problem: 
    # Take the best compiler for the generated C code
    problem.set_c_compiler("distutils").optimize_for_max_speed()

    # Adaption and refinement settings
    problem.initial_adaption_steps=0 # Do not adapt the initial mesh
    problem.max_refinement_level=4 # Maximum level of mesh refinement
    problem.boussinesq=False # Use the full continuity equation, i.e. with varying density
    
    # Attach plotters for visualization: one for near-nozzle region, one for entire domain
    problem.plotter=[JetPlotter(problem,filetrunk="near_nozzle_{:05d}"),JetPlotter(problem,filetrunk="all_{:05d}")]
    problem.plotter[1].range_mode="all" # Second plotter shows full domain
    problem.plotter[0].file_ext="pdf" # Save near-nozzle plots as PDF
    problem.plotter[1].file_ext="pdf" # Save full domain plots as PDF

    # Initial solve: composition is pure water everywhere including jet inflow
    problem.initialise()
    problem.solve()
    
    # Gradually introduce jet composition using blending parameter
    # Start with 10% jet composition
    problem.go_to_param(jet_composition_blending=0.1)
    # Refine mesh at compositional gradients
    problem.adapt()
    
    # Increase to 20% jet composition
    problem.go_to_param(jet_composition_blending=0.2)
    problem.adapt()
    
    # Increase to 50% jet composition
    problem.go_to_param(jet_composition_blending=0.5)
    problem.get_mesh("host").set_max_neighbour_finding_tolerance(1e-12)
    problem.adapt()
    
    # Reach 100% jet composition (full physical problem)
    problem.go_to_param(jet_composition_blending=1)
    # Solve with one spatial adaptation step
    problem.solve(spatial_adapt=1)
    
    # Final refinement: reduce error tolerances for higher mesh resolution
    problem.get_mesh("host").max_permitted_error*=0.25 # Reduce max error by 75%
    problem.get_mesh("host").min_permitted_error*=0.0001 # Prevent unrefinement
    problem.solve(spatial_adapt=1) # Solve with refined mesh
    problem.output() # Output final results