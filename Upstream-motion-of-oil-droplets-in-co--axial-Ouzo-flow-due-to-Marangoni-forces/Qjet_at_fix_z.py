from problem_def import *
from problem_def.plotters import FarPlotterWithDroplet, NearPlotterWithDroplet

with OuzoJetWithDroplet() as problem:
    # Configure compiler and solver settings
    problem.set_c_compiler("distutils").optimize_for_max_speed()
    problem.DTSF_max_increase_factor = 1.25  # Maximum time step increase factor
    problem.max_refinement_level = 6  # Maximum mesh refinement level
    problem.initial_adaption_steps = 0  # No initial adaptation steps

    # Add plotters for visualization
    problem.plotter = [FarPlotterWithDroplet(), NearPlotterWithDroplet()]
    
    # Set initial parameters
    problem.droplet_radius_param.value = 75  # Droplet radius in micrometers
    problem.jet_flow_rate_init_val.value = 6  # Initial jet flow rate in µL/min (start with a good guess)
    
    # Blend surface tension between interface value and constant value
    iprops = problem.get_interface_properties()
    sigma_blend = problem.define_global_parameter(sigma_blend=0)
    iprops.surface_tension = iprops.surface_tension * sigma_blend + (1 - sigma_blend) * 30 * milli * newton / meter
    
    # Initial solve
    problem.solve(max_newton_iterations=20)
    
    # Gradually increase jet composition blending (0 -> 0.1 -> 0.5 -> 1)
    problem.go_to_param(jet_composition_blending=0.1)
    problem.adapt() # adapt mesh after parameter change
    problem.go_to_param(jet_composition_blending=0.5, startstep=0.1)
    problem.go_to_param(jet_composition_blending=1, startstep=0.1) 
    problem.adapt() # adapt mesh after parameter change
    
    # Gradually increase surface tension blending (0 -> 0.1 -> 0.5 -> 1)
    problem.go_to_param(sigma_blend=0.1)
    problem.adapt() # adapt mesh after parameter change
    problem.go_to_param(sigma_blend=0.5, startstep=0.1)
    problem.adapt() # adapt mesh after parameter change
    problem.go_to_param(sigma_blend=1, startstep=0.1)
    problem.adapt() # adapt mesh after parameter change
    
    # Enable jet flow with Lagrange multiplier
    problem.go_to_param(jet_flow_lm_blend=1, startstep=0.1)
    problem.output_at_increased_time()