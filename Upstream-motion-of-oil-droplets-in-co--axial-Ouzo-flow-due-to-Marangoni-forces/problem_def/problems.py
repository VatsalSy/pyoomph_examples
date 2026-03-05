from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.units import *
from pyoomph.equations.multi_component import *
from pyoomph.materials import *
from pyoomph.materials.default_materials import *
from pyoomph.equations.ALE import *
from pyoomph.utils.num_text_out import NumericalTextOutputFile
from pyoomph.output.meshio import TextFileOutputAlongLine
from .materials import *
from .meshes import *
from .equations import *


# Problem to check the jet distribution with a realistic nozzle inlet, but without any droplet
class JetWithRealisticNozzle(Problem):
    """
    Problem definition for simulating a jet with a realistic nozzle geometry.
    This is a simplified version without droplets, useful for studying jet flow characteristics.
    """
    def __init__(self):
        super().__init__()
        
        # ============================
        # Geometry Configuration
        # ============================
        # Store the experimental geometry
        self.geom = ExperimentalGeometry()

        # ============================
        # Physical Parameters
        # ============================
        # Ambient temperature (20°C) - used to evaluate liquid properties
        self.temperature = 20 * celsius
        
        # Gravitational acceleration in downward direction
        self.gravity = 9.81 * meter / second**2
        
        # Enable Boussinesq approximation for buoyancy effects
        self.boussinesq = True        
        
        # ============================
        # Fluid Definitions
        # ============================
        # Outer inflowing fluid: pure water
        self.outer_fluid = Mixture(get_pure_liquid("water"))
        
        # Jet composition: 88% ethanol + 12% anethole by volume
        self.jet_fluid = Mixture(88*percent*get_pure_liquid("ethanol")+12*percent*get_pure_liquid("anethole"), quantity="volume_fraction")
        
        # Internal variables for mixture composition (set by get_host_mixture)
        self._host_mix = None  # Combined mixture of jet and outer inflow
        self._jet_composition = {}  # Mass fractions for jet composition
        self._outer_composition = {}  # Mass fractions for outer composition

        # ============================
        # Flow Rate Parameters
        # ============================
        # Jet flow rate: 5 µL/min
        self.jet_flow_rate = 5 * micro * liter / minute
        
        # Water flow rate: 175 µL/min
        self.water_flow_rate = 175 * micro * liter / minute       
        
        # ============================
        # Blending Parameter
        # ============================
        # Parameter to blend composition from water to jet liquid at inflow
        # Useful for quickly calculating stationary solutions
        self.jet_composition_blending = self.define_global_parameter(jet_composition_blending=0)

    def get_host_mixture(self):
        """
        Creates and returns the host mixture by combining outer fluid and jet fluid components.
        The jet fluid components are initialized with zero mass fraction but will be modified in the jet region.
        Also populates composition dictionaries for outer and jet fluids.
        
        Returns:
            Mixture: The combined host mixture at the specified temperature.
        """
        # Only create mixture once
        if self._host_mix is None:
            # Build mixture starting with outer fluid components at their initial concentrations
            mix = sum(self.outer_fluid.initial_condition["massfrac_"+c]*self.outer_fluid.get_pure_component(c) for c in self.outer_fluid.components)
            
            # Add jet fluid components with zero initial mass fraction
            for jc in self.jet_fluid.components:
                mix += 0 * self.jet_fluid.get_pure_component(jc)
            
            # Create the mixture object
            self._host_mix = Mixture(mix, quantity="mass_fraction", temperature=self.temperature)
            
            # Store composition dictionaries for boundary conditions
            for c in self._host_mix.required_adv_diff_fields:
                self._outer_composition["massfrac_"+c] = self.outer_fluid.initial_condition.get("massfrac_", 0)
                self._jet_composition["massfrac_"+c] = self.jet_fluid.initial_condition.get("massfrac_"+c, 0)
        return self._host_mix
    
    def define_problem(self):
        """
        Main method to define the complete problem setup without droplet.
        """
        # ============================
        # Coordinate System and Scaling
        # ============================
        self.set_coordinate_system("axisymmetric")
        
        # Use nozzle inner radius as spatial scale
        self.set_scaling(spatial=self.geom.nozzle_inner_radius)
        
        # Set temporal scale if not already set
        if self.get_scaling("temporal", none_if_not_set=True) is None:
            self.set_scaling(temporal=1*milli*second)
        
        # Use host liquid properties for nondimensionalization
        self.get_host_mixture().set_reference_scaling_to_problem(self, temperature=self.temperature)

        # ============================
        # Mesh Definition
        # ============================
        self += FullNozzleGeometryMesh()

        # ============================
        # Temperature Definition
        # ============================
        # Isothermal flow: define temperature as global variable
        # Liquid properties depend on temperature, so we define it globally
        self.define_named_var(temperature=self.temperature)

        # ============================
        # Host Domain Equations
        # ============================
        host_eqs = MeshFileOutput(tesselate_tri=True)
        host_eqs += CompositionFlowEquations(self.get_host_mixture(), compo_space="C2", 
                                            boussinesq=self.boussinesq, gravity=self.gravity*vector(0,-1), 
                                            ns_mode="TH", with_IC=True)
        
        # Axisymmetry boundary condition at axis
        host_eqs += AxisymmetryBC()@"host_axis"
        
        # No-slip boundary condition at walls
        host_eqs += NoSlipBC()@"host_wall"
        host_eqs += NoSlipBC()@["nozzle_inner", "nozzle_top", "nozzle_outer"]

        # Enforce volumetric flow rate at water inflow by adjusting pressure
        host_eqs += EnforceVolumetricInflowByAdjustingThePressure(self.water_flow_rate)@"water_inflow"

        # Enforce volumetric flow rate at jet inflow
        host_eqs += EnforceVolumetricInflowByAdjustingThePressure(self.jet_flow_rate)@"jet_inflow"

        # ============================
        # Composition Initialization
        # ============================
        # Initialize entire domain with water composition
        host_eqs += InitialCondition(**self._outer_composition)
        
        # Enforce water composition at water inflow
        host_eqs += DirichletBC(**self._outer_composition)@"water_inflow"

        # Blend between jet and outer composition based on blending parameter
        # This helps in finding stationary solutions more easily
        jb = self.jet_composition_blending
        host_eqs += DirichletBC(**{fieldname: jb*self._jet_composition[fieldname]+(1-jb)*self._outer_composition[fieldname] 
                                   for fieldname in self._jet_composition.keys()})@"jet_inflow"

        # ============================
        # Adaptive Mesh Refinement
        # ============================
        # Refine mesh where composition gradients are large
        host_eqs += SpatialErrorEstimator(**{c: 1 for c in self._jet_composition.keys()})

        # ============================
        # Add some output files
        # ============================
        # Cross sections through the mesh to monitor the jet profile
        radial_range=self.geom.outer_radius
        N_sample=2000
        for offset_in_mm in [1,2,3,4,5,6,7,8,9]:
            offset=offset_in_mm*milli*meter            
            host_eqs+=TextFileOutputAlongLine("at_h_"+str(offset_in_mm)+"_mm",start=[0,offset],end=[radial_range,offset],N=N_sample)
        host_eqs+=TextFileOutput()
        
        # ============================
        # Add Equations to Problem
        # ============================
        self += host_eqs@"host"


# Problem definition for an ouzo jet with a droplet inside a coflow box
class OuzoJetWithDroplet(Problem):
    """
    Problem definition for simulating an ouzo jet with a droplet inside a coflow box.
    The problem includes geometry, physical parameters, fluid definitions, flow rates,
    droplet properties, mesh configuration, and constraint flags.
    """
    def __init__(self):
        super().__init__()

        # ============================
        # Geometry Configuration
        # ============================
        # Store the experimental geometry
        self.geom = ExperimentalGeometry()

        # ============================
        # Physical Parameters
        # ============================
        # Ambient temperature (20°C)
        self.temperature = 20 * celsius
        
        # Gravitational acceleration
        self.gravity = 9.81 * meter / second**2
        
        # Enable Boussinesq approximation for buoyancy
        self.boussinesq = True

        # ============================
        # Fluid Definitions
        # ============================
        # Outer fluid: pure water
        self.outer_fluid = Mixture(get_pure_liquid("water"))
        
        # Jet fluid: pure ethanol
        self.jet_fluid = Mixture(get_pure_liquid("ethanol"))  

        # Internal variables for mixture composition (set by get_host_mix)
        self._host_mix = None
        self._jet_composition = {}
        self._outer_composition = {}

        # ============================
        # Flow Rate Parameters
        # ============================
        # Initial jet flow rate parameter
        self.jet_flow_rate_init_val = self.define_global_parameter(jet_flow_rate_init_val=0)

        # Blending parameter for jet flow rate for getting stationary solution more easily
        self.jet_flow_lm_blend = self.define_global_parameter(jet_flow_lm_blend=0)
        
        # Actual jet flow rate (blended between initial value and variable)
        self.jet_flow_rate = self.jet_flow_rate_init_val * micro * liter / minute * (1 - self.jet_flow_lm_blend) + \
                             var("jet_flow_lm", domain="globals") * self.jet_flow_lm_blend
        
        # Outer water flow rate, fixed to 175 µL/min
        self.water_flow_rate = 175 * micro * liter / minute

        # Parameter to blend composition from water to jet liquid at inflow
        # Useful for quickly calculating stationary solutions
        self.jet_composition_blending = self.define_global_parameter(jet_composition_blending=0)

        # ============================
        # Droplet Properties
        # ============================
        # Droplet material: pure anethole
        self.drop_mix = Mixture(get_pure_liquid("anethole"))
        
        # Droplet radius parameter (in micrometers)
        self.droplet_radius_param = self.define_global_parameter(Rdrop=100)
        self.droplet_radius = self.droplet_radius_param * micro * meter
        
        # Initial droplet position along the jet
        self.droplet_position = 6.5 * milli * meter
        
        # Interface properties (set later)
        self._iprops = None

        # ============================
        # Mesh Configuration
        # ============================
        # Remeshing options for adaptive mesh refinement
        self.remeshing_options = RemeshingOptions()

        # ============================
        # Constraint Flags
        # ============================
        # Flag to fix droplet position
        self.fix_position = True
        
        # Flag to fix droplet volume
        self.fix_volume = True

    # Assemble the host mixture from outer fluid and jet fluid (there will have mass fraction 0, but will be changed in the jet)
    def get_host_mixture(self):
        """
        Creates and returns the host mixture by combining outer fluid and jet fluid components.
        The jet fluid components are initialized with zero mass fraction but will be modified in the jet region.
        Also populates composition dictionaries for outer and jet fluids.
        
        Returns:
            Mixture: The combined host mixture at the specified temperature.
        """
        if self._host_mix is None:
            # Build mixture starting with outer fluid components at their initial concentrations
            mix=sum(self.outer_fluid.initial_condition["massfrac_"+c]*self.outer_fluid.get_pure_component(c) for c in self.outer_fluid.components)
            
            # Add jet fluid components with zero initial mass fraction
            for jc in self.jet_fluid.components:
                mix+=0*self.jet_fluid.get_pure_component(jc)
            
            # Create the mixture object
            self._host_mix=Mixture(mix,quantity="mass_fraction",temperature=self.temperature)
            
            # Store composition dictionaries for boundary conditions
            for c in self._host_mix.required_adv_diff_fields:
                self._outer_composition["massfrac_"+c]=self.outer_fluid.initial_condition.get("massfrac_"+c,0)
                self._jet_composition["massfrac_"+c]=self.jet_fluid.initial_condition.get("massfrac_"+c,0)            
        return self._host_mix
    
    def get_interface_properties(self):
        """
        Returns the interface properties between the host mixture and the droplet.
        
        Returns:
            InterfaceProperties: Combined properties of host mixture and droplet mixture.
        """
        if self._iprops is None:
            self._iprops=self.get_host_mixture() | self.drop_mix
        return self._iprops
    
    # Define the problem
    def define_problem(self):
        """
        Main method to define the complete problem setup.
        """
        # ============================
        # Coordinate System and Scaling
        # ============================
        self.set_coordinate_system("axisymmetric")
        if self.get_scaling("temporal", none_if_not_set=True) is None:
            self.set_scaling(temporal=1*micro*second)
        self.set_scaling(spatial=self.geom.nozzle_inner_radius)
        self.get_host_mixture().set_reference_scaling_to_problem(self,temperature=self.temperature)
        # Isothermal flow
        self.define_named_var(temperature=self.temperature, absolute_pressure=1*atm)

        # ============================
        # Mesh Definition
        # ============================
        mesh=FullNozzleGeometryWithDropletMesh(self.droplet_radius,self.droplet_position)
        self+=mesh

        # ============================
        # Host Domain Equations
        # ============================
        host_eqs=MeshFileOutput(tesselate_tri=True)
        host_eqs+=PseudoElasticMesh()
        host_eqs+=RemeshWhen(remeshing_opts=self.remeshing_options, on_invalid_triangulation=True)
        host_eqs+=CompositionFlowEquations(self.get_host_mixture(),compo_space="C2",boussinesq=self.boussinesq,gravity=self.gravity*vector(0,-1),ns_mode="TH",with_IC=True)
        
        # Axisymmetry boundary conditions
        host_eqs+=AxisymmetryBC()@"host_axis"
        host_eqs+=AxisymmetryBC()@"host_axis_nozzle"
        host_eqs+=RemeshMeshSize(0.2)@"host_axis/host_axis_nozzle"
        
        # Wall boundary conditions
        host_eqs+=NoSlipBC()@"host_wall"
        host_eqs+=DirichletBC(mesh_x=True)@"host_wall"
        
        # Nozzle boundary conditions
        host_eqs+=DirichletBC(mesh_x=True,mesh_y=True)@["nozzle_top","nozzle_outer","nozzle_inner"]
        host_eqs+=DirichletBC(mesh_y=True)@["outflow","water_inflow","jet_inflow"]
        host_eqs+=PinMeshCoordinates()@["nozzle_inner","nozzle_outer","nozzle_top"]
        
        # Water inflow: enforce volumetric flow rate
        host_eqs+=EnforceVolumetricInflowByAdjustingThePressure(self.water_flow_rate)@"water_inflow"
        # Set initial condition to water composition everywhere
        host_eqs+=InitialCondition(**self._outer_composition)
        # Enforce water composition at water inflow
        host_eqs+=DirichletBC(**self._outer_composition)@"water_inflow"
        
        # Jet inflow: enforce volumetric flow rate
        host_eqs+=EnforceVolumetricInflowByAdjustingThePressure(self.jet_flow_rate)@"jet_inflow"
        # Blend between jet and outer composition based on blending parameter
        jb=self.jet_composition_blending
        host_eqs+=DirichletBC(**{fieldname:jb*self._jet_composition[fieldname]+(1-jb)*self._outer_composition[fieldname] for fieldname in self._jet_composition.keys()})@"jet_inflow"

        # No slip at the nozzle walls
        host_eqs+=NoSlipBC()@["nozzle_inner","nozzle_outer","nozzle_top"]

        # Refine where we have gradients in the composition
        host_eqs+=SpatialErrorEstimator(**{c:1 for c in self._jet_composition.keys()})

        # ============================
        # Droplet Domain Equations
        # ============================
        drop_eqs=MeshFileOutput(tesselate_tri=True)
        drop_eqs+=RemeshWhen(remeshing_opts=self.remeshing_options)
        drop_eqs+=CompositionFlowEquations(self.drop_mix,gravity=self.gravity*vector(0,-1),boussinesq=self.boussinesq,compo_space="C2",wrap_params_in_subexpressions=True)
        drop_eqs+=PseudoElasticMesh()
        drop_eqs+=AxisymmetryBC()@"droplet_axis"

        # Fix droplet position. fix_z Lagrange multiplier should give the force to keep the droplet at the position
        if self.fix_position:
            SF=scale_factor("spatial")**2/scale_factor("pressure")
            # Add global Lagrange multiplier for vertical position constraint
            self+=GlobalLagrangeMultiplier(fix_z=var("fix_z")*SF)@"globals"+Scaling(fix_z=scale_factor("pressure")/scale_factor("spatial"))@"globals"+TestScaling(fix_z=1/scale_factor("spatial"))@"globals"
            # Weak form: apply constraint force to droplet velocity
            drop_eqs+=WeakContribution(var("fix_z", domain="globals"),"velocity_y")
            # Weak form: enforce position constraint
            drop_eqs+=WeakContribution((var("coordinate_y")-self.droplet_position),testfunction("fix_z", domain="globals"))
            # Pin mesh far from droplet to prevent distortion
            Rnd=float(self.droplet_radius/self.get_scaling("spatial"))
            offs_nd=float(self.droplet_position/self.get_scaling("spatial"))
            host_eqs+=PinWhere(mesh_x=True,mesh_y=True,where=lambda xnd,ynd : xnd**2+(ynd-offs_nd)**2>(5*Rnd)**2 )

        if self.fix_volume:
            SF=scale_factor("pressure")
            # Add global Lagrange multiplier for volume constraint (only in stationary solve)
            self+=GlobalLagrangeMultiplier(fix_vol=-4/3*pi*self.droplet_radius**3, only_for_stationary_solve=True)@"globals"+Scaling(fix_vol=1/scale_factor("temporal"))@"globals"+TestScaling(fix_vol=1/scale_factor("spatial")**3)@"globals"
            # Weak form: integrate volume
            drop_eqs+=WeakContribution(1,testfunction("fix_vol", domain="globals"),dimensional_dx=True)
            # Weak form: apply volume constraint to pressure
            drop_eqs+=WeakContribution(var("fix_vol", domain="globals"), testfunction("pressure"))

        # Apply lagrange multiplier to get Qjet such that Fz=0 (zero net force on droplet)
        self+=GlobalLagrangeMultiplier(jet_flow_lm=0)@"globals"+Scaling(jet_flow_lm=scale_factor("velocity")*scale_factor("spatial")**2)@"globals"+TestScaling(jet_flow_lm=scale_factor("spatial")/scale_factor("pressure"))@"globals"
        # Couple jet flow rate to vertical force on droplet
        self+=WeakContribution(self.jet_flow_lm_blend*var("fix_z"),testfunction("jet_flow_lm"))@"globals"
        self+=ODEFileOutput()@"globals"

        # ============================
        # Interface Equations
        # ============================
        ieqs=MultiComponentNavierStokesInterface(self.get_interface_properties())
        ieqs+=ConnectMeshAtInterface()
        ieqs+=TextFileOutput()+MeshFileOutput(tesselate_tri=True)
        host_eqs+=ieqs@"host_droplet"

        # Refine mesh at the interface
        drop_eqs+=RefineToLevel()@"host_droplet"
        host_eqs+=RefineToLevel()@"host_droplet"

        # ============================
        # Output Configuration
        # ============================
        # Add some text file output
        drop_eqs+=TextFileOutput()
        host_eqs+=TextFileOutput()

        # ============================
        # Add Equations to Problem
        # ============================
        self.add_equations(host_eqs@"host"+drop_eqs@"droplet")
