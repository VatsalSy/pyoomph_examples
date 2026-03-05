from pyoomph import *
from pyoomph.expressions import *
from pyoomph.materials import *

# Material properties for trans-anethole (oil phase component)
# Trans-anethole is the primary component of anise oil used in ouzo droplet studies
@MaterialProperties.register()
class PureLiquidTransAneethole(PureLiquidProperties):
    name="anethole" 
    def __init__(self):
        super().__init__()
        # Molar mass of trans-anethole
        self.molar_mass=148.22*gram/mol
        # Mass density at atmospheric conditions
        self.mass_density=987.80*kilogram/meter**3
        # Dynamic viscosity at atmospheric conditions
        self.dynamic_viscosity=2.45e-3*pascal*second    
        # Surface tension with gas phase (air)
        self.default_surface_tension["gas"]=31.8*milli*newton/meter
        # UNIFAC functional groups for thermodynamic calculations
        # CH3: methyl group, CH=CH: vinyl group, ACH: aromatic CH, AC: aromatic C, CH30: methoxy group
        self.set_unifac_groups({"CH3":1,"CH=CH":1,"ACH":4,"AC":2,"CH3O":1},only_for={"Original","Dortmund"})

# Mixture of ethanol and trans-anethole: Only used to assemble the jet mixture
# This is a binary mixture class primarily for initialization purposes
@MaterialProperties.register()
class MixtureLiquidEthanolTransAnethole(MixtureLiquidProperties):
    components={"ethanol","anethole"}
    def __init__(self, pure_props: Dict[str, MaterialProperties]):
        super().__init__(pure_props)
        # Property definition intentionally skipped - only used as component of ternary mixture

# Ternary mixture of ethanol, water and trans-anethole
# This mixture represents the complete system in ouzo droplet experiments
@MaterialProperties.register()
class MixtureLiquidWaterEthanolTransAnethole(MixtureLiquidProperties):
    components={"ethanol","anethole","water"}
    def __init__(self, pure_props: Dict[str, MaterialProperties]):
        super().__init__(pure_props)
        # Extract local mass fractions for each component
        yA=self.get_mass_fraction_field("anethole")  # Mass fraction of anethole
        yE=self.get_mass_fraction_field("ethanol")   # Mass fraction of ethanol
        TKelvin=var("temperature")/kelvin            # Temperature in Kelvin

        # Mass density calculation:
        # Strategy: Linear blending between water-ethanol mixture density and pure anethole density
        # Water-ethanol density uses nonlinear correlation with ethanol mass fraction
        # Note: This approximation assumes negligible volume change on mixing with anethole
        rho_ethanol_water=(997.0479+(789.0-997.0479)*(0.65951*yE+(1.0-0.65951)*yE**2))*kilogram/(meter**3)
        rho_anethole=self.get_pure_component("anethole").mass_density
        self.mass_density=yA*rho_anethole+(1-yA)*rho_ethanol_water
        
        # Dynamic viscosity calculation:
        # Similar linear blending approach between water-ethanol and anethole
        # Water-ethanol viscosity: temperature-dependent polynomial in ethanol mass fraction
        # Coefficients fitted from experimental data
        # Note: This ignores potential non-ideal mixing effects in the ternary system
        mu_ethanol_water=((0.000834378-1.77674e-5*(TKelvin-298.15))+(0.00670473-0.00030053*(TKelvin-298.15))*yE + (-0.00884472+0.000451734*(TKelvin-298.15)) *yE**2 + (0.00237477-0.000152765*(TKelvin-298.15))*yE**3)* pascal*second 
        mu_anethole=self.get_pure_component("anethole").dynamic_viscosity
        self.dynamic_viscosity=yA*mu_anethole+(1-yA)*mu_ethanol_water
        

        # Diffusion coefficient calculation:
        # Simplified diagonal diffusion matrix based on ethanol-water binary diffusivity
        # Uses correlation dependent on ethanol mass fraction
        # Limitation: Neglects off-diagonal coupling terms and anethole-specific diffusion
        # A full treatment would require a 3x3 diffusion matrix with cross-diffusion terms
        self.set_diffusion_coefficient(1.25477e-9*(1.0-2.7794*yE+2.72277*yE**2)* meter**2/second )         

        # Activity coefficients using UNIFAC-Dortmund model
        # This accounts for non-ideal thermodynamic behavior in the ternary mixture
        self.set_activity_coefficients_by_unifac("Dortmund")

# Liquid-liquid interface properties between water-ethanol mixture and trans-anethole
# This interface is critical for ouzo droplet formation and stability
@MaterialProperties.register()
class LiquidLiquidInterfaceWaterEthanolVsAnethole(LiquidLiquidInterfaceProperties):
    # Phase A: water-ethanol mixture (aqueous phase)
    componentsA={"water","ethanol"}
    # Phase B: trans-anethole (oil phase)
    componentsB={"anethole"}
    
    def __init__(self, phaseA: MaterialProperties | BaseLiquidProperties | BaseGasProperties | BaseSolidProperties | PureSolidProperties | PureLiquidProperties | PureGasProperties | MixtureLiquidProperties | MixtureGasProperties, phaseB: MaterialProperties | BaseLiquidProperties | BaseGasProperties | BaseSolidProperties | PureSolidProperties | PureLiquidProperties | PureGasProperties | MixtureLiquidProperties | MixtureGasProperties, surfactant_dict: Dict[SurfactantProperties, Expression | int | float]):
        super().__init__(phaseA, phaseB, surfactant_dict)
        # Water mass fraction in the aqueous phase
        yWw=var("massfrac_water")
        # Interfacial tension correlation as function of water mass fraction
        # Sixth-order polynomial fit from experimental data
        # Reference: DOI https://doi.org/10.1039/D4SM00332B
        self.surface_tension= (194.895493504303*yWw**6 - 536.800060425943*yWw**5 + 596.903069368497*yWw**4 - 320.082260350984*yWw**3 + 97.3392901100121*yWw**2 - 8.57341513657779*yWw + 0.232830107990765)*milli*newton/meter