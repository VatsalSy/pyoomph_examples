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
from pyoomph.equations.ALE import *
from pyoomph.expressions.units import *
from pyoomph.materials import *
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.utils.dropgeom import DropletGeometry

# Build droplet mesh
class DropletMesh(GmshTemplate):
    def __init__(self, loaded_from_mesh_file: str | None = None, resolution: float | None = None):
        super().__init__(loaded_from_mesh_file) # inherit from GmshTemplate
        self.remesher=Remesher2d(self) # remesher object
        self.mesh_mode="tris" # mesh discretization mode
        self.default_resolution=resolution if resolution is not None else 0.02 # default resolution

    def define_geometry(self):        
        pr=self.get_problem() # get the problem object
        r=pr.radius # droplet radius
        assert isinstance(pr,DimensionalDropletProblem)
        geom=DropletGeometry(volume=pr.volume,base_radius=r) # droplet geometry

        # Liquid phase
        p00=self.point(0,0) # origin
        p0h=self.point(0,-geom.apex_height) # bottom point
        pR0=self.point(r,0) # right point
        self.circle_arc(pR0,p0h,through_point=(-r,0),name="interface") # interface
        self.create_lines(p0h,"axis",p00,"wall",pR0) # lines
        self.plane_surface("axis","wall","interface",name="liquid") # create the liquid domain

        # Gas phase
        R=pr.L # domain size
        gRR=self.point(R,-R,size=15*self.default_resolution) # bottom right point
        g0R=self.point(0,-R,size=10*self.default_resolution) # bottom left point
        gR0=self.point(R,0,size=10*self.default_resolution) # top right point
        self.create_lines(pR0,"wall_air",gR0,"right_air",gRR,"bottom_air",g0R,"axis_air",p0h) # lines
        self.plane_surface("wall_air","right_air","bottom_air","axis_air","interface",name="air") # create the air domain


# Properties of 1cSt silicon oil (https://doi.org/10.1016/j.ijthermalsci.2017.04.007)
new_pure_liquid("1cStSiliconeOil",
    mass_density=818*kilogram/meter**3, # density
    dynamic_viscosity=8.18e-4*kilogram/(meter*second), # dynamic viscosity
    specific_heat_capacity=2000*joule/(kilogram*kelvin), # specific heat capacity
    thermal_conductivity=0.1*watt/(meter*kelvin), # thermal conductivity        
    surface_tension=0.01687*newton/meter-7.55e-5*
        1/kelvin*(var("temperature")-25*celsius)*newton/meter # surface tension with air
    )

# Properties of air at 25 celsius
new_pure_gas("air",
    mass_density=1.293*kilogram/meter**3, # density
    dynamic_viscosity=1.81e-5*kilogram/(meter*second), # dynamic viscosity
    specific_heat_capacity=1005*joule/(kilogram*kelvin), # specific heat capacity
    thermal_conductivity=2.587e-2*watt/(meter*kelvin), # thermal conductivity
    override=True
    )

# Definition of Navier-Stokes equation with proper scaling
class NavierStokesEquations(Equations):
    def __init__(self, *, gravity=None, fluid_props:AnyFluidProperties=None, 
        dt_factor:ExpressionOrNum=1):
        super().__init__()
        self.gravity=gravity # bulk force
        self.fluid_props = fluid_props # fluid properties
        self.dt_factor=dt_factor # factor for advective terms

    # Define fields to be solved, i.e. velocity and pressure (Taylor-Hood elements)
    def define_fields(self):
        self.define_vector_field("velocity",space="C2",testscale=
            self.get_scaling("spatial")/self.get_scaling("pressure")) # velocity field
        self.define_scalar_field("pressure",space="C1",testscale=
            self.get_scaling("spatial")/self.get_scaling("velocity")) # pressure field

    # Define residual form of equations
    def define_residuals(self):
        u,utest=var_and_test("velocity") # velocity trial and test function
        p,ptest=var_and_test("pressure") # pressure trial and test function

        # Add residual forms to residual matrix
        self.add_weak(self.dt_factor*self.fluid_props.mass_density*
                material_derivative(u,u, ALE="auto"), utest) # advection term
        self.add_weak(2*self.fluid_props.dynamic_viscosity*
                sym(grad(u))-identity_matrix()*p, sym(grad(utest)))  # total stress
        self.add_weak(div(u),ptest) # continuity equation
        self.add_weak(-self.fluid_props.mass_density*
                self.gravity,utest) # gravity term

# Definition of heat equation with proper scaling
class HeatEquation(Equations):
    def __init__(self, fluid_props:AnyFluidProperties=None, dt_factor=1):
        super().__init__()
        self.fluid_props = fluid_props # fluid properties
        self.dt_factor=dt_factor # factor for dt terms

    # Define fields to be solved, i.e. temperature
    def define_fields(self):
        self.define_scalar_field("temperature",space="C2",
            testscale=scale_factor("temporal")/scale_factor("temperature")
            /(scale_factor("rho_cp")))

    # Define residual form of equations
    def define_residuals(self):
        T,Ttest=var_and_test("temperature") # temperature trial and test function
        u=var("velocity") # velocity field

        # Add residual forms to residual matrix
        self.add_weak(self.dt_factor*self.fluid_props.mass_density*
            self.fluid_props.specific_heat_capacity*
            material_derivative(T,velocity=u,ALE="auto"), Ttest) # advection term
        self.add_weak(self.fluid_props.thermal_conductivity*grad(T),
            grad(Ttest)) # diffusion term

# Definition of the free surface equations (Marangoni stresses and Laplace pressure)
class NavierStokesFreeSurfaceEquations(InterfaceEquations):
    def __init__(self, interface_properties:AnyFluidFluidInterface=None):
        super().__init__()
        self.interface_properties = interface_properties # fluid properties

    # Define Lagrange multiplier for the movement of the free surface
    def define_fields(self):
        self.define_scalar_field("kin_bc",space="C2",
            scale=1/test_scale_factor("mesh"),testscale=1/scale_factor("velocity"))

    # Define residual form of equations
    def define_residuals(self):
        u,u_test=var_and_test("velocity") # velocity trial and test function
        R,R_test=var_and_test("mesh") # mesh trial and test function
        l,l_test=var_and_test("kin_bc") # Lagrange multiplier trial and test function
        n=var("normal") # normal vector to interface

        # Add residual forms to residual matrix
        self.add_weak(dot(partial_t(R)-u,n),l_test) # free surface movement
        self.add_weak(l,dot(n,R_test)) # feedback of lagrange multiplier
        self.add_weak(self.interface_properties.surface_tension,
            div(u_test)) # Marangoni stresses

    
# Problem definition
class DimensionalDropletProblem(Problem):
    def __init__(self):
        super().__init__()        

        # Control parameters: volume, radius and temperature difference
        self.Vfactor=self.define_global_parameter(Vfactor=1) # factor for volume
        self.gradTfactor=self.define_global_parameter(gradTfactor=1) # factor for gradT
        self.ns_factor=self.define_global_parameter(ns_factor=1) # factor for dt terms to increase it slowly 
        
        # Problem settings
        self.remeshing=RemeshingOptions(min_expansion=0.9,max_expansion=1.1) # control remeshing
        self.mesh_element_size=0.02 # mesh resolution

        # Default configuration 
        # (same as: https://doi.org/10.1016/j.ijthermalsci.2017.04.007)
        self.radius=2.5*milli*meter # droplet radius
        self.L=3*self.radius # domain size
        self.gradT=self.gradTfactor*1*kelvin/(milli*meter) # temperature gradient
        self.volume=self.Vfactor*2/3*pi*(2.5*milli*meter)**3 # droplet volume
        self.temperature_wall=25*celsius # temperature of non-thermally-conducting top wall
        self.g=9.81*meter/second**2 # gravity
        
        # Store liquid properties
        self.liquid=get_pure_liquid("1cStSiliconeOil") # liquid phase
        self.gas=get_pure_gas("air") # air phase
    
    # Define problem
    def define_problem(self):
        
        # Set axisymmetric coordinate system
        self.set_coordinate_system("axisymmetric")

        # Set scaling for variables
        self.set_scaling(spatial=self.radius) # spatial
        self.set_scaling(temporal=1e-3*second) # temporal
        self.set_scaling(velocity=scale_factor("spatial")/scale_factor("temporal")) # velocity
        self.set_scaling(pressure=pascal) # pressure
        self.set_scaling(temperature=kelvin) # temperature
        self.set_scaling(rho_cp=self.liquid.mass_density*self.liquid.specific_heat_capacity) # rho*cp for temperature testscale

        # Add the mesh to the problem class
        self+=DropletMesh(resolution=self.mesh_element_size) # mesh object
        
        # Add the equations to the problem class
        for material in [self.liquid,self.gas]:
            eqs=MeshFileOutput() # output
            eqs+=RemeshWhen(self.remeshing) # control remeshing
            eqs+=LaplaceSmoothedMesh() # moving mesh
            eqs+=NavierStokesEquations(fluid_props=material,dt_factor=self.ns_factor,gravity=self.g*vector(0,-1)) # Navier-Stokes equations
            eqs+=HeatEquation(fluid_props=material,dt_factor=self.ns_factor) # heat equation
            eqs+=AxisymmetryBC()@("axis" if material==self.liquid else "axis_air") # axis of symmetry
            self+=eqs@"liquid" if material==self.liquid else eqs@"air"

        # Liquid boundary conditions
        leqs=DirichletBC(mesh_y=0,velocity_x=0,velocity_y=0,
            temperature=self.temperature_wall)@"wall" # imopse BCs at the wall
        p_V,p_V_test=self.add_global_dof("p_V_enforce",-self.volume,scaling=scale_factor("pressure"),
            testscaling=1/self.volume,only_for_stationary_solve=False) # global Lagrange multiplier to solve V=integral 1*dx == V0 by adjusting the pressure in one point. We first subtract V0=1 from the equation
        leqs+=WeakContribution(1,p_V_test,dimensional_dx=True) # then we add the volume integral to the equation
        leqs+=EnforcedBC(pressure=var("pressure")-p_V)@"interface/wall" # feed back to the average pressure in the droplet so that <p>=p_V_enforce by adjusting the pressure field
        
        # Air phase boundary conditions
        aeqs=DirichletBC(temperature=self.temperature_wall,
            mesh_y=0,velocity_x=0,velocity_y=0)@"wall_air" # impose BCs at the wall
        aeqs+=DirichletBC(temperature=self.temperature_wall-
            self.gradT*self.L,mesh_y=True,velocity_x=0,velocity_y=0)@"bottom_air" # impose BCs at the bottom
        aeqs+=DirichletBC(velocity_x=0,mesh_x=True)@"right_air" # no flow through the right boundary
        aeqs+=AverageConstraint(pressure=0) # average pressure constraint to avoid nullspace
        aeqs+=DirichletBC(mesh_x=True)@"axis_air" # fix y-movement of mesh and impose no-slip condition at the wall

        # Interface equations
        ieqs=NavierStokesFreeSurfaceEquations(interface_properties=self.liquid | self.gas) # free surface equations
        ieqs+=EnforcedBC(kin_bc=(var("mesh_x")-self.radius)*self.radius**2)@"wall" # adjust the kinematic boundary condition so that mesh_x=R holds
        ieqs+=ConnectMeshAtInterface() # connect the mesh at the interface
        ieqs+=ConnectFieldsAtInterface(["temperature","velocity_x","velocity_y"]) # connect fields at the interface
        leqs+=ieqs@"interface" # add the interface equations to the liquid equations
                    
        # Add the equation system
        self+=leqs@"liquid"+aeqs@"air"
