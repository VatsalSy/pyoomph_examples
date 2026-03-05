from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.remesher import Remesher2d
from pyoomph.expressions.units import *

# Storage class for experimental geometry parameters
class ExperimentalGeometry:
    """
    Container for experimental geometry dimensions used in the simulations.
    All dimensions are based on typical microfluidic experiments.
    """
    def __init__(self) -> None:
        # Calculate outer domain radius from box cross-sectional area
        dbox=2*milli*meter
        self.outer_radius=square_root(dbox**2/pi)
        
        # Domain and nozzle dimensions
        self.length=10*milli*meter
        self.nozzle_inner_radius=30*micro*meter/2
        self.nozzle_outer_radius=(self.nozzle_inner_radius+20*micro*meter)/2
        self.nozzle_length=1*milli*meter
        self.nozzle_angle=30*degree

# Mesh with the full nozzle geometry (without droplet)
class FullNozzleGeometryMesh(GmshTemplate):
    """
    Gmsh mesh template for the full box geometry without a droplet.
    The mesh includes only the nozzle geometry and the host phase.
    Coordinate system: axisymmetric with z=0 at the nozzle outlet.
    """
    def __init__(self, geom:ExperimentalGeometry=ExperimentalGeometry()):
        super().__init__()
        self.geom=geom

    def define_geometry(self):
        # Use quadrilateral elements for adaptive mesh refinement
        self.mesh_mode="only_quads"

        # Get the experimental geometry parameters
        g=self.geom
        
        # Characteristic mesh sizes (in micrometers) for different regions
        # These serve as initial values; spatial adaptation refines where needed
        stip=0.1        # Nozzle tip (finest resolution)
        sinflow=20      # Inflow regions
        swall=30        # Domain walls (coarsest resolution)
        
        # Define corner points of the nozzle geometry (nozzle outlet at z=0)
        # Axisymmetric axis points
        p00=self.point(0,0,size=stip)      # Axis at nozzle tip
        p0B=self.point(0,-g.nozzle_length,size=sinflow)  # Axis at nozzle base
        p0H=self.point(0,g.length,size=swall)            # Axis at top
        
        # Nozzle geometry points
        pRi0=self.point(g.nozzle_inner_radius,0,size=stip)  # Inner radius at outlet
        pRo0=self.point(g.nozzle_outer_radius,0,size=stip)  # Outer radius at outlet
        pRiB=self.point(g.nozzle_inner_radius+tan(g.nozzle_angle)*g.nozzle_length,-g.nozzle_length,size=sinflow)  # Inner radius at base
        pRoB=self.point(g.nozzle_outer_radius+tan(g.nozzle_angle)*g.nozzle_length,-g.nozzle_length,size=sinflow)  # Outer radius at base
        
        # Outer domain boundary points
        pRfB=self.point(g.outer_radius,-g.nozzle_length,size=sinflow)  # Outer wall at nozzle base
        pRfH=self.point(g.outer_radius,g.length,size=swall)            # Outer wall at top

        # Create separator line to improve mesh quality
        # This internal line helps guide the meshing algorithm
        self.line(p00,pRi0,name="_sep")

        # Create outer boundary of the host domain with named interfaces
        line_loop=self.create_lines(
            p0B,"jet_inflow",           # Inflow through nozzle inner channel
            pRiB,"nozzle_inner",        # Inner nozzle wall
            pRi0,"nozzle_top",          # Nozzle tip (inner to outer radius)
            pRo0,"nozzle_outer",        # Outer nozzle wall
            pRoB,"water_inflow",        # Inflow around nozzle exterior
            pRfB,"host_wall",           # Outer domain wall (side)
            pRfH,"outflow",             # Outflow at top
            p0H,"host_axis",            # Axis from top to nozzle tip
            p00,"host_axis",            # Axis at nozzle tip
            p0B
        )
        
        # Define host domain
        self.plane_surface(*line_loop,name="host")
    


# Mesh with the full nozzle geometry
class FullNozzleGeometryWithDropletMesh(GmshTemplate):
    """
    Gmsh mesh template for the full box geometry with a droplet at a certain position.
    The mesh includes the nozzle geometry and a spherical droplet in the host phase.
    Coordinate system: axisymmetric with z=0 at the nozzle outlet.
    """
    def __init__(self, droplet_radius, droplet_position, geom:ExperimentalGeometry=ExperimentalGeometry()):
        super().__init__()
        self.droplet_radius=droplet_radius  
        self.droplet_position=droplet_position
        self.geom=geom
        self.remesher=Remesher2d(self)

    def define_geometry(self):
        # Use quadrilateral elements for adaptive mesh refinement
        self.mesh_mode="only_quads"

        # Get the experimental geometry parameters
        g=self.geom
        
        # Characteristic mesh sizes (in micrometers) for different regions
        # These serve as initial values; spatial adaptation refines where needed
        stip=0.2        # Nozzle tip (finest resolution)
        sinflow=20      # Inflow regions
        swall=30        # Domain walls (coarsest resolution)
        sdroplet=1      # Droplet interface
        
        # Define corner points of the nozzle geometry (nozzle outlet at z=0)
        # Axisymmetric axis points
        p00=self.point(0,0,size=stip)      # Axis at nozzle tip
        p0B=self.point(0,-g.nozzle_length,size=sinflow)  # Axis at nozzle base
        p0H=self.point(0,g.length,size=swall)            # Axis at top
        
        # Nozzle geometry points
        pRi0=self.point(g.nozzle_inner_radius,0,size=stip)  # Inner radius at outlet
        pRo0=self.point(g.nozzle_outer_radius,0,size=stip)  # Outer radius at outlet
        pRiB=self.point(g.nozzle_inner_radius+tan(g.nozzle_angle)*g.nozzle_length,-g.nozzle_length,size=sinflow)  # Inner radius at base
        pRoB=self.point(g.nozzle_outer_radius+tan(g.nozzle_angle)*g.nozzle_length,-g.nozzle_length,size=sinflow)  # Outer radius at base
        
        # Outer domain boundary points
        pRfB=self.point(g.outer_radius,-g.nozzle_length,size=sinflow)  # Outer wall at nozzle base
        pRfH=self.point(g.outer_radius,g.length,size=swall)            # Outer wall at top

        # Define droplet geometry (spherical droplet centered at droplet_position)
        dr = self.droplet_radius
        dh = self.droplet_position
        p0dt=self.point(0,dh+dr,size=sdroplet)  # Top of droplet on axis
        p0dc=self.point(0,dh,size=sdroplet)     # Center of droplet on axis
        p0db=self.point(0,dh-dr,size=sdroplet)  # Bottom of droplet on axis
        pddc=self.point(dr,dh,size=sdroplet)    # Droplet equator (rightmost point)

        # Create separator lines to improve mesh quality
        # These internal lines help guide the meshing algorithm
        self.line(p00,pRi0,name="_sep")
        self.circle_arc(self.point(0,dh+2*dr,size=sdroplet),self.point(2*dr,dh,size=sdroplet),center=p0dc,name="_sep")
        self.circle_arc(self.point(0,dh-2*dr,size=sdroplet),self.point(2*dr,dh,size=sdroplet),center=p0dc,name="_sep")

        # Create droplet interface (two semicircular arcs)
        self.circle_arc(p0db,pddc,center=p0dc,name="host_droplet")  # Lower half
        self.circle_arc(pddc,p0dt,center=p0dc,name="host_droplet")  # Upper half
        
        # Create axis boundaries of droplet
        self.create_lines(p0dt,"droplet_axis",p0dc,"droplet_axis",p0db)
        
        # Define droplet domain
        self.plane_surface("host_droplet","droplet_axis",name="droplet")

        # Create outer boundary of the host domain with named interfaces
        line_loop=self.create_lines(
            p0db,"host_axis",           # Axis from droplet bottom to nozzle tip
            p00,"host_axis_nozzle",     # Axis inside nozzle
            p0B,"jet_inflow",           # Inflow through nozzle inner channel
            pRiB,"nozzle_inner",        # Inner nozzle wall
            pRi0,"nozzle_top",          # Nozzle tip (inner to outer radius)
            pRo0,"nozzle_outer",        # Outer nozzle wall
            pRoB,"water_inflow",        # Inflow around nozzle exterior
            pRfB,"host_wall",           # Outer domain wall (side)
            pRfH,"outflow",             # Outflow at top
            p0H,"host_axis",            # Axis from top to droplet top
            p0dt
        )
        
        # Define host domain (excludes droplet interior)
        self.plane_surface(*line_loop,"host_droplet",name="host")