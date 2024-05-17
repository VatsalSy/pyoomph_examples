# Case from the paper
#	Lukas Babor & Hendrik C. Kuhlmann, Linear stability of thermocapillary flow in a droplet attached to a hot or cold substrate, Phys. Rev. Fluids 8(11), 114003, (2023)
#	https://dx.doi.org/10.1103/PhysRevFluids.8.114003


# Import the framework and the required equations, also import tools for droplet geometries
from pyoomph import *
from pyoomph.equations.navier_stokes import *
from pyoomph.equations.advection_diffusion import *
from pyoomph.utils.dropgeom import DropletGeometry
from pyoomph.expressions.units import degree

# Mesh via Gmsh
class DropletMesh(GmshTemplate):
    def define_geometry(self):
        # Get the problem to obtain the contact angle to calculate Gamma=apex_height
        pr=cast(DropletProblem,self.get_problem())
        Gamma=pr.get_Gamma()
        self.mesh_mode="tris" # Triangular mesh
        self.default_resolution=1/pr.n*square_root(Gamma) # Adjust resolution depending on the contact angle
        # Three corner points
        p00=self.point(0,0)
        p10=self.point(1,0,size=self.default_resolution*0.2) # finer at the contact line
        p0h=self.point(0,Gamma)
        # Boundaries
        self.create_lines(p0h,"axis",p00,"substrate",p10)
        self.circle_arc(p0h,p10,through_point=(-1,0),name="interface")
        # And the domain circumscribed by the boundaries
        self.plane_surface("axis","substrate","interface",name="droplet")
        
# Problem class. Here, the problem is specified
class DropletProblem(Problem):
    def __init__(self):
        super().__init__()
        # Default parameters: Contact angle (fixed) and variable parameters, mesh resolution and either a hot or cold substrate
        self.contact_angle=90*degree
        self.Re, self.Pr, self.Bi, self.Bd=self.define_global_parameter(Re=1,Pr=16.36, Bi=0.236, Bd=0)
        self.n=50 # mesh resolution
        self.hot_substrate=True # select the thermal boundary condition at the substrate

    # Getters and setters to get MaH from Re and vice versa
    def get_MaH(self):
        return float(self.Bi*self.get_Gamma()**2/(1+self.Bi*self.get_Gamma())*self.Re*self.Pr)
    
    def set_MaH(self,MaH):        
        self.Re.value=float((MaH/(self.Bi*self.get_Gamma()**2/(1+self.Bi*self.get_Gamma())))/self.Pr)

    # Calculate Gamma from the contact angle
    def get_Gamma(self):
        return DropletGeometry(base_radius=1,contact_angle=self.contact_angle).apex_height

    # Combining mesh(es) and equations
    def define_problem(self):
        # Evaluate all on cylinderical coordinates
        self.set_coordinate_system("axisymmetric")

        self+=DropletMesh() # add the mesh
        
        # Assemble the equation system
        eqs=MeshFileOutput() # output for Paraview

        # Flow and temperature
        eqs+=NavierStokesEquations(mass_density=1,dynamic_viscosity=1,gravity=self.Re*self.Bd/self.get_Gamma()**2*var("T")*vector(0,1))
        eqs+=AdvectionDiffusionEquations("T",diffusivity=1/self.Pr)
        eqs+=InitialCondition(T=1 if self.hot_substrate else -1)

        # Boundary conditions: No-slip and isothermal substrate
        eqs+=(NoSlipBC()+DirichletBC(T=1 if self.hot_substrate else -1))@"substrate"
        eqs+=AxisymmetryBC()@"axis" # The AxisymmetryBC will take care of toggeling the boundary conditions for the different values of m
        # Free surface: Kinematic BC and Marangoni flow and heat transfer by a simple Biot number model
        eqs+=(NavierStokesFreeSurface(surface_tension=-self.Re*var("T")) + NeumannBC(T=self.Bi/self.Pr*var("T")))@"interface"

        # Pressure has a nullspace: Enforce the pressure at the contact line to be 0 by adjusting the pressure, but not for the angular eigenmodes
        eqs+=EnforcedBC(pressure=var("pressure")-0,set_zero_on_angular_eigensolve=True)@"interface/substrate"

        # Add the equation system to the "droplet" domain of the added mesh
        self+=eqs@"droplet"
