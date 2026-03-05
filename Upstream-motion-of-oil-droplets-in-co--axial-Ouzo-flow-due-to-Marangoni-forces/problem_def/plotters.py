import matplotlib
from matplotlib.patches import ArrowStyle
from matplotlib.colors import LinearSegmentedColormap
from pyoomph.output.plotting import MatplotlibPlotter
from .problems import *
from matplotlib import cm
from matplotlib.colors import ListedColormap


class JetPlotter(MatplotlibPlotter):
    """Plotter for jet visualization with configurable view range modes."""
    def __init__(self, problem: Problem, filetrunk: str = "plot_{:05d}", fileext: str | List[str] = "png", eigenvector: int | None = None, eigenmode: Literal['abs', 'real', 'imag', 'merge', 'angle'] = "abs", add_eigen_to_mesh_positions: bool = True, position_eigen_scale: float = 1):
        super().__init__(problem, filetrunk, fileext, eigenvector, eigenmode, add_eigen_to_mesh_positions, position_eigen_scale)
        self.range_mode = "near"  # Either "near" or "all"
        self.z_offset = 0  # Potential offset for the "near" mode
        self.useLaTeXFont()

    def define_plot(self):
        # Configure image dimensions
        self.image_size = (330, 330)
        self.defaults("colorbar").hide_some_ticks = False
        
        # Get the problem instance and geometry parameters
        geom = cast(JetWithRealisticNozzle, self.get_problem()).geom
        
        # Select the plotting range based on the range_mode
        if self.range_mode == "near":
            # Near-field view: zoomed in on nozzle region
            width = 15 * geom.nozzle_inner_radius
            height = width * 2
            self.set_view(-width, -height * 0.25 + self.z_offset, width, height * 0.75 + self.z_offset)
        else:
            # Far-field view: entire simulation domain
            self.set_view(-geom.outer_radius, -geom.nozzle_length, geom.outer_radius, geom.length)
        
        # Create custom colormap for ethanol concentration (teal to pink)
        eth_cmap = LinearSegmentedColormap.from_list("TealPink", ["#96DCDC", "#FFFFFF", "#dc96dc"], N=20)
        
        # Configure ethanol mass fraction colorbar (top left)
        cb_eth = self.add_colorbar("ethanol [wt$\\%$]", cmap=eth_cmap, position="top left", factor=100, vmin=0, vmax=99) 
        
        # Calculate velocity magnitude for colorbar range
        domain_data = self._get_mesh_data("host")
        velocity_magnitude = 1000 * numpy.sqrt(domain_data.get_data("velocity_x")**2 + domain_data.get_data("velocity_y")**2)
        velocity_min = max(numpy.amin(velocity_magnitude), 1e-1)  # Avoid log(0)
        velocity_max = min(numpy.amax(velocity_magnitude), 1e6)  
        
        # Create semi-transparent viridis colormap for velocity
        viridis = cm.get_cmap('viridis')
        viridis_alpha = viridis(numpy.linspace(0, 1, 20))
        viridis_alpha[:, -1] = 0.9  # Set alpha channel to 0.9
        viridis_alpha_cmap = ListedColormap(viridis_alpha, N=20)
        
        # Configure velocity colorbar (top right) with logarithmic scale
        cb_velocity = self.add_colorbar(
            "velocity [mm/s]", 
            position="top right", 
            factor=1e3, 
            cmap=viridis_alpha_cmap, 
            norm=matplotlib.colors.LogNorm(vmin=velocity_min, vmax=velocity_max), 
            vmin=velocity_min, 
            vmax=velocity_max
        )
        
        # Adjust colorbar margins and discretization
        cb_eth.ymargin += 0.02
        cb_eth.Ndisc = 20
        cb_velocity.ymargin += 0.02
        cb_velocity.Ndisc = 20
        self.background_color = "transparent"

        # Plot ethanol mass fraction field with mirror transformation
        self.add_plot("host/massfrac_ethanol", colorbar=cb_eth, transform="mirror_x")

        # Calculate line width factor based on aspect ratio
        line_width_factor = (self.ymax - self.ymin) / (self.xmax - self.xmin)
        line_width_factor = 1.3

        if self.range_mode == "near":
            # Near-field mode: show velocity field with arrows
            cb_eth.invisible = False
            
            # Plot velocity field
            self.add_plot("host/velocity", colorbar=cb_velocity)
            
            # Add velocity arrows
            self.add_plot(
                "host/velocity", 
                mode="arrows", 
                transform=["mirror_x", None], 
                linewidths=line_width_factor, 
                arrowlength=line_width_factor * 1.7e-5, 
                arrowdensity=18
            )
            
            # Add top background polygon for cleaner appearance
            ybox_top = self.ymin + 0.8 * (self.ymax - self.ymin)
            self.add_polygon(
                [(self.xmin, ybox_top), (self.xmax, ybox_top), (self.xmax, self.ymax), (self.xmin, self.ymax)], 
                edgecolor="black", 
                facecolor="white", 
                alpha=0.75
            ).zindex = 5
        else:
            # Far-field mode: hide colorbars
            cb_velocity.invisible = True
            cb_eth.invisible = True
            # Plot ethanol field without velocity overlay
            self.add_plot("host/massfrac_ethanol", colorbar=cb_eth)
        
        # Plot nozzle geometry outlines
        nozzle_line_width = line_width_factor / 2
        self.add_plot("host/nozzle_top", transform=["mirror_x", None], linewidths=nozzle_line_width)
        self.add_plot("host/nozzle_inner", transform=["mirror_x", None], linewidths=nozzle_line_width)
        self.add_plot("host/nozzle_outer", transform=["mirror_x", None], linewidths=nozzle_line_width)

class FarPlotterWithDroplet(MatplotlibPlotter):
    """Plotter for the entire simulation domain with far-field view."""
    def __init__(self, problem = None, filetrunk = "far_plot_{:05d}", fileext = "png", eigenvector = None, eigenmode = "abs", add_eigen_to_mesh_positions = True, position_eigen_scale = 1, eigenscale = 1):
        super().__init__(problem, filetrunk, fileext, eigenvector, eigenmode, add_eigen_to_mesh_positions, position_eigen_scale, eigenscale)
    
    def define_plot(self):
        # Get problem instance and geometry parameters
        pr = cast(OuzoJetWithDroplet, self.get_problem())
        w = pr.geom.outer_radius
        h = pr.geom.length
        z = pr.droplet_position
        
        # Configure image dimensions based on aspect ratio
        self.image_size = [round(1280 * float((2 * w) / h)), 1280]
        self.set_view(-w / 2, 0 * z, w / 2, 1.1 * z)
        
        # Create custom colormap for ethanol concentration
        eth_cmap = LinearSegmentedColormap.from_list(
            "TealPink", ["#96DCDC", "white", "#DC96DC"], N=20
        )
        
        # Configure ethanol mass fraction colorbar
        cb_eth = self.add_colorbar(
            "ethanol [wt%]",
            cmap=eth_cmap,
            position="center left",
            factor=100,
            vmin=0,
            vmax=99
        )
        cb_eth.Ndisc = 20
        cb_eth.orientation = "vertical"
        cb_eth.thickness *= 6
        cb_eth.length *= 0.2
        cb_eth.consider_range(0, 0.01)
        cb_eth.invisible = True
        
        # Plot ethanol mass fraction field with mirror transformation
        self.add_plot(
            "host/massfrac_ethanol",
            colorbar=cb_eth,
            transform=["mirror_x", None]
        )
        
        # Plot droplet interface
        self.add_plot(
            "host/host_droplet",
            transform=["mirror_x", None],
            linecolor="black"
        )


class NearPlotterWithDroplet(MatplotlibPlotter):
    """Plotter for near-field view, zoomed in on the droplet region."""
    def __init__(self, problem = None, filetrunk = "near_plot_{:05d}", fileext = "png", eigenvector = None, eigenmode = "abs", add_eigen_to_mesh_positions = True, position_eigen_scale = 1, eigenscale = 1):
        super().__init__(problem, filetrunk, fileext, eigenvector, eigenmode, add_eigen_to_mesh_positions, position_eigen_scale, eigenscale)
    
    def define_plot(self):
        # Show all colorbar ticks
        self.defaults("colorbar").hide_some_ticks = False
        
        # Get problem instance and calculate zoom window
        pr = cast(OuzoJetWithDroplet, self.get_problem())
        w = 3 * pr.droplet_radius_param.value * micro * meter
        h = pr.droplet_position
        self.set_view(-w, h - w, w, h + w)
        
        # Create custom colormap for ethanol concentration
        eth_cmap = LinearSegmentedColormap.from_list(
            "TealPink", ["#96DCDC", "white", "#DC96DC"], N=20
        )
        
        # Configure ethanol colorbar (bottom right)
        cb_eth = self.add_colorbar(
            "ethanol [wt%]",
            cmap=eth_cmap,
            position="bottom right",
            factor=100,
            vmin=0,
            vmax=99
        )
        cb_eth.Ndisc = 20
        cb_eth.consider_range(0, 0.01)
        
        # Configure velocity colorbar (bottom left)
        cb_velo = self.add_colorbar(
            "velocity [mm/s]",
            cmap="viridis",
            position="bottom left",
            factor=1000
        )
        cb_velo.xmargin += 0.02
        
        # Set text and tick sizes for both colorbars
        for cb in [cb_eth, cb_velo]:
            cb.textsize = 50
            cb.ticsize = 50
        
        # Plot ethanol mass fraction field
        self.add_plot("host/massfrac_ethanol", colorbar=cb_eth)
        
        # Plot velocity fields for host and droplet
        self.add_plot("host/velocity", colorbar=cb_velo, transform="mirror_x")
        self.add_plot(
            "droplet/velocity",
            colorbar=cb_velo,
            transform=["mirror_x", None]
        )
        
        # Plot droplet interface
        self.add_plot(
            "host/host_droplet",
            transform=["mirror_x", None],
            linecolor="yellow"
        )
        
        # Add velocity arrows for host and droplet
        R = int(pr.droplet_radius_param.value)
        arrow_params = {
            "mode": "arrows",
            "transform": ["mirror_x", None],
            "arrowdensity": 17,
            "arrowlength": 3 / 10 * R * 1e-6,
            "linewidths": 8
        }
        
        arrows1 = self.add_plot("host/velocity", **arrow_params)
        arrows2 = self.add_plot("droplet/velocity", **arrow_params)
        
        # Configure arrow style for both sets
        arrow_style = ArrowStyle("-|>", head_length=1.2, head_width=0.8)
        for arrows in arrows1 + arrows2:
            arrows.arrowstyle = arrow_style
        
        # Add scale bar
        scale_bar = self.add_scale_bar(position=[0.9, 0.17])
        scale_bar.text_yoffset += 0.02
        scale_bar.textsize = 50
        
        # Add background polygon behind scale bar
        yoffset = self.ymin + 0.15 * (self.ymax - self.ymin)
        ybox = 0.1 * (self.ymax - self.ymin) + yoffset
        xbox = self.xmax - 0.2 * (self.xmax - self.xmin)
        self.add_polygon(
            [(xbox, yoffset), (self.xmax, yoffset), (self.xmax, ybox), (xbox, ybox)],
            edgecolor=None,
            facecolor="lightgrey",
            alpha=0.75
        ).zindex = 5
        
        # Add top background polygon
        ybox_top = self.ymax - 0.15 * (self.ymax - self.ymin)
        self.add_polygon(
            [(self.xmin, self.ymax), (self.xmax, self.ymax),
             (self.xmax, ybox_top), (self.xmin, ybox_top)],
            edgecolor="black",
            facecolor="lightgrey",
            alpha=0.75
        ).zindex = 5
        
        # Add bottom background polygon
        ybox_bottom = self.ymin + 0.15 * (self.ymax - self.ymin)
        self.add_polygon(
            [(self.xmin, ybox_bottom), (self.xmax, ybox_bottom),
             (self.xmax, self.ymin), (self.xmin, self.ymin)],
            edgecolor="black",
            facecolor="lightgrey",
            alpha=0.75
        ).zindex = 5
        
        # Add droplet radius text (top left)
        rtext = self.add_text(
            r"$R=" + "{:2.0f}".format(R) + r"\:\mathrm{\:\mu m}$",
            position="top left",
            textsize=50,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )
        rtext.xmargin += 0.05
        
        # Add jet flow rate text (top right)
        jet_flow_rate_lm = pr.get_ode("globals").get_value("jet_flow_lm")
        jet_flow_rate_lm_blend = pr.get_global_parameter("jet_flow_lm_blend").value
        Qjet = float(jet_flow_rate_lm * jet_flow_rate_lm_blend / (micro * liter / minute) + (1-jet_flow_rate_lm_blend) * pr.jet_flow_rate_init_val.value)
        self.add_text(
            r"$Q_\mathrm{{jet}}=" + "{:2.1f}".format(Qjet) + 
            r"\:\mathrm{\:\mu L min^{{-1}}}$",
            position="top right",
            textsize=50,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=1)
        )