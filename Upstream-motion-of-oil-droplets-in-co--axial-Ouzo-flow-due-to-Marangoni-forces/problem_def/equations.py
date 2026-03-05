from pyoomph import *
from pyoomph.expressions import *

# Enforce flow rate at the nozzle
class EnforceVolumetricInflowByAdjustingThePressure(IntegralConstraint):
    def __init__(self,volumetric_inflow:ExpressionOrNum):
        super().__init__(velocity_y=volumetric_inflow,scaling_factor="velocity")