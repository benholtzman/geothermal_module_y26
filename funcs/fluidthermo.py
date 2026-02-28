import numpy as np
from iapws import IAPWS95 


class FluidProps:
    """
    Wrapper for IAPWS-95 high-precision water formulations.
    Assumes isobaric properties for the thermal transient (decoupled P/T).
    """
    def __init__(self, Pf_MPa, Tf_C):
        self.Pf_MPa = Pf_MPa # default pressure
        self.Tf_C = Tf_C  # default temperature

    def get_properties(self):
        """
        Calculates fluid state at specific temperature.

        Returns:
            rho_f (float): Density [kg/m^3]
            eta_f  (float): Dynamic viscosity [Pa.s]
            cp_f  (float): Specific heat capacity [J/kg.K]
        """
        T_K = self.Tf_C + 273.15
        iap = IAPWS95
        state = iap(T=T_K, P=self.Pf_MPa)
        rho_f = state.rho # density
        eta_f = state.mu # dynamic viscosity
        cp_f = state.cp * 1000.0 # heat capacity (kJ/kg.K to J/kg.K)

        return rho_f, eta_f, cp_f
    
    