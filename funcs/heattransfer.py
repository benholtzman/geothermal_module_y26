
import numpy as np
from scipy.special import erfc

from iapws import IAPWS95 




# ========================================================
# Geometry, conditions and properties for BOTH reservoir models
# ========================================================



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
        rho_f = state.rho
        eta_f = state.mu
        cp_f = state.cp * 1000.0

        return rho_f, eta_f, cp_f

class RockProps:
    def __init__(self, Pr_MPa, Tr_C):
        self.Pr_MPa = Pr_MPa # default pressure
        self.Tr_C = Tr_C  # default temperature

    def get_Cp_T_K(self):
        #[J/kg/K] 
        T0 = 273 # [K]
        a = 1   
        b = d = 0.065 
        Tr_K = self.Tr_C + 273.15
        #print(f"Tr_K= {Tr_K} K")
        Cp_0 = 790.0
        Cp = Cp_0*( a + b*(Tr_K/T0) - d*(Tr_K/T0)**(-2))
        return Cp


class ReservoirConfig:
    """
    Global configuration object defining the geologic control volume and
    operational constraints.
    """
    def make_cfg(self,params_common={}):
        # vars passed in by dict: 
        # T_res0_C,Tf_in_C,dP_Pa,K_m2,phi,phi2

        # --- GEOMETRY ---
        self.L_m = params_common['L_flowdir_z']  # Length in flow direction (m)
        self.x_res_m = params_common['x_face'] # Reservoir x-extent (m)
        self.z_res_m = params_common['y_face'] # Reservoir z-extent (m)
        self.A_m2 = self.x_res_m * self.z_res_m  # Reservoir cross-sectional area (m^2)
        self.V_res_m3 = self.A_m2 * self.L_m  # Reservoir Volume (m^3)

        # # Tushar's values:
        # self.Height = 500.0     # Reservoir Height (m)
        # self.Width = 500.0      # Reservoir Width (m)
        # self.H_strata = 200.0   # Geologic Cap (m) - Max height of any fault

        # --- INITIAL TEMPERATURES, Pressure ---
        self.Tf_in_C = params_common['Tf_in_C']         # Initial Reservoir Temperature (degC)
        self.T_surf_C = params_common['T_surf_C']          # Surface Temperature (degC)
        self.T_res0_C = params_common['T_res0_C']          # Initial Reservoir Temperature (degC)
        self.P_res_MPa = params_common['P_res_MPa']         # Initial Reservoir Pressure (MPa)

        # --- FLOW Boundary conditions ---
        # bring this back if self-consistently calculate flow through fracture, not set v ! 
        # self.dP_Pa = params_common['dP_Pa']             # Pressure Drop across Reservoir (Pa)

        # --- ROCK PROPS (Granite) ---
        rockprops = RockProps(self.P_res_MPa, self.T_res0_C)
        self.cp_r = rockprops.get_Cp_T_K()  # [J/kg/K]
        print(f"Rock Cp at {self.T_res0_C} C = {self.cp_r:.1f} J/kg/K")

        # these are Bodvarsson model parameters, not used in Gringarten model:
        # these are moved to params_bd. 
        #self.K_m2 = params_common['K_m2']             # Permeability (m^2)
        #self.phi = params_common['phi']               # Porosity (fraction)
        #self.phi2 = params_common['phi2']             # Porosity factor for area (fraction)

        # Tushar's values:
        #self.k_r = 2.5           # Thermal Conductivity [W/m.K]
        self.rho_r = params_common['rho_r']      # Rock Density [kg/m^3]
        #self.cp_r = 1000.0        # Rock Specific Heat [J/kg.K]

        # --- FLUID PROPS (Water) ---
        # ***********************
        # BEWARE THIS IS NOW PASSING LITHOSTATIC PRESSURE IN, not hydrostatic !
        # ***********************
        fluidprops = FluidProps(self.P_res_MPa, self.T_res0_C) 
        # for Bodvar, the T is assumed to T_res instantly. 
        # NOT changing these as a function of temp ! 
        rho_f, eta_f, cp_f = fluidprops.get_properties()
        self.rho_f = rho_f 
        self.eta_f = eta_f
        self.cp_f = cp_f
        # print(f"Fluid cp at {self.T_res0_C} C = {self.cp_f:.1f} J/kg/K")
        # print(f"Fluid rho at {self.T_res0_C} C = {self.rho_f:.1f} kg/m3")
        # print(f"Fluid eta at {self.T_res0_C} C = {self.eta_f:.4e} Pa.s")
        self.HiP  = self.V_res_m3 * self.rho_r * self.cp_r * (self.T_res0_C - self.Tf_in_C)  # J
        print(f"HiP at {self.T_res0_C} C = {self.HiP:.2e} J")


        return self






# ========================================================
# Gringarten Fracture Cooling Model
# ========================================================


class GringartenFractureModel:
    def __init__(self, params_common, params_gg):
        self.params = params_common.copy()
        self.params.update(params_gg)  # Update with Gringarten-specific parameters
        
        rc = ReservoirConfig()
        cfg = rc.make_cfg(self.params)
        self.cfg = cfg

        # --- 1. Flow Rate Calculation ---
        # The paper uses Q as volumetric flow per unit length (y-direction).
        # To match the paper's example (approx 145 kg/s total flow):
        # We need a higher velocity if using a thin fracture, or define Q directly.
        # Let's trust the user's v and w, but print the resulting Q to check magnitude.
        
        # Q [m^2/s] = v [m/s] * aperture [m]
        # aperture = 2 * w (half-width)
        self.Q_unit = params_gg['v'] * (2 * params_gg['w'])
        
        # --- 2. Thermal Properties Group ---
        # Group = (rho_w * cw)^2 / (K_r * rho_r * cr)
        # Watch out for units! Everything must be SI (J, kg, m, s).
        
        K_R = self.params['k'] # thermal CONDUCTIVITY-- confirm
        rho_c_R = self.params['rho_r'] * self.params['c_r']
        rho_c_W = self.params['rho_f'] * self.params['c_f']
        
        self.material_group = (rho_c_W**2) / (K_R * rho_c_R)

        self.HiP  = cfg.V_res_m3 * cfg.rho_r * cfg.cp_r * (cfg.T_res0_C - cfg.Tf_in_C)  # J

        # print(f"DEBUG INFO:")
        # print(f"  Q (per unit y)   : {self.Q_unit:.2e} m²/s")
        # print(f"  Material Group   : {self.material_group:.2e}")

    def solve(self):
        times_sec = np.array(self.params['times_s'])
        dt = times_sec[1]-times_sec[0]
        z = self.cfg.L_m # Distance along fracture height (z-direction in paper)
        
        # --- 3. Dimensionless Time Calculation (t_D') ---
        # Eq 10: t_D' = [ (rho_w*cw)^2 / K_R*rho_r*c_r ] * (Q/z)^2 * t
        
        geom_factor = (self.Q_unit / z)**2
        t_D_prime = self.material_group * geom_factor * times_sec
        
        print(f"  Max t_D' value   : {np.max(t_D_prime):.2e} (Should be > 0.1 to see cooling)")

        # --- 4. Dimensionless Temperature (T_WD) ---
        # Eq A18: T_WD = erfc( 1 / (2 * sqrt(t_D')) )
        
        T_WD = np.zeros_like(t_D_prime)
        
        # Avoid div by zero
        valid_mask = t_D_prime > 1e-15 
        
        # Calculate argument
        # If t_D' is extremely small, arg is extremely large -> erfc(large) -> 0
        arg = 1.0 / (2.0 * np.sqrt(t_D_prime[valid_mask]))
        T_WD[valid_mask] = erfc(arg)
        
        # --- 5. Redimensionalize ---
        # T_WD = (T_rock_0 - T_out) / (T_rock_0 - T_inlet)
        T_rock_0 = self.cfg.T_res0_C #params['T_rock_init']
        T_inlet = self.cfg.Tf_in_C #params['T_inlet']
        delta_T = T_rock_0 - T_inlet
        
        T_fluid_out = T_rock_0 - (T_WD * delta_T) # check this?

        # 3. Calculate Total Flow Rate (for reference)
        # Q_total = Q_per_unit_length * fracture_length
        Q_total = self.Q_unit * self.cfg.L_m * self.params['N_fracs'] # m^3/s

        print(f"Q_unit: Unit Flow Rate: {self.Q_unit:.4f} m³/s")
        print(f"Total Volumetric Flow Rate: {Q_total:.4f} m³/s")
        print(f"Total Mass Flow Rate: {Q_total * self.cfg.rho_f:.2f} kg/s")
        # 4. Calculate Total Enthalpy Rate (Thermal Power) over time
        # Power [Watts] = rho * c * Q_total * (T_out - T_in)
        # Note: This calculates power relative to the injection temperature.
        total_power_watts = self.cfg.rho_f * self.cfg.cp_f * Q_total * (T_fluid_out - T_inlet)
        # integrate over time to get total energy extracted:
        H_J_tt = np.cumsum(total_power_watts*dt) # 
        # Recovery_tt = H_J_tt / BD.cfg.U_th_0 # dimensionless
        Recovery_tt = H_J_tt / self.HiP
        
        self.results = {
            'times_years': times_sec / (365.25 * 24 * 3600),
            't_D_prime': t_D_prime,
            'T_WD': T_WD,
            'T_fluid_out': T_fluid_out,
            'Q_total_m3_per_s': Q_total,
            'total_power_MW': total_power_watts / 1e6  # Convert to MW
        }
        return self.results



# ================================================================================
# POROUS FLOW MODEL (Bodvarsson 1972 + Darcy)
# ================================================================================


import numpy as np
from iapws import IAPWS95 




class BodvarssonDarcy:
         
        # def __init__(self,d_init):
        #     self.cfg = cfg

        def compute(self,params_common, params_bd):
            rc = ReservoirConfig()
            cfg = rc.make_cfg(params_common)
            self.cfg = cfg
            # fluid flow: 
            dP_dx = params_bd['dP_Pa'] / cfg.L_m
            vf_ms = (params_bd['K_m2'] / cfg.eta_f) * dP_dx  # m/s
            # ADDED PHI ! ! ! 
            qf_kgs = vf_ms * params_bd['phi'] * cfg.rho_f  # m/s* kg/m^3 = kg/s/m^2
            Qf_kgs = params_bd['phi2'] * cfg.A_m2 * qf_kgs  # kg/s

            # energy flow: 
            # advected enthalpy rate Qh
            dT = (cfg.T_res0_C - cfg.Tf_in_C)
            cp_f = cfg.cp_f# or maybe it is cfg.fluid[2] ? ? 
            Qh_W = Qf_kgs * cp_f * dT  # [kg/s * J/kg/K * K] = W = J/s

            # temp field advection
            Wh_ms = (qf_kgs / cfg.rho_f) * ( cfg.cp_f / cfg.cp_r )  # 
            t_R_s = cfg.L_m / Wh_ms  # s
            #H_tot = Qf_kgs * cfg.cp_f * dT * t_R_s 
            H_tot = Qh_W * t_R_s 

            U_th_0  = (cfg.A_m2 * cfg.L_m ) * cfg.rho_r * cfg.cp_r * dT # (cfg.T_res0_C - cfg.Tf_in_C)  # J
            fR = H_tot / U_th_0

            s_yr = 365*24*3600

            self.dP_dx = dP_dx
            self.vf_ms = vf_ms
            self.qf_kgs = qf_kgs
            self.Qf_kgs = Qf_kgs
            self.dT = dT
            self.Qh_W = Qh_W
            self.Wh_ms = Wh_ms
            self.t_R_s = t_R_s
            self.H_tot = H_tot
            self.U_th_0 = U_th_0
            self.fR = fR 
            self.fR_yrly = (fR/t_R_s)*s_yr

            return self
# ======================================



def pulse_rlx_analytical_soln(x, t, kappa, T0, Ts):
    """
    Analytical solution for semi-infinite diffusion.
    
    Parameters:
    -----------
    x : array
        Distance from surface [m]
    t : float
        Time [s]
    kappa : float
        Thermal diffusivity [m²/s]
    T0 : float
        Far-field (initial) temperature [°C]
    Ts : float
        Surface temperature (held constant) [°C]
    
    Returns:
    --------
    T : array
        Temperature at position x, time t [°C]
    """
    
    if t == 0:
        T = np.where(x == 0, Ts, T0)
        return T
    
    # Dimensionless parameter
    eta = x / (2.0 * np.sqrt(kappa * t))
    
    # Temperature
    T = T0 + (Ts - T0) * erfc(eta)
    
    return T

