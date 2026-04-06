import numpy as np 

from scipy.integrate import odeint

from dataclasses import dataclass

@dataclass
class ReactionConfig:
    """GroundTruth for one Episode agent never sees this directly"""
    order: int           # 1, 2, or 3 (Reversible)
    k: float             # rate constant (/s for 1st, L/mol/s for 2nd)
    activation_energy: float   # J/mol (used in Task 3)
    k_ref_temp: float    # reference temperature for k (Kelvin)
    noise_level: float   # std dev of Gaussian noise added to readings
    k_reverse: float = 0.0      # reverse rate constant (for type 3)
    kr_ref_temp: float = 298.0  # ref temp for k_reverse
    scenario: str = "Generic"   # Scenario name for internal tracking

#order differential equation (ode)
def first_order_ode(C,t,k):
    """dC/dt = -k * C"""
    return -k*C

#second order ode : where rate is proportional to square of concentration
def second_order_ode(C,t,k):
    """dC/dt = -k * C^2"""
    return -k * C[0]**2

#reversible order ode
def reversible_ode(C,t,kf,kr,C0):
    """dC/dt = -kf*C + kr*(C0 - C)"""
    return -kf*C + kr*(C0 - C)

def arrhenius_k(k_ref, Ea, T_ref, T):
    """
    Adjusting the rate constant for temperature using Arrhenius equation.
    k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
    """
    R = 8.314  # J/mol/K
    return k_ref * np.exp(-Ea / R * (1/T - 1/T_ref))

from scipy.optimize import curve_fit

def compute_hints(experimental_data: list[dict], current_temp: float, history: list[dict] = None, actual_k_at_T: float = None) -> dict:
    """
    Compute metrics to help the agent using direct ODE curve fitting.
    Fits each kinetic model (1st, 2nd, reversible) to the data and picks the best.
    This is robust to saturation, high noise, and all reaction orders.
    """
    if not experimental_data or len(experimental_data) < 3:
        return {}

    t = np.array([d["time"] for d in experimental_data], dtype=float)
    C = np.array([d["concentration"] for d in experimental_data], dtype=float)
    C = np.maximum(C, 1e-10)
    C0 = C[0]

    results = {}

    # --- 1st order fit: C(t) = C0 * exp(-k*t) ---
    try:
        def model_1st(t, k):
            return C0 * np.exp(-k * t)
        popt1, _ = curve_fit(model_1st, t, C, p0=[0.01], bounds=(1e-8, 10), maxfev=300, ftol=1e-4, xtol=1e-4)
        k1 = abs(popt1[0])
        C_pred1 = model_1st(t, k1)
        ss_res = np.sum((C - C_pred1)**2)
        ss_tot = np.sum((C - np.mean(C))**2)
        r2_1 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[1] = {"k": k1, "r2": max(0.0, r2_1)}
    except Exception:
        results[1] = {"k": 0.01, "r2": 0.0}

    # --- 2nd order fit: C(t) = C0 / (1 + k*C0*t) ---
    try:
        def model_2nd(t, k):
            return C0 / (1.0 + k * C0 * t)
        popt2, _ = curve_fit(model_2nd, t, C, p0=[0.01], bounds=(1e-8, 10), maxfev=300, ftol=1e-4, xtol=1e-4)
        k2 = abs(popt2[0])
        C_pred2 = model_2nd(t, k2)
        ss_res = np.sum((C - C_pred2)**2)
        ss_tot = np.sum((C - np.mean(C))**2)
        r2_2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[2] = {"k": k2, "r2": max(0.0, r2_2)}
    except Exception:
        results[2] = {"k": 0.01, "r2": 0.0}

    # --- Reversible fit: C(t) = Ceq + (C0-Ceq)*exp(-(kf+kr)*t) ---
    try:
        def model_rev(t, kf, kr):
            Ceq = C0 * kr / (kf + kr)
            return Ceq + (C0 - Ceq) * np.exp(-(kf + kr) * t)
        popt3, _ = curve_fit(model_rev, t, C, p0=[0.01, 0.005],
                             bounds=([1e-8, 1e-8], [10, 10]), maxfev=300, ftol=1e-4, xtol=1e-4)
        kf, kr = abs(popt3[0]), abs(popt3[1])
        C_pred3 = model_rev(t, kf, kr)
        ss_res = np.sum((C - C_pred3)**2)
        ss_tot = np.sum((C - np.mean(C))**2)
        r2_3 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[3] = {"k": kf, "r2": max(0.0, r2_3)}
    except Exception:
        results[3] = {"k": 0.01, "r2": 0.0}

    # Pick best fitting model
    best_order = max(results, key=lambda o: results[o]["r2"])
    best_k = results[best_order]["k"]
    best_r2 = results[best_order]["r2"]

    hints = {
        "linearity_1st_order": round(results[1]["r2"], 3),
        "linearity_2nd_order": round(results[2]["r2"], 3),
        "fit_reversible": round(results[3]["r2"], 3),
        "suggested_order": best_order,
        "estimated_k": round(float(best_k), 6),
        "temperature": current_temp
    }

    # Arrhenius Analysis: use actual k_at_T (from ground truth Arrhenius eq, not exposed to agent)
    # or fall back to curve_fit best_k if actual_k_at_T not provided.
    k_for_arrhenius = actual_k_at_T if actual_k_at_T is not None else best_k

    if history and len(history) >= 1:
        # Always include current experiment
        all_results = history + [{"k": k_for_arrhenius, "T": current_temp}]

        # Group by temperature and average
        temp_groups = {}
        for h in all_results:
            t_val = round(float(h["T"]), 1)
            if t_val not in temp_groups:
                temp_groups[t_val] = []
            temp_groups[t_val].append(h["k"])

        if len(temp_groups) >= 2:
            temps = np.array(list(temp_groups.keys()))
            ks = np.array([np.mean(temp_groups[t]) for t in temp_groups.keys()])

            invT = 1.0 / temps
            lnk = np.log(np.maximum(ks, 1e-10))

            ea_slope, ea_intercept = np.polyfit(invT, lnk, 1)
            R_gas = 8.314
            suggested_ea = -ea_slope * R_gas

            lnk_pred = ea_slope * invT + ea_intercept
            denom = np.sum((lnk - np.mean(lnk))**2)
            ea_r2 = 1 - (np.sum((lnk - lnk_pred)**2) / denom) if denom > 0 else 1.0

            hints["suggested_ea"] = round(float(suggested_ea), 1)
            hints["ea_confidence"] = round(float(ea_r2), 3)

    return hints

def run_experiment(
    config: ReactionConfig,
    temperature: float,
    initial_concentration: float,
    time_points : list[float]
) -> list[dict]:
    """
    Simulates an experiment and return noisy concentration readings.
    """
    k_at_T = arrhenius_k(
        config.k,
        config.activation_energy,
        config.k_ref_temp,
        temperature
    )

    t = np.array(sorted(time_points))
    C0 = initial_concentration

    if config.order == 1:
        solution = odeint(first_order_ode, [C0], t, args=(k_at_T,))
        concentrations = solution[:, 0]
    elif config.order == 2:
        solution = odeint(second_order_ode, [C0], t, args=(k_at_T,))
        concentrations = solution[:, 0]
    else: # Reversible
        kr_at_T = arrhenius_k(
            config.k_reverse,
            config.activation_energy, # use same Ea for simplicity or adjust?
            config.kr_ref_temp,
            temperature
        )
        solution = odeint(reversible_ode, [C0], t, args=(k_at_T, kr_at_T, C0))
        concentrations = solution[:, 0]

    #adding measurement noise    
    noise = np.random.normal(0, config.noise_level, size=len(concentrations))
    concentrations = np.clip(concentrations + noise, 0, None) 
    
    return [
        {
            "time": round(float(t[i]), 2),
            "concentration": round(float(concentrations[i]), 6),
            "temperature": temperature
        }
        for i in range(len(t))
    ]

def generate_task_config(task_id: int, seed: int = None) -> ReactionConfig:
    """
    Generate a mechanistically diverse reaction config based on real-world scenarios.
    """
    rng = np.random.default_rng(seed)

    if task_id == 1:
        # Scenario: Pharmacokinetics (Antibiotic Clearance)
        # Goal: Identify metabolic order (1st vs 2nd)
        # Reference Temp: 310.15 K (Body Temp)
        # Characteristics: Low noise, predictable, but critical for dosing.
        reaction_type = int(rng.choice([1, 2]))
        return ReactionConfig(
            order=reaction_type,
            k=float(rng.uniform(0.02, 0.08)),
            activation_energy=float(rng.uniform(30000, 45000)),
            k_ref_temp=310.15,
            noise_level=0.003,
            scenario="Pharmacokinetics"
        )
    elif task_id == 2:
        # Scenario: Green Energy (Industrial Carbon Capture)
        # Goal: Characterize Reversible Equilibrium
        # Reference Temp: 298.15 K (Standard)
        # Characteristics: Reversible (Order 3), finding the "plateau".
        return ReactionConfig(
            order=3,
            k=float(rng.uniform(0.01, 0.04)),
            k_reverse=float(rng.uniform(0.005, 0.02)),
            activation_energy=float(rng.uniform(40000, 60000)),
            k_ref_temp=298.15,
            noise_level=0.015,
            scenario="Carbon Capture"
        )
    else: # task_id == 3
        # Scenario: Propellant Stability (Rocket Fuel)
        # Goal: High-precision k and Activation Energy (Ea)
        # Reference Temp: 350.0 K (Operating Temp)
        # Characteristics: High noise, high sensitivity to temperature.
        reaction_type = int(rng.choice([1, 2, 3]))
        return ReactionConfig(
            order=reaction_type,
            k=float(rng.uniform(0.002, 0.01)),
            k_reverse=float(rng.uniform(0.001, 0.005)) if reaction_type == 3 else 0.0,
            activation_energy=float(rng.uniform(70000, 110000)), # High Ea
            k_ref_temp=350.15,
            noise_level=0.045, # High noise
            scenario="Propellant Stability"
        )
