import numpy as np 

from scipy.integrate import odeint

from dataclasses import dataclass

@dataclass
class ReactionConfig:
    """GroundTruth for one Episode agent never sees this directly"""
    order: int           # 1 or 2
    k: float             # rate constant (/s for 1st order, L/mol/s for 2nd)
    activation_energy: float   # J/mol (used in Task 3)
    k_ref_temp: float    # reference temperature for k (Kelvin)
    noise_level: float   # std dev of Gaussian noise added to readings

#order differential equation (ode)
def first_order_ode(C,t,k):
    """dC/dt = -k * C"""
    return -k*C

#second order ode : where rate is proportional to square of concentration
def second_order_ode(C,t,k):
    """dC/dt = -k * C^2"""
    return -k * C[0]**2

def arrhenius_k(k_ref, Ea, T_ref, T):
    """
    Adjusting the rate constant for temperature using Arrhenius equation.
    k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref)), delta_t = (1/T - 1/T_ref)
    """
    R = 8.314  # J/mol/K , R is the universal gas constant
    return k_ref * np.exp(-Ea / R * (1/T - 1/T_ref))

def run_experiment(
    config: ReactionConfig,
    temperature: float,
    initial_concentration: float,
    time_points : list[float]
) -> list[dict]:
    """
    Simulatse an experiment and return noisy concentration readings.
    This is what the agent calls when it picks run_experiment.
    """
    #adjustig the K for temperature using Arhenius
    k_at_T = arrhenius_k(
        config.k,
        config.activation_energy,
        config.k_ref_temp,
        temperature
    )

    t = np.array(sorted(time_points))
    C0 = [initial_concentration]

    if config.order == 1:
        solution = odeint(first_order_ode, C0, t, args=(k_at_T,)) #The odeint func is used to solve ODEs by numerical integration
        concentrations = solution[:, 0] #extracts the first column of the solution array which contains the concentration values
    else:
        solution = odeint(second_order_ode, C0, t, args=(k_at_T,))
        concentrations = solution[:, 0]

    #adding measurement noise    
    noise = np.random.normal(0, config.noise_level, size=len(concentrations))
    concentrations = np.clip(concentrations + noise, 0, None) #the clip function ensures that the concentration values are not negative
    #return the list of dictionaries for each time point
    return [
        {
            "time": round(float(t[i]), 2),
            "concentration": round(float(concentrations[i]), 6),
            "temperature": temperature
        }
        for i in range(len(t))
    ]
def generate_task_config(task_id:int , seed:int=None)-> ReactionConfig:
    """
    Generate a random but valid reaction config for a given task.
    Each task has different difficulty — more noise, tighter k range, etc.
    """
    rng = np.random.default_rng(seed)

    if task_id == 1:
        #Easy Task : low noise, predicatble order, wide k range
        return ReactionConfig(
            order=int(rng.choice([1, 2])),
            k=float(rng.uniform(0.01, 0.1)),
            activation_energy=float(rng.uniform(40000, 60000)),
            k_ref_temp=298.0,
            noise_level=0.005
        )
    elif task_id == 2:
        #Medium Difficulty : moderate Noise, Narrower K
        return ReactionConfig(
            order=int(rng.choice([1, 2])),
            k=float(rng.uniform(0.005, 0.05)),
            activation_energy=float(rng.uniform(50000, 80000)),
            k_ref_temp=298.0,
            noise_level=0.02
        )
    else:
        #Hard Difficulty Task : high Noise, Must also find the activation Energy
        return ReactionConfig(
            order=int(rng.choice([1, 2])),
            k=float(rng.uniform(0.001, 0.02)),
            activation_energy=float(rng.uniform(60000, 100000)),
            k_ref_temp=298.0,
            noise_level=0.04
        )        
    
    
    

