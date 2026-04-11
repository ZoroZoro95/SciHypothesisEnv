# import math

# def compute_rewards(
#     task_id: int,
#     final_order: int,
#     final_k: float,
#     final_activation_energy: float | None,
#     true_order: int,
#     true_k: float,
#     true_activation_energy: float,
#     steps_used: int,
#     max_steps: int
# ) -> tuple[float, dict]:

#     # Component 1: Order correctness (0 or 0.5)
#     order_score = 0.5 if final_order == true_order else 0.0

#     # Component 2: Rate constant accuracy (0–0.4)
#     # Use log scale — being 10x off is not the same as being 1000x off
#     if final_k and final_k > 0 and true_k > 0:
#         log_error = abs(math.log10(final_k) - math.log10(true_k))
#         # Full marks within 0.5 log units (~3x), zero at 2 log units (100x)
#         k_score = 0.4 * max(0.0, 1.0 - (log_error / 2.0))
#     else:
#         k_score = 0.0

#     # Component 3: Efficiency bonus (0–0.1)
#     steps_fraction = steps_used / max_steps
#     efficiency_score = 0.1 * max(0.0, 1.0 - steps_fraction)

#     # Component 4: Activation energy (Task 3 only, 0–0.2)
#     ea_score = 0.0
#     if task_id == 3 and final_activation_energy and final_activation_energy > 0:
#         ea_error = abs(final_activation_energy - true_activation_energy) / true_activation_energy
#         ea_score = 0.2 * max(0.0, 1.0 - (ea_error / 0.5))
#         k_score = k_score * 0.75  # reduce k weight to make room for Ea

#     total = round(order_score + k_score + efficiency_score + ea_score, 4)
#     total = min(1.0, total)

#     breakdown = {
#         "order_correct": order_score,
#         "k_accuracy": round(k_score, 4),
#         "efficiency": round(efficiency_score, 4),
#         "activation_energy": round(ea_score, 4),
#         "total": total
#     }

#     return total, breakdown

import math
from typing import Optional


def score_diagnostic_action(
    action_type: str,
    step_count: int,
    max_steps: int,
    experiment_log: list,
    temperature: Optional[float] = None,
    concentration: Optional[float] = None,
    predicted_order: Optional[int] = None,
    predicted_k: Optional[float] = None,
    target_sensor: Optional[str] = None,
    primary_target: Optional[str] = None,
) -> tuple[float, dict]:
    """
    Per-step reward for run_experiment and propose_hypothesis actions.
    Small signals to guide the agent toward good scientific behavior.
    Returns (reward, breakdown)
    """
    breakdown = {}
    total = 0.0

    if action_type == "run_experiment":

        # 1. First Discovery Bonus - Identifying the threat sensor
        discovery_bonus = 0.0
        if target_sensor == primary_target:
            # Check if we've probed this sensor successfully before
            probed_before = any(
                e.get("target") == target_sensor for e in experiment_log[:-1]
                if e.get("action_type", "run_experiment") == "run_experiment"
            )
            if not probed_before:
                discovery_bonus = 0.08
        breakdown["discovery_bonus"] = round(discovery_bonus, 4)
        total += discovery_bonus

        # 2. Symmetry/Comparison Bonus - Probing different sensors at same conditions
        symmetry_bonus = 0.0
        if temperature and target_sensor:
            # Did we probe a DIFFERENT sensor at this same Temperature before?
            other_sensor_at_T = any(
                e.get("temperature") == temperature and e.get("target") != target_sensor
                for e in experiment_log[:-1]
            )
            if other_sensor_at_T:
                symmetry_bonus = 0.04
        breakdown["symmetry_bonus"] = round(symmetry_bonus, 4)
        total += symmetry_bonus

        # 3. Novelty bonus
        past_temps = [e["temperature"] for e in experiment_log[:-1] if "temperature" in e]
        temp_novel = temperature not in past_temps
        novelty = 0.03 if temp_novel else 0.0
        breakdown["novelty"] = round(novelty, 4)
        total += novelty

        # 4. Repetition penalty
        repeat_penalty = 0.0
        exact_match = any(
            e.get("target") == target_sensor and
            e.get("temperature") == temperature and
            e.get("concentration") == concentration
            for e in experiment_log[:-1]
        )
        if exact_match:
            repeat_penalty = -0.05
        breakdown["repeat_penalty"] = round(repeat_penalty, 4)
        total += repeat_penalty

    elif action_type == "propose_hypothesis":

        # 1. Timing bonus — proposing early means agent is confident
        timing_fraction = step_count / max_steps
        # Best reward for proposing in first 60% of steps
        timing_bonus = 0.03 * max(0.0, 1.0 - timing_fraction)
        breakdown["timing_bonus"] = round(timing_bonus, 4)
        total += timing_bonus

        # 2. Completeness — did agent provide both order and k?
        completeness = 0.0
        if predicted_order is not None:
            completeness += 0.02
        if predicted_k is not None and predicted_k > 0:
            completeness += 0.02
        breakdown["completeness"] = round(completeness, 4)
        total += completeness

    total = round(total, 4)
    breakdown["step_total"] = total
    return total, breakdown


def calculate_triage_score(
    task_id: int,
    final_order: int,
    final_k: float,
    final_activation_energy: Optional[float],
    conclusion_text: str,
    true_order: int,
    true_k: float,
    true_activation_energy: float,
    primary_target: str,
    available_sensors: list[str],
    steps_used: int,
    max_steps: int,
    budget_spent: float,
    total_budget: float,
    accumulated_step_rewards: float = 0.0
) -> tuple[float, dict]:
    """
    Final episode reward on conclude action.
    Combines scientific accuracy + efficiency + accumulated step rewards.
    All components normalized so total is 0.0–1.0
    """

    # Component 1: Order correctness (0.25)
    order_score = 0.25 if final_order == true_order else 0.0

    # Component 2: Rate constant accuracy (0.20) log scale
    k_score = 0.0
    if final_k and final_k > 0 and true_k > 0:
        log_error = abs(math.log10(final_k) - math.log10(true_k))
        k_score = 0.20 * max(0.0, 1.0 - (log_error / 2.0))

    # Component 3: Activation energy Task 3 & 4 (0.15)
    ea_score = 0.0
    if task_id in [3, 4] and final_activation_energy and final_activation_energy > 0:
        ea_error = abs(final_activation_energy - true_activation_energy) / true_activation_energy
        ea_score = 0.15 * max(0.0, 1.0 - (ea_error / 0.5))

    # Component 4: Reasoning & Noise Filtering (0.30)
    reasoning_score = 0.0
    conclusion_lower = conclusion_text.lower()
    
    # Keyword Bonuses (0.15)
    keywords = {
        "stable": 0.03, "inert": 0.03, "constant": 0.03, "noise": 0.03,
        "arrhenius": 0.03, "activation energy": 0.03, "decay": 0.03,
        "reversible": 0.04, "plateau": 0.04, "equilibrium": 0.04
    }
    for kw, val in keywords.items():
        if kw in conclusion_lower:
            reasoning_score += val
    
    # Noise Filtering Bonus (0.15)
    # Check if agent explicitly dismissed other sensors
    other_sensors = [s for s in available_sensors if s != primary_target]
    dismissal_count = sum(1 for s in other_sensors if f"sensor {s.lower()}" in conclusion_lower)
    
    if dismissal_count >= len(other_sensors):
        reasoning_score += 0.15
    elif dismissal_count > 0:
        reasoning_score += 0.07
    elif task_id == 1:
        # Give a small mercy bonus for Task 1 if they at least identify the threat
        reasoning_score += 0.03

    # Component 5: Efficiency (0.10)
    budget_frac = (total_budget - budget_spent) / total_budget
    efficiency_score = 0.10 * max(0.0, budget_frac)

    # Component 6: Step bonus (0.10)
    step_bonus = min(0.10, accumulated_step_rewards)

    total = order_score + k_score + ea_score + reasoning_score + efficiency_score + step_bonus
    
    # CRITICAL: Filtering Cap
    # If they didn't mention ANY other sensor, hard cap at 0.50 (Exempt Task 1)
    if dismissal_count == 0 and total > 0.50 and task_id != 1:
        total = 0.50

    total = round(max(0.0011, min(0.9989, total)), 4)

    breakdown = {
        "order_correct": round(order_score, 4),
        "k_accuracy": round(k_score, 4),
        "ea_accuracy": round(ea_score, 4),
        "reasoning_and_filtering": round(reasoning_score, 4),
        "efficiency": round(efficiency_score, 4),
        "step_bonus": round(step_bonus, 4),
        "total": total
    }

    return total, breakdown