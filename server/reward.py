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


def compute_step_reward(
    action_type: str,
    step_count: int,
    max_steps: int,
    experiment_log: list,
    temperature: Optional[float] = None,
    concentration: Optional[float] = None,
    predicted_order: Optional[int] = None,
    predicted_k: Optional[float] = None,
) -> tuple[float, dict]:
    """
    Per-step reward for run_experiment and propose_hypothesis actions.
    Small signals to guide the agent toward good scientific behavior.
    Returns (reward, breakdown)
    """
    breakdown = {}
    total = 0.0

    if action_type == "run_experiment":

        # 1. Novelty bonus — reward exploring new conditions
        # Check if this temperature has been tried before
        past_temps = [e["temperature"] for e in experiment_log[:-1]]
        past_concs = [e["concentration"] for e in experiment_log[:-1]]

        temp_novel = temperature not in past_temps
        conc_novel = concentration not in past_concs

        novelty = 0.0
        if temp_novel:
            novelty += 0.03
        if conc_novel:
            novelty += 0.02
        breakdown["novelty"] = round(novelty, 4)
        total += novelty

        # 2. Extremity bonus — extreme conditions are more informative
        # Temperature: reward being far from 298K (reference)
        if temperature:
            temp_range = 400 - 270
            temp_extremity = abs(temperature - 298) / (temp_range / 2)
            extremity_bonus = 0.02 * temp_extremity
            breakdown["extremity"] = round(extremity_bonus, 4)
            total += extremity_bonus
        else:
            breakdown["extremity"] = 0.0

        # 3. Repetition penalty — penalize running same condition twice
        repeat_penalty = 0.0
        if not temp_novel and not conc_novel:
            repeat_penalty = 0.0
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


def compute_rewards(
    task_id: int,
    final_order: int,
    final_k: float,
    final_activation_energy: Optional[float],
    true_order: int,
    true_k: float,
    true_activation_energy: float,
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

    # Component 1: Order correctness (0.30)
    order_score = 0.30 if final_order == true_order else 0.0

    # Component 2: Rate constant accuracy (0.25) log scale
    if final_k and final_k > 0 and true_k > 0:
        log_error = abs(math.log10(final_k) - math.log10(true_k))
        # Full marks within 0.5 log units, zero at 2 log units
        k_score = 0.25 * max(0.0, 1.0 - (log_error / 2.0))
    else:
        k_score = 0.0

    # Component 3: Activation energy Task 3 only (0.15)
    ea_score = 0.0
    if task_id == 3:
        if final_activation_energy and final_activation_energy > 0:
            ea_error = abs(
                final_activation_energy - true_activation_energy
            ) / true_activation_energy
            ea_score = 0.15 * max(0.0, 1.0 - (ea_error / 0.5))
        # No adjustment to k_score here as weights are predefined

    # Component 4: Budget efficiency (0.15)
    budget_remaining = max(0.0, total_budget - budget_spent)
    budget_efficiency = 0.15 * (budget_remaining / total_budget)

    # Component 5: Step efficiency (0.05)
    # Remaining steps normalized
    steps_fraction = steps_used / max_steps
    efficiency_score = 0.05 * max(0.0, 1.0 - steps_fraction)

    # Component 6: Accumulated step rewards (0.10 max)
    step_bonus = min(0.10, accumulated_step_rewards)

    total = round(
        order_score + k_score + efficiency_score + ea_score + step_bonus + budget_efficiency, 4
    )
    # total = min(1.0, total)
    total = max(0.0011, min(0.9989, total))

    breakdown = {
        "order_correct": order_score,
        "k_accuracy": round(k_score, 4),
        "ea_accuracy": round(ea_score, 4),
        "budget_efficiency": round(budget_efficiency, 4),
        "step_efficiency": round(efficiency_score, 4),
        "step_bonus": round(step_bonus, 4),
        "total": total
    }

    return total, breakdown