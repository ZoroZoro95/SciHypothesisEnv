import math

def compute_rewards(
    task_id: int,
    final_order: int,
    final_k: float,
    final_activation_energy: float | None,
    true_order: int,
    true_k: float,
    true_activation_energy: float,
    steps_used: int,
    max_steps: int
) -> tuple[float, dict]:

    # Component 1: Order correctness (0 or 0.5)
    order_score = 0.5 if final_order == true_order else 0.0

    # Component 2: Rate constant accuracy (0–0.4)
    # Use log scale — being 10x off is not the same as being 1000x off
    if final_k and final_k > 0 and true_k > 0:
        log_error = abs(math.log10(final_k) - math.log10(true_k))
        # Full marks within 0.5 log units (~3x), zero at 2 log units (100x)
        k_score = 0.4 * max(0.0, 1.0 - (log_error / 2.0))
    else:
        k_score = 0.0

    # Component 3: Efficiency bonus (0–0.1)
    steps_fraction = steps_used / max_steps
    efficiency_score = 0.1 * max(0.0, 1.0 - steps_fraction)

    # Component 4: Activation energy (Task 3 only, 0–0.2)
    ea_score = 0.0
    if task_id == 3 and final_activation_energy and final_activation_energy > 0:
        ea_error = abs(final_activation_energy - true_activation_energy) / true_activation_energy
        ea_score = 0.2 * max(0.0, 1.0 - (ea_error / 0.5))
        k_score = k_score * 0.75  # reduce k weight to make room for Ea

    total = round(order_score + k_score + efficiency_score + ea_score, 4)
    total = min(1.0, total)

    breakdown = {
        "order_correct": order_score,
        "k_accuracy": round(k_score, 4),
        "efficiency": round(efficiency_score, 4),
        "activation_energy": round(ea_score, 4),
        "total": total
    }

    return total, breakdown