import sys
import os
import asyncio
import numpy as np

# Add current dir to path to import local modules
sys.path.append(os.getcwd())

from server.environment import LabTriageEnvironment
from models import SciHypothesisAction

async def test_env():
    env = LabTriageEnvironment()
    
    print("--- Testing Reset (Task 1) ---")
    obs = env.reset(task_id=1)
    print(f"Task: {obs.task_id}")
    print(f"Budget Remaining: ${obs.budget_remaining}")
    # Task 1 budget was increased to $700
    assert obs.budget_remaining == 700.0
    
    print("\n--- Testing Experiment (Room Temp, Many Points) ---")
    action = SciHypothesisAction(
        action_type="run_experiment",
        temperature=298.0,
        concentration=0.5,
        time_points=[0, 10, 20, 30, 40, 50],
        metadata={"episode_id": obs.metadata["episode_id"]}
    )
    obs = env.step(action)
    print(f"Cost: {obs.budget_spent}")
    print(f"Budget Remaining: {obs.budget_remaining}")
    print(f"Hints: {obs.analysis_hints}")
    # Base 50 + Many points 50 = 100
    assert obs.budget_spent == 100.0
    assert obs.analysis_hints is not None
    
    print("\n--- Testing High Temp Experiment ---")
    action = SciHypothesisAction(
        action_type="run_experiment",
        temperature=360.0, # High temp
        concentration=0.5,
        time_points=[0, 10, 20],
        metadata={"episode_id": obs.metadata["episode_id"]}
    )
    obs = env.step(action)
    print(f"New Budget Spent: {obs.budget_spent}")
    # Previous 100 + (High Temp 150) = 250
    assert obs.budget_spent == 250.0

    print("\n--- Testing High Conc Experiment ---")
    action = SciHypothesisAction(
        action_type="run_experiment",
        temperature=298.0,
        concentration=1.5, # High conc
        time_points=[0, 10, 20],
        metadata={"episode_id": obs.metadata["episode_id"]}
    )
    obs = env.step(action)
    print(f"New Budget Spent: {obs.budget_spent}")
    # Previous 250 + (Base 50 + High Conc 30) = 330
    assert obs.budget_spent == 330.0

    print("\n--- Testing Conclusion and Reward ---")
    # Fetch ground truth for validation
    primary_config = env._config[env._primary_target]
    action = SciHypothesisAction(
        action_type="conclude",
        conclusion="I have analyzed Sensor " + env._primary_target + ". Sensors A, B, and C were considered.",
        final_order=primary_config.order,
        final_k=primary_config.k,
        final_activation_energy=primary_config.activation_energy if obs.task_id == 3 else None,
        metadata={"episode_id": obs.metadata["episode_id"]}
    )
    obs = env.step(action)
    print(f"Final Reward: {obs.reward}")
    print(f"Score Breakdown: {obs.score_breakdown}")
    # Reward should be high since we used correct answers
    assert obs.reward > 0.4 
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    asyncio.run(test_env())
