import sys
import os
import asyncio
import numpy as np

# Add parent directory to sys.path to allow importing the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sci_hypothesis_env.server.sci_hypothesis_env_environment import SciHypothesisEnvironment
from sci_hypothesis_env.models import SciHypothesisAction

async def test_task_4():
    env = SciHypothesisEnvironment()
    
    print("--- Testing Task 4 (Vitamin C) ---")
    # Test initialization with task_id=4
    obs = env.reset(task_id=4)
    print(f"Task ID: {obs.task_id}")
    print(f"Description: {obs.task_description}")
    
    assert obs.task_id == 4
    assert "Vitamin C" in obs.task_description
    assert obs.budget_remaining == 1000.0
    
    # Test a step
    print("\n--- Testing Step ---")
    action = SciHypothesisAction(
        action_type="run_experiment",
        temperature=343.15,
        concentration=0.5,
        time_points=[0, 30, 60, 120, 300],
        metadata={"episode_id": obs.metadata["episode_id"]}
    )
    
    obs_step = env.step(action)
    print(f"Step 1 Reward: {obs_step.reward}")
    print(f"Budget Remaining: {obs_step.budget_remaining}")
    
    assert obs_step.budget_remaining < 1000.0
    assert obs_step.experimental_data is not None
    
    print("\nTask 4 verification successful!")

if __name__ == "__main__":
    asyncio.run(test_task_4())
