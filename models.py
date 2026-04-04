# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Sci Hypothesis Env Environment.

The sci_hypothesis_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Literal,Optional

class SciHypothesisAction(Action):
    """
    The agent can do one of 3 things each step:
    1. run_experiment  — query the simulator with conditions
    2. propose_hypothesis — state what it currently believes
    3. conclude — final answer, ends the episode
    """

    action_type: Literal["run_experiment", "propose_hypothesis", "conclude"]
    
    #used when action_type is == run_experiment
    temperature : Optional[float] = None #optional[float] means it can be float or None | Kelvin, valid: 270–400
    concentration : Optional[float] = None  # mol/L, valid: 0.001–2.0
    time_points : Optional[list[float]] = None # seconds, e.g. [0,10,30,60,120]

    #used when action_type is == propose_hypothesis
    hypothesis_text : Optional[str] = None #free text beleif system
    predicted_order : Optional[int] = None # 1 or 2
    predicted_k : Optional[float] = None #estimated value of rate constant

    #used when action_type == conclude
    conclusion : Optional[str] = None #Reasoning Summary
    final_order : Optional[int] = None # 1 or 2
    final_k : Optional[float] = None #estimated value of rate constant
    final_activation_energy : Optional[float] = None  # J/mol (Task 3 only)



class SciHypothesisObservation(Observation):
    """ What the Agnet sees after each action"""

    # Always present
    task_id: int                        # 1=easy, 2=medium, 3=hard
    task_description: str               # natural language description
    current_step: int                   # how many steps taken so far
    max_steps: int                      # budget (varies by task)
    experiments_remaining: int          # budget remaining

    # After run_experiment
    experimental_data : Optional[list[dict]] = None # 
    # e.g. [{"time": 0, "concentration": 1.0},
    #        {"time": 30, "concentration": 0.55}]

    # present after propose_hypothesis
    hypothesis_feedback : Optional[str] = None
    # e.g. "Hypothesis recorded. 4 experiments remaining."

    #present at episode start (hint about whats knowable)
    known_context: Optional[str] = None
    # e.g. "This is an aqueous reaction at near-neutral pH."

    # Present in all responses to match Environment interface
    done: bool = False
    reward: Optional[float] = None

    # Present when done=True
    final_feedback: Optional[str] = None

    # Scoring breakdown (visible after conclude)
    score_breakdown: Optional[dict] = None
    # e.g. {"order_correct": 0.5, "k_accuracy": 0.38, "efficiency": 0.12}
    
