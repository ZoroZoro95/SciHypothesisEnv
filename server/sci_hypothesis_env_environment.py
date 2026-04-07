# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sci Hypothesis Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""
import numpy as np
import uuid
from typing import Optional, ClassVar
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.env_server.types import Action, Observation

try:
    from .simulator import generate_task_config, run_experiment, ReactionConfig
    from .reward import compute_step_reward, compute_rewards
    from ..models import SciHypothesisAction, SciHypothesisObservation
except ImportError:
    from server.simulator import generate_task_config, run_experiment, ReactionConfig
    from server.reward import compute_step_reward, compute_rewards
    from models import SciHypothesisAction, SciHypothesisObservation

TASK_DESCRIPTIONS = {
    1: (
        "PHARMACOKINETICS — Antibiotic Clearance Challenge. "
        "A Phase I clinical trial is underway for a new aqueous antibiotic. "
        "Your goal is to determine if the drug is cleared via 1st or 2nd order kinetics "
        "to ensure safe patient dosing. Budget: $500. "
        "Note: Body temperature is 310.15 K. Start there for baseline data."
    ),
    2: (
        "GREEN ENERGY — Carbon Capture Equilibrium. "
        "You are optimizing a reversible CO2 absorption reaction for industrial scrubbers. "
        "Unlike standard reactions, this one reaches a plateau (equilibrium). "
        "Characterize the forward and reverse balance (Order 3). Budget: $800. "
        "The reaction is sensitive to noise, so collect enough points to see the plateau."
    ),
    3: (
        "AEROSPACE — Propellant Thermal Stability. "
        "A high-energy rocket propellant is being tested for long-term storage safety. "
        "You must determine the reaction order, k, AND the activation energy (Ea). "
        "Data is extremely noisy. Precision is critical for mission success. Budget: $1200. "
        "You MUST vary temperature strategically to build an Arrhenius model."
    )
}

MAX_STEPS = {1: 5, 2: 6, 3: 8}
TASK_BUDGETS = {1: 500, 2: 800, 3: 1200}


class SciHypothesisEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True
    _sessions: ClassVar[dict] = {}

    def __init__(self):
        super().__init__()
        self._config: Optional[ReactionConfig] = None
        self._task_id: Optional[int] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._experiment_log: list = []
        self._accumulated_step_rewards: float = 0.0
        self._budget_spent: float = 0.0

    def _save_session(self):
        if self._episode_id:
            SciHypothesisEnvironment._sessions[self._episode_id] = {
                "config": self._config,
                "task_id": self._task_id,
                "step_count": self._step_count,
                "done": self._done,
                "experiment_log": self._experiment_log,
                "accumulated_step_rewards": self._accumulated_step_rewards,
                "budget_spent": self._budget_spent,
            }

    def _load_session(self, episode_id: str) -> bool:
        session = SciHypothesisEnvironment._sessions.get(episode_id)
        if session:
            self._config = session["config"]
            self._task_id = session["task_id"]
            self._step_count = session["step_count"]
            self._done = session["done"]
            self._experiment_log = session["experiment_log"]
            self._accumulated_step_rewards = session.get("accumulated_step_rewards", 0.0)
            self._budget_spent = session.get("budget_spent", 0.0)
            self._episode_id = episode_id
            return True
        return False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs
    ) -> SciHypothesisObservation:
        task_id = kwargs.get("task_id", None)
        self._task_id = task_id or int(np.random.choice([1, 2, 3]))
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._experiment_log = []
        self._accumulated_step_rewards = 0.0
        self._budget_spent = 0.0

        self._config = generate_task_config(
            self._task_id,
            seed=seed or np.random.randint(0, 99999)
        )
        self._save_session()

        return SciHypothesisObservation(
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            current_step=0,
            max_steps=MAX_STEPS[self._task_id],
            experiments_remaining=MAX_STEPS[self._task_id],
            budget_remaining=float(TASK_BUDGETS[self._task_id]),
            budget_spent=0.0,
            known_context=(
                "The reaction occurs in aqueous solution. "
                "Types: 1=1st order, 2=2nd order, 3=Reversible. "
                "Valid temperature range: 270–400 K. "
                "Valid concentration range: 0.001–2.0 mol/L."
            ),
            done=False,
            reward=None,
            metadata={"episode_id": self._episode_id}
        )

    def _calculate_experiment_cost(self, temp: float, conc: float, times: list[float]) -> float:
        cost = 50.0 # Base
        if temp > 350.0:
            cost = 150.0
        elif temp < 280.0:
            cost = 120.0
        
        # High concentration premium
        if conc > 1.0:
            cost += 30.0
            
        # Long/Many points premium
        if len(times) > 5 or max(times) > 500:
            cost += 50.0
            
        return cost

    def step(
        self,
        action: SciHypothesisAction,
        timeout_s: Optional[float] = None,
        **kwargs
    ) -> SciHypothesisObservation:

        # Load session from episode_id in action metadata
        episode_id = (action.metadata or {}).get("episode_id")
        if episode_id and self._task_id is None:
            self._load_session(episode_id)

        if self._task_id is None:
            raise ValueError(
                "No active session. Call reset() first and pass "
                "episode_id in action metadata."
            )

        if self._done:
            raise ValueError("Episode done. Call reset() to start a new one.")

        self._step_count += 1
        remaining_steps = MAX_STEPS[self._task_id] - self._step_count
        total_budget = float(TASK_BUDGETS[self._task_id])

        # ---- run_experiment ----
        if action.action_type == "run_experiment":
            temp = max(270.0, min(400.0, action.temperature or 298.0))
            conc = max(0.001, min(2.0, action.concentration or 1.0))
            times = action.time_points or [0, 30, 60, 120, 300]

            cost = self._calculate_experiment_cost(temp, conc, times)
            
            if self._budget_spent + cost > total_budget:
                self._save_session()
                return SciHypothesisObservation(
                    task_id=self._task_id,
                    task_description=TASK_DESCRIPTIONS[self._task_id],
                    current_step=self._step_count,
                    max_steps=MAX_STEPS[self._task_id],
                    experiments_remaining=remaining_steps,
                    budget_remaining=total_budget - self._budget_spent,
                    budget_spent=self._budget_spent,
                    hypothesis_feedback=f"INSUFFICIENT BUDGET. Action cost ${cost}, but only ${total_budget - self._budget_spent:.2f} remains.",
                    done=False,
                    reward=-0.05, # Penalty for trying to overspend
                    metadata={"episode_id": self._episode_id}
                )

            self._budget_spent += cost
            data = run_experiment(self._config, temp, conc, times)

            # Compute the true k at this temperature (used internally for Arrhenius hints only)
            from .simulator import arrhenius_k, compute_hints
            actual_k_at_T = arrhenius_k(
                self._config.k,
                self._config.activation_energy,
                self._config.k_ref_temp,
                temp
            )

            # Extract history of (actual_k_at_T, T) from previous experiments
            history = [
                {"k": entry["actual_k_at_T"], "T": entry["temperature"]}
                for entry in self._experiment_log
                if "actual_k_at_T" in entry
            ]

            hints = compute_hints(data, temp, history=history, actual_k_at_T=actual_k_at_T)

            self._experiment_log.append({
                "step": self._step_count,
                "temperature": temp,
                "concentration": conc,
                "cost": cost,
                "actual_k_at_T": actual_k_at_T,
                "data": data,
                "hints": hints
            })

            # Per-step reward AFTER appending to log
            step_reward, step_breakdown = compute_step_reward(
                action_type="run_experiment",
                step_count=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiment_log=self._experiment_log,
                temperature=temp,
                concentration=conc,
            )
            self._accumulated_step_rewards += step_reward
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                task_description=TASK_DESCRIPTIONS[self._task_id],
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=remaining_steps,
                budget_remaining=total_budget - self._budget_spent,
                budget_spent=self._budget_spent,
                experimental_data=data,
                analysis_hints=hints,
                done=False,
                reward=step_reward,
                metadata={
                    "episode_id": self._episode_id,
                    "step_breakdown": step_breakdown
                }
            )

        # ---- propose_hypothesis ----
        elif action.action_type == "propose_hypothesis":
            step_reward, step_breakdown = compute_step_reward(
                action_type="propose_hypothesis",
                step_count=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiment_log=self._experiment_log,
                predicted_order=action.predicted_order,
                predicted_k=action.predicted_k,
            )
            self._accumulated_step_rewards += step_reward

            feedback = (
                f"Hypothesis recorded: '{action.hypothesis_text}'. "
                f"Predicted order: {action.predicted_order}, "
                f"Predicted k: {action.predicted_k}. "
                f"{remaining_steps} steps remaining."
            )
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                task_description=TASK_DESCRIPTIONS[self._task_id],
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=remaining_steps,
                budget_remaining=total_budget - self._budget_spent,
                budget_spent=self._budget_spent,
                hypothesis_feedback=feedback,
                done=False,
                reward=step_reward,
                metadata={
                    "episode_id": self._episode_id,
                    "step_breakdown": step_breakdown
                }
            )

        # ---- conclude ----
        elif action.action_type == "conclude":
            self._done = True

            reward, breakdown = compute_rewards(
                task_id=self._task_id,
                final_order=action.final_order or 0,
                final_k=action.final_k or 0.0,
                final_activation_energy=action.final_activation_energy,
                true_order=self._config.order,
                true_k=self._config.k,
                true_activation_energy=self._config.activation_energy,
                steps_used=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                budget_spent=self._budget_spent,
                total_budget=total_budget,
                accumulated_step_rewards=self._accumulated_step_rewards
            )
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                task_description=TASK_DESCRIPTIONS[self._task_id],
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=0,
                budget_remaining=total_budget - self._budget_spent,
                budget_spent=self._budget_spent,
                final_feedback=(
                    f"Episode complete. "
                    f"True order: {self._config.order} ({'Reversible' if self._config.order == 3 else 'Normal'}), "
                    f"True k: {round(self._config.k, 6)}, "
                    f"True Ea: {round(self._config.activation_energy, 1)} J/mol. "
                    f"Your score: {breakdown['total']}"
                ),
                score_breakdown=breakdown,
                done=True,
                reward=reward,
                metadata={"episode_id": self._episode_id}
            )

        else:
            raise ValueError(f"Unknown action_type: {action.action_type}")

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            done=self._done,
            experiments_run=len(self._experiment_log),
            budget_spent=self._budget_spent
        )