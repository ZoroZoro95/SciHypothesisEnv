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
from typing import Optional,ClassVar
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from openenv.core.env_server.types import Action, Observation
from .simulator import generate_task_config, run_experiment, ReactionConfig
from .reward import compute_rewards
try:
    from ..models import SciHypothesisAction, SciHypothesisObservation
except ImportError:
    from models import SciHypothesisAction, SciHypothesisObservation

TASK_DESCRIPTIONS = {
    1: (
        "EASY — Reaction Order Identification. "
        "A chemical reaction is occurring in aqueous solution at near-neutral pH. "
        "Run experiments by choosing temperature (270–400 K), "
        "initial concentration (0.001–2.0 mol/L), and time points (seconds). "
        "Determine whether the reaction is 1st or 2nd order "
        "and estimate the rate constant k. "
        "You have 6 experiments. Conclude with final_order and final_k."
    ),
    2: (
        "MEDIUM — Rate Constant Estimation. "
        "A reaction of unknown order is occurring. The rate constant k is small "
        "and data is noisy. "
        "Precisely estimate k within 10% error and confirm the reaction order. "
        "You have 8 experiments. Think carefully about which conditions "
        "are most informative."
    ),
    3: (
        "HARD — Full Kinetic Characterization. "
        "Determine reaction order, rate constant k, AND activation energy Ea. "
        "Data is noisy. You have 12 experiments — use them wisely. "
        "Vary both temperature and concentration strategically. "
        "Conclude with final_order, final_k, and final_activation_energy (J/mol)."
    )
}

MAX_STEPS = {1: 6, 2: 8, 3: 12}

class SciHypothesisEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Class-level session store — survives across HTTP requests
    _sessions: ClassVar[dict] = {}

    def __init__(self):
        super().__init__()
        self._config: Optional[ReactionConfig] = None
        self._task_id: Optional[int] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._experiment_log: list = []

    def _save_session(self):
        """Persist current state to class-level store."""
        if self._episode_id:
            SciHypothesisEnvironment._sessions[self._episode_id] = {
                "config": self._config,
                "task_id": self._task_id,
                "step_count": self._step_count,
                "done": self._done,
                "experiment_log": self._experiment_log,
            }

    def _load_session(self, episode_id: str) -> bool:
        """Load state from class-level store. Returns True if found."""
        session = SciHypothesisEnvironment._sessions.get(episode_id)
        if session:
            self._config = session["config"]
            self._task_id = session["task_id"]
            self._step_count = session["step_count"]
            self._done = session["done"]
            self._experiment_log = session["experiment_log"]
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

        self._config = generate_task_config(
            self._task_id,
            seed=seed or np.random.randint(0, 99999)
        )

        # Save so HTTP step can find it
        self._save_session()

        return SciHypothesisObservation(
            task_id=self._task_id,
            task_description=TASK_DESCRIPTIONS[self._task_id],
            current_step=0,
            max_steps=MAX_STEPS[self._task_id],
            experiments_remaining=MAX_STEPS[self._task_id],
            known_context=(
                "The reaction occurs in aqueous solution. "
                "Concentration is in mol/L, time in seconds, "
                "temperature in Kelvin. "
                "Valid temperature range: 270–400 K. "
                "Valid concentration range: 0.001–2.0 mol/L."
            ),
            done=False,
            reward=None,
            # Pass episode_id in metadata so client can send it back
            metadata={"episode_id": self._episode_id}
        )

    def step(
        self,
        action: SciHypothesisAction,
        timeout_s: Optional[float] = None,
        **kwargs
    ) -> SciHypothesisObservation:

        # Try to load session from episode_id in action metadata
        episode_id = (action.metadata or {}).get("episode_id")
        if episode_id and self._task_id is None:
            self._load_session(episode_id)

        # Final guard
        if self._task_id is None:
            raise ValueError(
                "No active session. Call reset() first and pass "
                "episode_id in action metadata."
            )

        if self._done:
            raise ValueError("Episode done. Call reset() to start a new one.")

        self._step_count += 1
        remaining = MAX_STEPS[self._task_id] - self._step_count

        if action.action_type == "run_experiment":
            temp = max(270.0, min(400.0, action.temperature or 298.0))
            conc = max(0.001, min(2.0, action.concentration or 1.0))
            times = action.time_points or [0, 30, 60, 120, 300]

            data = run_experiment(self._config, temp, conc, times)
            self._experiment_log.append({
                "step": self._step_count,
                "temperature": temp,
                "concentration": conc,
                "data": data
            })
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                task_description=TASK_DESCRIPTIONS[self._task_id],
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=remaining,
                experimental_data=data,
                done=False,
                reward=0.0,
                metadata={"episode_id": self._episode_id}
            )

        elif action.action_type == "propose_hypothesis":
            feedback = (
                f"Hypothesis recorded: '{action.hypothesis_text}'. "
                f"Predicted order: {action.predicted_order}, "
                f"Predicted k: {action.predicted_k}. "
                f"{remaining} steps remaining."
            )
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                task_description=TASK_DESCRIPTIONS[self._task_id],
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=remaining,
                hypothesis_feedback=feedback,
                done=False,
                reward=0.0,
                metadata={"episode_id": self._episode_id}
            )

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
                max_steps=MAX_STEPS[self._task_id]
            )
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                task_description=TASK_DESCRIPTIONS[self._task_id],
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=0,
                final_feedback=(
                    f"Episode complete. "
                    f"True order: {self._config.order}, "
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
            experiments_run=len(self._experiment_log)
        )