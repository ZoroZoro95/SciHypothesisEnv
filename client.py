# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sci Hypothesis Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import SciHypothesisAction, SciHypothesisObservation


class SciHypothesisEnv(
    EnvClient[SciHypothesisAction, SciHypothesisObservation, State]
):
    """
    Client for the Sci Hypothesis Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SciHypothesisEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(SciHypothesisAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SciHypothesisEnv.from_docker_image("sci_hypothesis_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SciHypothesisAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SciHypothesisAction) -> Dict:
        """
        Convert SciHypothesisAction to JSON payload for step message.

        Args:
            action: SciHypothesisAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SciHypothesisObservation]:
        """
        Parse server response into StepResult[SciHypothesisObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SciHypothesisObservation
        """
        obs_data = payload.get("observation", {})
        observation = SciHypothesisObservation(
            task_id=obs_data.get("task_id", 1),
            task_description=obs_data.get("task_description", ""),
            current_step=obs_data.get("current_step", 0),
            max_steps=obs_data.get("max_steps", 6),
            experiments_remaining=obs_data.get("experiments_remaining", 0),
            experimental_data=obs_data.get("experimental_data"),
            hypothesis_feedback=obs_data.get("hypothesis_feedback"),
            known_context=obs_data.get("known_context"),
            final_feedback=obs_data.get("final_feedback"),
            score_breakdown=obs_data.get("score_breakdown"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        from openenv.core.client_types import StepResult
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
