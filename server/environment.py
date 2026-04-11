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
    from .graders import score_diagnostic_action, calculate_triage_score
    from ..models import SciHypothesisAction, SciHypothesisObservation
except ImportError:
    from server.simulator import generate_task_config, run_experiment, ReactionConfig
    from server.graders import score_diagnostic_action, calculate_triage_score
    from models import SciHypothesisAction, SciHypothesisObservation

INCIDENT_VARIANTS = {
    1: [
        (
            "🚨 INCIDENT REPORT [VARIANT ALPHA]: EXPERIMENTAL REACTOR GAMMA-7.\n"
            "The Experimental Reactor at the Sector 7 refinery is screaming. A localized heat-exchanger failure "
            "has triggered a rapid, unidirectional decomposition in the primary chamber. You are the Senior "
            "Incident Commander on site. Dashboard sensors A, B, and C are flooding with data, but two are known "
            "red-herring readings from inert coolant loops.\n"
            "Identify the single sensor showing true kinetic degradation and characterize its decay order and "
            "rate constant (k) to initialize emergency quenching procedures.\n"
            "Status: High-Pressure Warning. Budget: $700. Baseline: 310.15 K."
        ),
        (
            "🚨 INCIDENT REPORT [VARIANT BETA]: MEDICAL ISOTOPE CONTAINMENT BREACH.\n"
            "A medical isotope facility reports a cracked storage vial in the Triage Wing. One of the storage "
            "vials containing a decaying tracer is leaking radiation. Multiple 'ghost' sensors (A, B, C) "
            "are being triggered by background gamma flares.\n"
            "As the onsite kinetics expert, you must isolate the leaking sensor and determine if it follows 1st "
            "or 2nd order decay to predict the contaminated fallout radius and evacuation zone.\n"
            "Status: Radiation Leak. Budget: $700."
        )
    ],
    2: [
        (
            "🚨 INCIDENT REPORT [VARIANT GAMMA]: PETROCHEMICAL SCRUBBER RUPTURE.\n"
            "An industrial gas scrubber at the North-South refinery has failed, releasing two toxic byproducts. "
            "Internal monitors A and B are active. One byproduct is a predictable, slow 1st-order decay. "
            "The other is a lethal Reversible (Order 3) equilibrium that will plateau at toxic atmospheric levels.\n"
            "Identify which sensor captures the plateauing threat and characterize its balance constants "
            "before the facility-wide evacuation sirens time out.\n"
            "Status: Atmospheric Leak. Budget: $900."
        ),
        (
            "🚨 INCIDENT REPORT [VARIANT DELTA]: POLYMERIZATION SURGE.\n"
            "A chemical manufacturing line is overheating due to an unintended polymerization surge. "
            "Two potential catalysts are being monitored via sensors A and B. One is a standard linear driver; "
            "the other is a self-terminating Reversible equilibrium that will saturate the line.\n"
            "Deduce which sensor shows the reversible surge to allow the Command Center to adjust the pressure "
            "valves correctly before the gaskets fail.\n"
            "Status: Surge Detected. Budget: $900."
        )
    ],
    3: [
        (
            "🚨 INCIDENT REPORT [VARIANT KAPPA]: SPACEPORT PROPELLANT BUBBLING.\n"
            "Liquid Propellant Storage at Launch Pad 39B is bubbling. High-energy propellants are degrading "
            "spontaneously in the heat. Strategic data is needed to isolate the root cause: is it "
            "'Thermal Breakdown' (High Activation Energy sensitivity) or 'Foreign Dust Contamination' (Flat temperature response)?\n"
            "Monitoring data for Sensor A is messy and noisy. You must perform a precision temperature variance "
            "study to find the Activation Energy (Ea) and save the pre-launch window.\n"
            "Status: T-Minus 2 Hours. Budget: $1200."
        ),
        (
            "🚨 INCIDENT REPORT [VARIANT EPSILON]: BATTERY ARRAY THERMAL RUNAWAY.\n"
            "An experimental battery electrolyte array is bubbling in the test lab. It could be a simple 2nd order "
            "decay or a catastrophic high-Ea Arrhenius event triggered by the room temperature.\n"
            "You are tasked with quantifying the Activation Energy (Ea) of Sensor A to determine if this array "
            "presents a fire risk at standard operating temperatures or requires immediate cryogenic immersion.\n"
            "Status: Overheating Alert. Budget: $1200."
        )
    ],
    4: [
        (
            "🚨 INCIDENT REPORT [VARIANT SIGMA]: GLOBAL SEED VAULT REFRIGERATION FAILURE.\n"
            "The refrigeration system in the Svalbard Seed Vault has failed. Three irreplaceable genetic "
            "vials (A, B, and C) are beginning to warm. Logistics dictate you can only perform intensive quenching "
            "on ONE sample. You must identify which vial has the HIGHEST Activation Energy (Ea).\n"
            "The highest Ea vial is the most critically sensitive to this heat catastrophe. Minimize your "
            "experiments to find this vial and characterize its kinetics for stabilization.\n"
            "Status: Arctic Alert. Budget: $1000."
        ),
        (
            "🚨 INCIDENT REPORT [VARIANT OMEGA]: DEEP-SPACE PROBE 'HORIZON' FUEL TRIAGE.\n"
            "The 'Horizon' deep-space probe reports that its hydrazine tanks are warming due to solar flares. "
            "Three propellant mix variants (A, B, C) are under remote monitoring. To save the probe, you must "
            "identify which variant has the highest sensitivity to heat (Highest Ea).\n"
            "Characterize the kinetics of the highest-risk variant immediately so the probe can re-orient its "
            "solar shields correctly without a total loss of power.\n"
            "Status: Deep Space Emergency. Budget: $1100."
        )
    ]
}

MAX_STEPS = {1: 8, 2: 8, 3: 8, 4: 7}
TASK_BUDGETS = {1: 700, 2: 900, 3: 1200, 4: 1000}


class LabTriageEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True
    _sessions: ClassVar[dict] = {}

    def __init__(self):
        super().__init__()
        self._config: Optional[dict] = None
        self._primary_target: Optional[str] = None
        self._task_id: Optional[int] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._done: bool = False
        self._experiment_log: list = []
        self._accumulated_step_rewards: float = 0.0
        self._budget_spent: float = 0.0

    def _save_session(self):
        if self._episode_id:
            LabTriageEnvironment._sessions[self._episode_id] = {
                "config": self._config,
                "primary_target": self._primary_target,
                "task_id": self._task_id,
                "step_count": self._step_count,
                "done": self._done,
                "experiment_log": self._experiment_log,
                "accumulated_step_rewards": self._accumulated_step_rewards,
                "budget_spent": self._budget_spent,
            }

    def _load_session(self, episode_id: str) -> bool:
        session = LabTriageEnvironment._sessions.get(episode_id)
        if session:
            self._config = session["config"]
            self._primary_target = session.get("primary_target", "A")
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
        self._task_id = task_id or int(np.random.choice([1, 2, 3, 4]))
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
        
        # Pick a random text variant for this task
        variants = INCIDENT_VARIANTS[self._task_id]
        variant_idx = np.random.randint(len(variants))
        self._selected_report = variants[variant_idx]

        # Randomize the primary "Threat" sensor (A, B, or C) where applicable
        sensor_roll = np.random.choice(list(self._config.keys()))
        
        if self._task_id == 1:
            self._primary_target = sensor_roll # The one degrading
        elif self._task_id == 2:
            self._primary_target = sensor_roll # The reversible threat
        elif self._task_id == 4:
            # The vial with the highest Ea (Fastest threat)
            self._primary_target = max(self._config.keys(), key=lambda k: self._config[k].activation_energy)
        else:
            self._primary_target = "A"

        self._save_session()

        return SciHypothesisObservation(
            task_id=self._task_id,
            incident_report=self._selected_report,
            task_description="[CRISIS OVERRIDE] Please refer to incident_report.",
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

            target = action.target_sensor or "A"
            if target not in self._config:
                error_msg = f"Invalid target_sensor '{target}'. Available: {list(self._config.keys())}"
                return SciHypothesisObservation(
                    task_id=self._task_id,
                    incident_report=self._selected_report,
                    task_description="[CRISIS OVERRIDE] Please refer to incident_report.",
                    current_step=self._step_count,
                    max_steps=MAX_STEPS[self._task_id],
                    experiments_remaining=remaining_steps,
                    hypothesis_feedback=error_msg,
                    reward=-0.01,
                    metadata={"episode_id": self._episode_id}
                )

            cost = self._calculate_experiment_cost(temp, conc, times)
            
            if self._budget_spent + cost > total_budget:
                self._save_session()
                return SciHypothesisObservation(
                    task_id=self._task_id,
                    incident_report=self._selected_report,
                    task_description="[CRISIS OVERRIDE]",
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
            target_config = self._config[target]
            data = run_experiment(target_config, temp, conc, times)

            # Compute the true k at this temperature (used internally for Arrhenius hints only)
            from .simulator import arrhenius_k, compute_hints
            actual_k_at_T = arrhenius_k(
                target_config.k,
                target_config.activation_energy,
                target_config.k_ref_temp,
                temp
            )

            # Extract history of (actual_k_at_T, T) from previous experiments FOR THIS SENSOR
            history = [
                {"k": entry["actual_k_at_T"], "T": entry["temperature"]}
                for entry in self._experiment_log
                if "actual_k_at_T" in entry and entry.get("target") == target
            ]

            hints = compute_hints(data, temp, history=history, actual_k_at_T=actual_k_at_T)
            
            # For multi-sensor tasks, inject the target name
            hints["measured_sensor"] = target

            self._experiment_log.append({
                "step": self._step_count,
                "target": target,
                "temperature": temp,
                "concentration": conc,
                "cost": cost,
                "actual_k_at_T": actual_k_at_T,
                "data": data,
                "hints": hints
            })

            # Per-step reward AFTER appending to log
            step_reward, step_breakdown = score_diagnostic_action(
                action_type="run_experiment",
                step_count=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiment_log=self._experiment_log,
                temperature=temp,
                concentration=conc,
                target_sensor=target,
                primary_target=self._primary_target,
            )
            self._accumulated_step_rewards += step_reward
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                incident_report=self._selected_report,
                task_description="[CRISIS OVERRIDE]",
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
            step_reward, step_breakdown = score_diagnostic_action(
                action_type="propose_hypothesis",
                step_count=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiment_log=self._experiment_log,
                predicted_order=action.predicted_order,
                predicted_k=action.predicted_k,
                primary_target=self._primary_target,
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
                task_description="[CRISIS OVERRIDE]",
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=remaining_steps,
                budget_remaining=total_budget - self._budget_spent,
                budget_spent=self._budget_spent,
                hypothesis_feedback=feedback,
                incident_report=self._selected_report,
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
            
            # Extract the actual primary threat config to grade the agent against
            primary_config = self._config[self._primary_target]

            reward, breakdown = calculate_triage_score(
                task_id=self._task_id,
                final_order=action.final_order or 0,
                final_k=action.final_k or 0.0,
                final_activation_energy=action.final_activation_energy,
                conclusion_text=action.conclusion or "",
                true_order=primary_config.order,
                true_k=primary_config.k,
                true_activation_energy=primary_config.activation_energy,
                primary_target=self._primary_target,
                available_sensors=list(self._config.keys()),
                steps_used=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                budget_spent=self._budget_spent,
                total_budget=total_budget,
                accumulated_step_rewards=self._accumulated_step_rewards
            )
            self._save_session()

            return SciHypothesisObservation(
                task_id=self._task_id,
                incident_report=self._selected_report,
                task_description="[CRISIS OVERRIDE]",
                current_step=self._step_count,
                max_steps=MAX_STEPS[self._task_id],
                experiments_remaining=0,
                budget_remaining=total_budget - self._budget_spent,
                budget_spent=self._budget_spent,
                final_feedback=(
                    f"Triage complete. Real Threat was Sensor {self._primary_target}. "
                    f"True order: {primary_config.order} ({'Reversible' if primary_config.order == 3 else 'Normal'}), "
                    f"True k: {round(primary_config.k, 6)}, "
                    f"True Ea: {round(primary_config.activation_energy, 1)} J/mol. "
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