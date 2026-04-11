# inference.py
import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import SciHypothesisEnv
from models import SciHypothesisAction as HypothesisAction

# --- Required env vars ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "https://Quaxg-sci-hypothesis-env.hf.space")
IMAGE_NAME = os.getenv("IMAGE_NAME", None)

BENCHMARK = "sci_hypothesis_env"
MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
You are a Senior Chemical Kinetics Engineer at a leading national laboratory.
You manage high-stakes crisis situations. Multiple instruments (Sensors A, B, C) might trigger alarms, but only one is the root cause threat.
You must respond with ONLY valid JSON.

Available actions:
1. Run a diagnostic (experiment) on a specific sensor:
{"action_type": "run_experiment", "target_sensor": "A", "temperature": <270-400>, "concentration": <0.001-2.0>, "time_points": [...]}

2. Propose a hypothesis:
{"action_type": "propose_hypothesis", "hypothesis_text": "<belief>", "predicted_order": <1, 2, or 3>, "predicted_k": <float>}

3. Conclude (ends the triage, resolves the incident):
{"action_type": "conclude", "conclusion": "<reasoning>", "final_order": <1, 2, or 3>, "final_k": <float>, "final_activation_energy": <float or null>}

Scenarios & Strategy:
- General: You will see multiple sensors. Use small, cheap experiments to identify which sensor is the true threat (e.g., degrading rapidly or plateauing), then focus on it.
- Task 3: Extremely noisy. Determine if the failure is Thermal Breakdown (Arrhenius) or Contamination (flat) by varying temperature.
- Task 4: Multi-Vial Epidemic. Find the vial with the HIGHEST Activation Energy (Ea) and fully characterize it.

Budget Management:
Each experiment costs money. Room temp: $50. High/Low temp: +$100. Long/dense points: +$50. 
If an action exceeds your budget, you receive a penalty.

Analysis Hints:
The environment provides "analysis_hints" after each experiment, giving R^2 fits and Arrhenius estimates specifically for the targeted sensor.
""").strip()


# --- Logging helpers (exact format required by hackathon) ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, budget: float, error: Optional[str]):
    error_val = error if error else "null"
    action_str = action.replace("\n", " ").replace("\r", " ")
    if len(action_str) > 200:
        action_str = action_str[:200] + "..."
    # Keep the budget log variable if platform needs it, otherwise it's just harmless
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"budget={budget:.1f} done={str(done).lower()} error={error_val}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


def call_llm(client: OpenAI, messages: list) -> dict:
    """Call LLM and parse JSON action."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


async def run_episode(task_id: int) -> float:
    """Run one full episode. Returns final score."""
    task_name = f"task_{task_id}"
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    if IMAGE_NAME:
        env = await SciHypothesisEnv.from_docker_image(IMAGE_NAME)
    else:
        env = SciHypothesisEnv(base_url=ENV_URL)

    try:
        # Reset
        result = await env.reset(task_id=task_id)
        obs = result.observation

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Incident Report: {obs.incident_report}\n"
                f"Context: {obs.known_context}\n"
                f"Max steps: {obs.max_steps}\n"
                f"Starting Budget: ${obs.budget_remaining}\n"
                "Respond with your first action as JSON."
            )}
        ]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get action from LLM
            error = None
            try:
                action_dict = await asyncio.to_thread(call_llm, llm, messages)
                # print(f"[DEBUG] action_dict={action_dict}", flush=True)
                action = HypothesisAction(**action_dict)
            except Exception as e:
                error = str(e)[:80]
                # Fallback: conclude with best guess
                action = HypothesisAction(
                    action_type="conclude",
                    conclusion="fallback due to parse error",
                    final_order=1,
                    final_k=0.01,
                    final_activation_energy=None
                )
                action_dict = action.model_dump(exclude_none=True)

            # Step environment
            result = await env.step(action)
            obs = result.observation
            reward = obs.reward if obs.reward is not None else (result.reward or 0.0)
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(action_dict),
                reward=reward,
                done=done,
                budget=obs.budget_remaining or 0,
                error=error
            )

            # Build next message from observation
            obs_text = ""
            if obs.experimental_data:
                obs_text = f"Experiment results: {json.dumps(obs.experimental_data)}\n"
                if obs.analysis_hints:
                    obs_text += f"Analysis Hints: {json.dumps(obs.analysis_hints)}\n"
            elif obs.hypothesis_feedback:
                obs_text = obs.hypothesis_feedback + "\n"
            elif obs.final_feedback:
                obs_text = obs.final_feedback + "\n"

            obs_text += f"Steps remaining: {obs.experiments_remaining}\n"
            if obs.budget_remaining is not None:
                obs_text += f"Budget remaining: ${obs.budget_remaining:.1f}"

            messages.append({"role": "assistant", "content": json.dumps(action_dict)})
            messages.append({"role": "user", "content": obs_text + "\nNext action as JSON:"})

            if done:
                break

        # Compute final score
        # Compute final score
        final_rewards = [r for r in rewards if r > 0]

        # Ensure we never return exactly 0.0 if no rewards were found
        score = max(final_rewards) if final_rewards else 0.001

        # CRITICAL: Change 0.0 to 0.001 and 1.0 to 0.999
        # This ensures the score is strictly within the (0, 1) range
        score = min(max(score, 0.001), 0.999)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main():
    all_scores = []
    for task_id in [1, 2, 3, 4]:
        score = 0.0
        for attempt in range(2):  # retry once on failure
            score = await run_episode(task_id)
            if score > 0:
                break
        all_scores.append(score)

if __name__ == "__main__":
    asyncio.run(main())
