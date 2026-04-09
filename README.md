---
title: Sci Hypothesis Env Environment Server
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# 🔬 Sci Hypothesis Env — Chemical Kinetics Laboratory

An **AI for Science** benchmark environment where agents must act as real scientists: design experiments within a budget, analyze concentration data, and deduce the hidden kinetic parameters of a chemical reaction.

This environment tests **scientific reasoning**, **strategic planning under budget constraints**, and **hypothesis-driven experimentation** — skills far beyond pattern recognition.

---

## 🎯 What Is This Environment About?

The agent is placed inside a virtual chemistry laboratory. A hidden chemical reaction is taking place, governed by unknown kinetic parameters. The agent must:

1. **Design and run experiments** (choosing temperature, concentration, and time points)
2. **Interpret the results** using automated analysis hints (R² fits, Arrhenius analysis)
3. **Propose hypotheses** and refine them
4. **Conclude** with the correct reaction order, rate constant (k), and activation energy (Ea)

Each experiment **costs money**, so the agent must be strategically efficient.

---

## 🧪 Tasks — Real-World Case Studies

All three tasks use the same action/observation interface but require different scientific strategies.

| Task | Scenario | Domain | Hidden Parameters | Budget |
|------|----------|--------|------------------|--------|
| **Task 1** | Antibiotic Clearance | Pharmacokinetics | Order (1 or 2), k | $500 |
| **Task 2** | CO₂ Carbon Capture | Industrial Chemistry | Order (always 3/Reversible), kf, kr | $800 |
| **Task 3** | Rocket Propellant Stability | Aerospace Engineering | Order (1, 2, or 3), k, **Ea** | $1200 |
| **Task 4** | Vitamin C Degradation | Food Science | Order (1 or 2), k, **Ea** | $1000 |

### Task 1 — Pharmacokinetics 💊
> *A Phase I clinical trial is underway for a new aqueous antibiotic. Determine if the drug is cleared via 1st or 2nd order kinetics to ensure safe patient dosing.*

- **Reference temperature**: 310.15 K (body temperature)
- **Noise level**: Low (`0.003`) — data is clean
- **Goal**: Identify reaction order and estimate rate constant k
- **Tip**: Start at 310.15 K, observe concentration decay, check Analysis Hints

### Task 2 — Carbon Capture ♻️
> *Optimizing a reversible CO₂ absorption reaction for industrial scrubbers. Unlike normal reactions, this one reaches an equilibrium plateau.*

- **Reaction type**: Always **Reversible (Order 3)**
- **Noise level**: Medium (`0.015`)
- **Goal**: Identify that concentration levels off (doesn't go to zero) and estimate k
- **Tip**: Use long time points to observe the plateau; check `fit_reversible` hint

### Task 3 — Propellant Stability 🚀
> *Determining the shelf-life stability of a high-energy rocket propellant. You must find the reaction order, k, AND the activation energy Ea.*

- **Reference temperature**: 350.15 K (operating temperature)
- **Noise level**: High (`0.045`) — very noisy data
- **Goal**: Characterize order, k, and Ea using the Arrhenius equation
- **Tip**: Vary temperature across ≥3 different values; check `suggested_ea` in hints

### Task 4 — Vitamin C Degradation 🍊
> *A fruit juice manufacturer wants to optimize pasteurization to minimize Vitamin C loss. Determine the degradation kinetics to ensure high product quality.*

- **Reference temperature**: 343.15 K (70°C)
- **Noise level**: Moderate (`0.025`)
- **Goal**: Identify reaction order (1 or 2), k, and activation energy Ea
- **Tip**: Perform experiments between 330 K and 380 K; prioritize temperature diversity for Ea accuracy

---

## 💰 Experiment Costs

The agent must run experiments within a fixed budget per task.

| Condition | Cost |
|-----------|------|
| Room temperature (280–350 K) | $50 |
| High temperature (> 350 K) | $150 |
| Low temperature (< 280 K) | $120 |
| High concentration (> 1.0 mol/L) | +$30 |
| Long/many time points (> 5 pts or > 500s) | +$50 |

> If the agent attempts an experiment it cannot afford, it receives a penalty and the experiment is rejected.

---

## 📊 Grading & Reward Structure

Each episode is scored on five components, weighted as follows:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Order Accuracy** | 30% | Correct reaction order (1, 2, or 3) |
| **k Accuracy** | 25% | Rate constant accuracy (log-scale error) |
| **Ea Accuracy** | 15% | Activation energy accuracy (Task 3 only) |
| **Budget Efficiency** | 15% | More reward for spending less |
| **Step Efficiency** | 5% | More reward for concluding earlier |
| **Step Bonuses** | 10% | Accumulated reward from intermediate steps |

### Scoring Details

**k Accuracy** uses a log-scale metric that is tolerant of order-of-magnitude estimates:
```
k_score = max(0, 1 - |log10(k_pred) - log10(k_true)|)
```

**Ea Accuracy** measures relative error:
```
ea_score = max(0, 1 - |Ea_pred - Ea_true| / Ea_true)
```

**Success threshold**: `score >= 0.5`

---

## 🔬 Reaction Types

| Order | Type | ODE | Signature |
|-------|------|-----|-----------|
| 1 | 1st Order | `dC/dt = -k·C` | `ln(C) vs t` is linear |
| 2 | 2nd Order | `dC/dt = -k·C²` | `1/C vs t` is linear |
| 3 | Reversible | `dC/dt = -kf·C + kr·(C₀-C)` | C plateaus at non-zero equilibrium |

---

## 🤖 Analysis Hints

After every experiment, the environment automatically provides the following hints in the observation:

```json
{
  "linearity_1st_order": 0.995,
  "linearity_2nd_order": 0.712,
  "fit_reversible": 0.621,
  "suggested_order": 1,
  "estimated_k": 0.04312,
  "temperature": 310.15,
  "suggested_ea": 52400.0,
  "ea_confidence": 0.998
}
```

| Field | Description |
|-------|-------------|
| `linearity_1st_order` | R² of 1st order ODE fit |
| `linearity_2nd_order` | R² of 2nd order ODE fit |
| `fit_reversible` | R² of reversible ODE fit |
| `suggested_order` | Best-fitting reaction order |
| `estimated_k` | Rate constant from best ODE fit |
| `suggested_ea` | Estimated Ea from Arrhenius regression (appears after ≥2 temperatures) |
| `ea_confidence` | R² of Arrhenius linear regression |

---

## ⚡ Quick Start

### Running Locally

```bash
# 1. Start the server
export PYTHONPATH=$PYTHONPATH:.
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# 2. In another terminal, run the agent
export ENV_URL="http://localhost:8000"
export GROQ_API_KEY="your_api_key"
python inference.py
```

### Connecting via Python Client

```python
from client import SciHypothesisEnv
from models import SciHypothesisAction

async with SciHypothesisEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task_id=1)
    obs = result.observation
    print(f"Task: {obs.task_description}")
    print(f"Budget: ${obs.budget_remaining}")

    # Run an experiment
    action = SciHypothesisAction(
        action_type="run_experiment",
        temperature=310.15,
        concentration=0.5,
        time_points=[0, 30, 60, 120, 300],
        metadata={"episode_id": obs.metadata["episode_id"]}
    )
    result = await env.step(action)
    print(f"Hints: {result.observation.analysis_hints}")
```

### Available Actions

```json
// Run an experiment
{"action_type": "run_experiment", "temperature": 310.15, "concentration": 0.5, "time_points": [0, 30, 60, 120]}

// Propose a hypothesis
{"action_type": "propose_hypothesis", "hypothesis_text": "...", "predicted_order": 1, "predicted_k": 0.04}

// Conclude the episode (triggers final scoring)
{"action_type": "conclude", "final_order": 1, "final_k": 0.04, "final_activation_energy": null}
```

---

## 🚀 Deploying to Hugging Face Spaces

```bash
# From the environment directory
openenv push

# Or with options
openenv push --repo-id my-org/sci-hypothesis-env --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` — Interactive UI for exploring the environment
- **API Docs** at `/docs` — Full OpenAPI/Swagger interface
- **WebSocket** at `/ws` — Persistent session endpoint

---

## 📁 Project Structure

```
sci_hypothesis_env/
├── .gitignore                  # Git exclusions (includes verify_*.py, .venv/)
├── .dockerignore               # Docker build exclusions
├── __init__.py                 # Module exports
├── README.md                   # This file
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependencies
├── client.py                   # SciHypothesisEnv WebSocket client
├── models.py                   # Action and Observation Pydantic models
├── inference.py                # LLM agent inference loop
└── server/
    ├── __init__.py             # Server module exports
    ├── app.py                  # FastAPI application (HTTP + WebSocket)
    ├── sci_hypothesis_env_environment.py  # Core environment logic (budget, hints, sessions)
    ├── simulator.py            # ODE simulator, curve fitting, Arrhenius hints
    ├── reward.py               # Scoring and reward computation
    └── Dockerfile              # Container image definition
```

---

## 🔧 Development

### Building the Docker Image

```bash
docker build -t sci_hypothesis_env-env:latest -f Dockerfile .
```

### Running with Docker

```bash
docker run -p 8000:8000 sci_hypothesis_env-env:latest
```
 
