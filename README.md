---
title: Lab Triage Environment
emoji: 🧪
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

<div align="center">
  <h1>🧪 Lab-Triage-Environment</h1>
  <p><em>An OpenEnv benchmark testing the ability of AI agents to act as <b>Senior Chemical Kinetics Engineers</b> by diagnosing and mitigating high-stakes reactive crises.</em></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Phase_2_Verified-success?style=for-the-badge&logo=huggingface" />
</p>

## 💡 The Problem Statement
Industrial chemical accidents, toxic leaks, and pharmaceutical spoilage cause millions of dollars in damages and severe environmental hazards. When a runaway reaction begins, the window to stop it is drastically short.

Resolving these crises requires a **Senior Chemical Research Scientist** who can:
1. **Noise Filtering** — Pinpointing the true degrading chemical amidst stable, inert "red herring" sensors.
2. **Deductive Prioritization** — Identifying which reaction out of many has the highest mathematical risk (e.g., Activation Energy) of cascading.
3. **Strict Resource Management** — Executing targeted, budget-constrained experiments to parameterize the threat before catastrophe strikes.

**Can an LLM Agent successfully simulate a Staff Chemical Engineer in a crisis?**

No existing automated benchmark evaluates LLM agents on *strategic experimental triage*. This environment fills that gap. The LLM must read messy incident reports, conduct dynamic experiments on targeted physical sensors, identify the underlying kinetics (Order, $k$, $E_a$), and deploy a mathematical conclusion—all graded deterministically without "LLM-as-a-judge" variance.

---

## 🎯 Task Scenarios

The environment evaluates agents across 4 distinct crisis tiers, with scenarios and underlying values regenerating dynamically per episode.

| Tier | Scenario Variant | The "Triage" Factor |
|------|---------|---------------------|
| **Easy** | **Reactor Gamma-7 / Isotope Breach** | **Noise Filtering**. Search for the true unidirectional decay among inert coolant red-herrings (Sensors A, B, C). |
| **Medium** | **Scrubber Rupture / Polymer Surge** | **Deductive Reasoning**. Distinguishing lethal Reversible (Order 3) plateaus from standard linear driver leakage. |
| **Hard** | **Propellant Bubbling / Battery Runaway** | **Root Cause isolation**. High-noise Arrhenius variance studies to find Activation Energy (Ea) under tight launch windows. |
| **Expert** | **Seed Vault / Deep-Space Probe** | **Resource Allocation**. Multi-vial optimization to find the mix with the highest heat sensitivity (Highest Ea) across all sensors. |

> [!NOTE]
> **Stochastic Engine Enabled**: Every `reset()` call picks a random scenario variant and randomizes the "True Threat" sensor position, making this environment immune to benchmark memorization.

---

## 🚀 Quick Start (Try It Now)

The environment natively exposes standard OpenEnv HTTP endpoints, deployed on HuggingFace Spaces. 

### Local Python Client
```python
import asyncio 
from openenv.client import EnvClient 

async def main(): 
    # Connect to the remote environment
    env = await EnvClient.create("Quaxg/sci-hypothesis-env") 

    # Start a crisis scenario (Task 4: Multi-Vial Epidemic)
    result = await env.reset(task_id=4) 
    print("====== INCIDENT REPORT ======") 
    print(result.observation["incident_report"]) 

    # Step — Run targeted diagnostic experiments
    action = {
        "action_type": "run_experiment",
        "target_sensor": "A", # Targeting Vial A
        "temperature": 343.15,
        "concentration": 1.0,
        "time_points": [0, 30, 60, 120]
    }
    
    result = await env.step(action) 
    print(f"\nExperimental Data: {result.observation['experimental_data']}")
    
asyncio.run(main())
```

---

## 🧩 Observation and Action Schema

Agents are presented with real-time `Incident Reports` and must issue exact commands to the laboratory simulator.

### Agent Action Schema
```json
{
  "action_type": "run_experiment",
  "target_sensor": "A",
  "temperature": 310.15,
  "concentration": 1.0,
  "time_points": [0, 10, 30, 100]
}
```

### Environment Observation
```json
{
  "task_id": 4,
  "incident_report": "🚨 INCIDENT REPORT: MULTI-VIAL EPIDEMIC TRIAGE.\nA refrigeration unit failed. Three vital experimental vaccines (A, B, C) are decaying...",
  "budget_remaining": 800.0,
  "experimental_data": [
    {"time": 0.0, "concentration": 1.0, "temperature": 310.15},
    {"time": 10.0, "concentration": 0.94, "temperature": 310.15}
  ],
  "analysis_hints": {
    "measured_sensor": "A",
    "suggested_order": 1,
    "estimated_k": 0.0051
  }
}
```

---

## ⚖️ Immaculate Discovery: Grading Rigor

This environment uses a unique **"Noise-Filtering"** grading architecture to ensure agents cannot game the benchmark via lucky guessing. In accordance with elite triage standards, agents are graded on:

- **Mathematical Precision**: Accuracy of $k$ and $E_a$ estimates compared to the ground-truth ODE.
- **Filtering Multiplier (CRITICAL)**: Following winning submissions, an agent **cannot score higher than 0.50** if it identifies the threat but fails to explicitly mention the status of the "Red Herring" sensors in its conclusion.
- **Causal Reasoning Bonus**: Deterministic regex-based scoring for scientific keywords (*Arrhenius*, *Inert*, *Reversible*, *Plateau*) within the final reasoning text.
- **Strategic Information Gain**: Per-step rewards are granted for "Discovery" (first probe of a threat) and "Symmetry" (comparing sensors at identical conditions).

---

## 🧪 Baseline Inference

Evaluation executed via `inference.py` using `llama-3.1-8b-instant` operating strictly at `TEMPERATURE=0.3`.

```bash
# Run the automated inference loop on all tasks
uv run python inference.py
```

*Note: The inference baseline runs natively in isolated `[START]...[END]` loop blocks as required by the OpenEnv Phase 2 strict stdout protocols.*

---

## 🔮 Future Roadmap (Next Rounds)

To scale this benchmark for autonomous engineering fleets, we have outlined the following development phases:

### Phase 1: Procedural Kinetic Fuzzing
Implementing a generative model for chemical "noise profiles," allowing the environment to simulate sensor jitter, calibration drift, and signal cross-talk realistically.

### Phase 2: Interactive Laboratory Tool-Use
Transitioning from a single `run_experiment` action to a suite of specific laboratory tools:
- `mass_spec()` for molecular weight checks.
- `ph_probe()` for acidity monitoring.
- `titrate_sample()` for precise endpoint detection.

### Phase 3: Fuzzy Semantic Determinism
Integrating LLM-based verification for the `conclusion` field that uses semantic embedding similarity (rather than exact string matching) to reward deep scientific reasoning that matches the ground truth.

### Phase 4: Massive Triage Scaling
Simulating "Cloud Lab" outages where an agent must manage hundreds of concurrent reactors across several global sites, introducing network latency and asynchronous callback handling to the triage logic.

---

## 👨‍🔬 About
A zero-LLM deterministic benchmark testing the ability of AI agents to act as Senior Chemical Engineers by prioritizing and neutralizing mathematical kinetic catastrophes.
