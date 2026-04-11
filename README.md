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
  <p><b>Scientific Decision Making & Triage Under Pressure</b></p>
  <p><em>An OpenEnv benchmark evaluating whether AI agents can manage limited resources to solve high-stakes chemical crises.</em></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Certified_Benchmark-blue?style=for-the-badge&logo=huggingface" />
  <img src="https://img.shields.io/badge/Grading-Immaculate_Discovery-red?style=for-the-badge" />
</p>

| Property | Description |
| :--- | :--- |
| **Domain** | Chemical Kinetics & Process Safety |
| **Primary Task** | Stochastic Multi-Sensor Triage |
| **Persona** | Senior Chemical Kinetics Engineer |
| **Grading** | Deterministic (Zero-LLM Variance) |
| **Complexity** | Differential Equation ODE Simulator |

---

## 🚩 The Mission
In a real-world chemical plant or high-tech laboratory, things go wrong fast. A cooling fan fails, a vial starts to leak, or a reactor begins to overheat. 

The problem is **Noise**. In a crisis, you are flooded with data from dozens of sensors. Most are reporting normal conditions ("Red Herrings"), while only one is showing the actual threat. 

**Lab-Triage-Environment** tests if an AI agent can act as an expert scientist in these moments. The agent must find the "needle in the haystack," mathematically identify the threat, and recommend a fix—all while managing a strictly limited budget and time.

### 🧠 The Core Challenge: Three Pillars of Triage
This benchmark evaluates three specific skills that differentiate a "guessing" agent from a "thinking" agent:
1. **Strategic Probing**: Can the agent efficiently use a limited budget to find the one decaying sensor among the noise?
2. **Kinetic Accuracy**: Can the agent accurately calculate the physics (Reaction Order and Rate) of the threat?
3. **Resource Management**: Can the agent solve the crisis with the minimum number of experiments before the "reactor" fails?

---

## 🏗️ Environment Architecture

```mermaid
graph TD
    A[Incident Report] -->|Stochastic Narrative| B(AI Agent)
    B -->|Action: run_experiment| C[Lab Simulator]
    C -->|Observation: Time/Conc Data| B
    B -->|Action: propose_hypothesis| D[Deductive Feedback]
    B -->|Action: conclude| E[Immaculate Grader]
    E -->|Success Metric| F{Score: 0.0 - 1.0}
    F -->|Filtering Check| G[0.50 Noise-Filtering Cap]
```

---

## ⚖️ Immaculate Discovery: The Grading Philosophy
To eliminate "lucky guessing," we implemented a deterministic grading system based on actual kinetic ground truth.

> [!IMPORTANT]
> **The 0.50 Noise-Filtering Cap**: Following elite triage standards, an agent **cannot score higher than 0.50** if it identifies the primary threat but fails to explicitly mention or dismiss the "Red Herring" sensors in its conclusion. This ensures the agent is actually performing *triage*, not just guessing the outlier.

| Component | Weight | Logic |
|-----------|--------|-------|
| **Kinetics Accuracy** | 40% | Log-scale deviation of $k$ and $E_a$ from the ODE ground truth. |
| **Noise Filtering** | 30% | Explicit dismissal of inert sensors (A, B, C) in reasoning. |
| **Reasoning Bonus** | 15% | Detection of causal keywords (*Arrhenius, Plateau, Reversible*). |
| **Efficiency** | 15% | High fractional bonus for minimizing experiments and budget. |

---

## 📊 Crisis Tiers (Mission Variants)

The environment features a **Stochastic Scenario Engine**. Every `reset()` pulls from a pool of narrative variants, ensuring benchmark integrity.

| Mission | Scenario | The Triage Factor |
|---------|----------|-------------------|
| **Level 1** | **Reactor Gamma-7** | **Signal Isolation**. Identify simple decay $(Order 1/2)$ among inert coolant loops. |
| **Level 2** | **Scrubber Rupture** | **Equilibrium Detection**. Isolate lethal Reversible plateaus from linear leaks. |
| **Level 3** | **Launch Pad 39B** | **Arrhenius Study**. Find root cause: Thermal breakdown vs. Contamination. |
| **Level 4** | **Global Seed Vault** | **Resource Priority**. Identify the mix with the HIGHEST $E_a$ (Heat Sensitivity). |

---

## 🚀 Try It Now: Zero-Config API

You can query the live environment directly via `curl` to test connection and responses.

### 1. Reset the Environment (Task 1)
```bash
curl -X POST "https://quaxg-lab-triage-env.hf.space/reset?task_id=1" \
     -H "Accept: application/json"
```

### 2. Run a Diagnostic Experiment
```bash
curl -X POST "https://quaxg-lab-triage-env.hf.space/step" \
     -H "Content-Type: application/json" \
     -d '{
       "action_type": "run_experiment",
       "target_sensor": "A",
       "temperature": 343.15,
       "concentration": 1.0,
       "time_points": [0, 30, 60, 120]
     }'
```

---

## 🧩 Technical Specifications

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

### Observation Space
```json
{
  "task_id": 4,
  "incident_report": "🚨 INCIDENT REPORT: SVALBARD SEED VAULT FAILURE...",
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

## 🔮 Future Roadmap (Next Rounds)

- **Phase 1: Procedural Fuzzing**: Generative "noise profiles" to simulate sensor jitter and cross-talk.
- **Phase 2: Complex Tool-Use**: Integration of `mass_spec()`, `ph_probe()`, and `titrate_sample()` actions.
- **Phase 3: Multi-Reactor Fleet**: Managing 10+ concurrent reactors across global sites with network latency.

---

## 👨‍🔬 About / Citation
This benchmark is built for the **OpenEnv** ecosystem to push the boundaries of LLM scientific reasoning. 
*Contact: [Quaxg](https://huggingface.co/Quaxg)*
