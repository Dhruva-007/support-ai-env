---
title: Support AI RL Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# 🤖 Support AI RL Environment

A production-grade Reinforcement Learning (RL) environment for simulating and evaluating customer support decision-making agents.

---

## 🎯 Problem Statement

Customer support systems must decide:
- When to reply
- When to request more information
- When to escalate

Incorrect decisions lead to poor user experience and operational inefficiencies.

---

## 💡 Solution

This environment simulates realistic support tickets and evaluates agent decisions using deterministic rewards and structured grading.

---

## 🧭 Architecture

User → Agent (Policy) → Environment (FastAPI) → Reward → Grader → Score → Feedback Loop

---

## 📁 Project Structure

```
support-ai-env/
│
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── frontend.html
│   └── requirements.txt
│
├── support_env/
│   ├── __init__.py
│   ├── environment.py
│   ├── graders.py
│   ├── models.py
│   ├── tasks.py
│   └── data/
│       └── tickets.json
│
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── README.md
```

---

## 🧠 RL Formulation

| Component | Description |
|----------|------------|
| State | Ticket + sentiment + urgency |
| Action | reply / request_info / escalate |
| Reward | Deterministic scoring |
| Goal | Maximize cumulative reward |

---

## 🧠 How Agent Learns

The agent follows the RL loop:

1. Observe state (ticket)
2. Choose action
3. Receive reward
4. Update policy

### Multi-step reasoning (Medium tasks)

- Step 1 → request_info
- Step 2 → reply

This ensures the agent learns **sequential decision-making**, not single-step guessing.

---

## 🔁 Training Loop (Conceptual)

```
for episode in episodes:
    state = env.reset()

    done = False
    while not done:
        action = policy(state)
        next_state, reward, done = env.step(action)

        policy.update(state, action, reward, next_state)

        state = next_state
```

---

## 📊 Sample API Interaction

### Reset
```
POST /reset
```

### Response
```
{
  "observation": {
    "customer_message": "Where is my order?",
    "urgency": "medium",
    "sentiment": -0.2
  },
  "done": false
}
```

---

### Step
```
POST /step_logged
{
  "action": {
    "action_type": "request_info"
  }
}
```

### Response
```
{
  "reward": 0.3,
  "done": false,
  "observation": {...}
}
```

---

## ⚖️ Good vs Bad Trajectories

### ❌ Bad Policy
```
step 1: reply → -0.30
END
```

### ✅ Optimal Policy (Medium Task)
```
step 1: request_info → +0.45
step 2: reply        → +1.00
END (Total: +1.45)
```

### ✅ Optimal Policy (High Urgency)
```
step 1: escalate → +0.95
END
```

---

## ⚙️ Action Space

- reply → respond directly
- request_info → gather details
- escalate → handover to human

---

## 📊 Reward Logic

| Condition | Base Reward | With SLA Bonus (step 1) |
|----------|------------|------------------------|
| Easy correct (reply) | +1.00 | +1.15 |
| Medium step 1 correct (request_info) | +0.30 | +0.45 |
| Medium step 2 correct (reply) | +0.90 | +1.00 |
| Hard correct (escalate) | +0.80 | +0.95 |
| Hard correct + fraud/vip context | +0.85 | +1.00 |
| Wrong action | −0.30 | — |
| Repeated action | −0.10 penalty | — |

---

## 🏅 SLA-Based Reward Shaping

Rewards now include a **time bonus** based on step efficiency. Fewer steps to the correct action = higher reward. This introduces continuous signal variation and teaches agents to resolve tickets quickly — mirroring real-world SLA expectations.

```
time_bonus = max(0, (4 - step_count) * 0.05)
reward += time_bonus
```

- Step 1 correct → +0.15 bonus
- Step 2 correct → +0.10 bonus
- Step 3 correct → +0.05 bonus
- Step 4+ → no bonus

---

## 🎯 Ambiguous Hard Tasks

Hard tasks now include tickets where **sentiment contradicts urgency** — the agent must reason across all signals (tone + urgency + category + context flags) to pick the correct action, rather than relying on surface-level text alone.

**Example:**
```json
{
  "ticket": "I think there might be a small issue with my last payment.",
  "sentiment": -0.3,
  "urgency": "high",
  "category": "billing",
  "context": {"fraud": true},
  "expected_action": "escalate"
}
```
Calm tone, but `urgency=high` and `context.fraud=true` → correct action is `escalate`, not `reply`.

---

## 🔌 API Endpoints

| Endpoint | Description |
|--------|------------|
| POST /reset | Start episode |
| POST /step_logged | Take action |
| GET /state | View state |
| GET /score | Final score |
| GET /tasks | List all task types with descriptions |
| GET /health | Health check |

### GET /tasks Response
```json
{
  "tasks": [
    {"type": "easy", "description": "Simple informational queries", "expected_action": "reply"},
    {"type": "medium", "description": "Issues requiring investigation", "expected_action": "request_info → reply"},
    {"type": "hard", "description": "High urgency escalations", "expected_action": "escalate"}
  ]
}
```

---

## 📈 Baseline Performance

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API.
Each task type runs one episode with LLM policy + deterministic fallback.
Scores are grader outputs in `[0.01, 0.99]`.

| Task | Difficulty | Trajectory | Reward | Score | Notes |
|------|-----------|------------|--------|-------|-------|
| easy | Easy | `reply` | 1.15 | 0.950 | Correct direct reply; deterministic |
| medium | Medium | `request_info → reply` | 0.45, 1.00 | 0.950 | Correct 2-step flow; deterministic |
| hard | Hard | `escalate` | 0.95 | 0.900 | Clearly distressed ticket (sentiment < −0.5) |
| hard | Hard | `escalate` | 1.00 | 0.900 | Fraud/VIP ticket; context bonus applied |
| hard | Hard | `escalate` | 0.95 | 0.800 | Ambiguous ticket (calm tone, high urgency) |

**Score distribution across all 57 tickets (correct agent):**

| Tier | Min Score | Max Score | Mean Score | Ticket Count |
|------|-----------|-----------|------------|--------------|
| Easy | 0.950 | 0.950 | 0.950 | 19 |
| Medium | 0.950 | 0.950 | 0.950 | 19 |
| Hard | 0.800 | 0.900 | 0.868 | 19 |

Hard score varies because `grade_hard` awards a sentiment bonus (`+0.1`) only when `sentiment < −0.5`. 13 of 19 hard tickets are clearly distressed (score 0.90); 6 are ambiguous/calm-tone tickets (score 0.80), specifically designed to test context-based reasoning.

---

## 🚀 Setup & Run

### Build
```
docker build -t support_env .
```

### Run
```
docker run -p 8000:8000 support_env
```

### Local Run
```
python inference.py
```

### Reproducible Benchmark Run
```
RANDOM_SEED=42 python inference.py
```

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `RANDOM_SEED` | No | — | Integer seed for reproducible ticket sampling |

---

## 📈 Output Format

```
[START] task=easy env=support_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=reply reward=1.15 done=true error=null
[END] success=true steps=1 score=0.950 rewards=1.15

[START] task=medium env=support_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=request_info reward=0.45 done=false error=null
[STEP] step=2 action=reply reward=1.00 done=true error=null
[END] success=true steps=2 score=0.950 rewards=0.45,1.00

[START] task=hard env=support_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=escalate reward=0.95 done=true error=null
[END] success=true steps=1 score=0.900 rewards=0.95
```

---

## 🏆 Features

- Real-world support scenarios (57 tickets, 10 categories)
- Balanced difficulty tiers: 19 easy / 19 medium / 19 hard
- Multi-step RL environment with 2-step medium task flow
- Deterministic grading with clear, reproducible criteria
- SLA-based reward shaping (time bonus for faster resolution)
- Context-aware reward bonus for fraud and VIP tickets
- Ambiguous hard tasks requiring multi-signal reasoning
- Context flags in observation (`fraud`, `vip_user`, `system_issue`)
- Optional deterministic seed for reproducible benchmark runs
- Self-documenting `/tasks` endpoint
- OpenEnv compliant
- FastAPI backend
- Docker ready

---

## ✅ Validation

```
openenv validate http://localhost:8000
```

---

## 👨‍💻 Author

G Dhruvann

---

## 🏁 Conclusion

This project delivers a practical and realistic Reinforcement Learning environment for customer support decision-making. By combining multi-step interactions, deterministic rewards, and real-world scenarios, it enables agents to learn meaningful and structured policies rather than one-step heuristics.

The environment is stable, interpretable, and OpenEnv-compatible, making it well-suited for both research and real-world agent evaluation.
