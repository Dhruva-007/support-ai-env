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
| State | Ticket + sentiment + urgency + category + context |
| Action | reply / request_info / escalate |
| Reward | Deterministic scoring with SLA time bonus |
| Goal | Maximize cumulative reward |

---

## 📋 Observation Space

Each observation returned by `reset()` and `step()` contains:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Episode identifier |
| `customer_message` | string | The support ticket text |
| `sentiment` | float | Sentiment score of the ticket (−1.0 to +1.0) |
| `urgency` | string | `low` / `medium` / `high` |
| `difficulty` | string | `easy` / `medium` / `hard` (derived from urgency) |
| `category` | string | Ticket category: billing, delivery, complaint, technical, etc. |
| `context` | dict | Structured flags: `{"fraud": true}`, `{"vip_user": true}`, `{"system_issue": true}`, etc. |
| `history` | list | Actions taken so far in this episode |
| `time_elapsed` | int | Number of steps taken |
| `status` | string | `open` or `resolved` |
| `reward` | float | Reward from the last action |
| `done` | bool | Whether the episode has ended |

---

## 🗂️ Dataset

The ticket pool contains **57 real-world support tickets** balanced equally across all three difficulty tiers:

| Tier | Count | Urgency | Categories |
|------|-------|---------|------------|
| Easy | 19 | low | account, billing, complaint, delivery, general, order, payment, product, refund, technical |
| Medium | 19 | medium | account, billing, complaint, delivery, order, payment, product, refund, technical |
| Hard | 19 | high | billing, complaint, delivery, payment, refund, technical |

Hard tickets include **6 ambiguous cases** where tone is calm but urgency is high (e.g. fraud, system failures phrased politely) — requiring agents to reason beyond surface sentiment.

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
```json
{
  "observation": {
    "customer_message": "Where is my order?",
    "urgency": "medium",
    "sentiment": -0.2,
    "category": "delivery",
    "context": {"order_status": "delayed"},
    "difficulty": "medium"
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
```json
{
  "reward": 0.45,
  "done": false,
  "observation": {...}
}
```

---

## ⚖️ Good vs Bad Trajectories

### ❌ Bad Policy (medium task — wrong action)
```
step 1: escalate → -0.30
END (score: 0.10)
```

### ✅ Optimal Policy (Medium Task — with SLA bonus)
```
step 1: request_info → +0.45
step 2: reply        → +1.00
END (Total: +1.45, score: 0.95)
```

### ✅ Optimal Policy (High Urgency — with SLA bonus)
```
step 1: escalate → +0.95
END (score: 0.90)
```

### ✅ Optimal Policy (Easy Task — with SLA bonus)
```
step 1: reply → +1.15
END (score: 0.95)
```

---

## ⚙️ Action Space

- `reply` → respond directly to the customer
- `request_info` → ask the customer for more details before acting
- `escalate` → hand off to a human agent immediately

---

## 📊 Reward Logic

| Condition | Base Reward | With SLA Bonus (step 1) |
|----------|------------|------------------------|
| Easy correct (reply) | +1.00 | +1.15 |
| Medium step 1 correct (request_info) | +0.30 | +0.45 |
| Medium step 2 correct (reply) | +0.90 | +1.00 |
| Hard correct (escalate) | +0.80 | +0.95 |
| Wrong action | −0.30 | — |
| Repeated action | −0.10 penalty | — |

---

## 🏅 SLA-Based Reward Shaping

Rewards include a **time bonus** based on step efficiency. Fewer steps to the correct action = higher reward, mirroring real-world SLA expectations.

```
time_bonus = max(0, (4 - step_count) * 0.05)
reward += time_bonus  # only when reward > 0
```

| Step | Time Bonus |
|------|-----------|
| Step 1 correct | +0.15 |
| Step 2 correct | +0.10 |
| Step 3 correct | +0.05 |
| Step 4+ | no bonus |

---

## 🎯 Ambiguous Hard Tasks

Hard tasks include tickets where **sentiment contradicts urgency** — the agent must reason across all signals (urgency + category + context) rather than relying on emotional tone alone.

**Example:**
```json
{
  "ticket": "No rush, but there seems to be a duplicate transaction on my account.",
  "sentiment": 0.0,
  "urgency": "high",
  "category": "billing",
  "context": {"fraud": true},
  "expected_action": "escalate"
}
```
Calm tone, but `urgency=high` and `context.fraud=true` → correct action is `escalate`, not `reply`.

These ambiguous tickets score **0.80** (no sentiment bonus) vs **0.90** for clearly distressed tickets, creating genuine score variation within the hard tier.

---

## 📊 Grading Criteria

### Easy
| Trajectory | Score |
|-----------|-------|
| `reply` (correct) | 0.95 |
| Any wrong action | 0.10 |

### Medium
| Trajectory | Score |
|-----------|-------|
| `request_info → reply` (full correct path) | 0.95 |
| `reply` only (skipped investigation step) | 0.40 |
| `request_info → wrong` (info gathered, wrong follow-up) | 0.30 |
| Any other trajectory | 0.10 |

### Hard
| Trajectory | Score |
|-----------|-------|
| `escalate` + sentiment < −0.5 (clearly distressed) | 0.90 |
| `escalate` + sentiment ≥ −0.5 (ambiguous ticket) | 0.80 |
| Any wrong action | 0.30 |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|--------|--------|------------|
| `/reset` | POST / GET | Start a new episode (optional `task_type` body param) |
| `/step_logged` | POST | Take an action |
| `/state` | GET | View current environment state |
| `/score` | GET | Get final episode score |
| `/tasks` | GET | List all task types with live counts, criteria, and sample tickets |
| `/health` | GET | Health check |

### GET /tasks Response
Returns live data from the ticket pool — `sample` is drawn randomly each call:
```json
{
  "tasks": [
    {
      "type": "easy",
      "description": "Simple informational queries...",
      "ticket_count": 19,
      "categories": ["account", "billing", "delivery", ...],
      "scoring_criteria": {"correct (reply at step 1)": 0.95, "wrong action": 0.10},
      "reward_range": {"correct_step_1": 1.15, "wrong": -0.3},
      "sample": {"ticket": "...", "category": "...", "urgency": "low", ...}
    }
  ],
  "total_ticket_pool": 57,
  "action_space": ["reply", "request_info", "escalate"],
  "score_range": [0.01, 0.99]
}
```

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

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |

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

- Real-world customer support scenarios (57 tickets, 10 categories)
- Balanced difficulty tiers: 19 easy / 19 medium / 19 hard
- Multi-step RL environment with 2-step medium task flow
- Deterministic grading with clear, reproducible criteria
- SLA-based reward shaping (time bonus for faster resolution)
- Ambiguous hard tasks requiring multi-signal reasoning
- Rich observation space: sentiment, urgency, category, context flags
- Self-documenting `/tasks` endpoint with live ticket counts and sample
- OpenEnv compliant
- FastAPI backend with route conflict resolution
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
