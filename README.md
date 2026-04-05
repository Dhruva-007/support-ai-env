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

A real-world Reinforcement Learning (RL) environment for evaluating AI agents in customer support decision-making scenarios.

---

## 🎯 Problem Statement

Customer support systems must decide:
- When to reply
- When to request more information
- When to escalate to human agents

Incorrect decisions lead to poor user experience and delays.

---

## 💡 Solution

This project simulates realistic customer support tickets and evaluates agent decisions using rewards and graders.

---

## 🧠 RL Formulation

| Component | Description |
|----------|------------|
| State | Customer ticket + metadata |
| Action | reply, request_info, escalate |
| Reward | Context-aware scoring |
| Goal | Maximize reward |

---

## 📦 Dataset

Includes realistic tickets:
- Delivery issues
- Billing problems
- Technical errors

Example:
{
  "ticket": "Payment failed but money deducted",
  "sentiment": -0.6,
  "urgency": "medium",
  "expected_action": "request_info"
}

---

## ⚙️ Action Space

- reply
- request_info
- escalate

---

## 📊 Reward Logic

- Correct action → positive reward
- Wrong action → penalty
- Extra penalties:
  - repeated actions
  - unnecessary steps

---

## 🔁 API Endpoints

### Reset
POST /reset

### Step
POST /step
{
  "action": {
    "action_type": "reply"
  }
}

### State
GET /state

### Metadata
GET /metadata

---

## 🚀 Setup

### Build
docker build -t support_env .

### Run
docker run -p 8000:8000 support_env

### Access
http://localhost:8000/docs

---

## 🧪 Inference

python inference.py

---

## 🔐 Environment Variables

- API_BASE_URL (default provided)
- MODEL_NAME (default provided)
- HF_TOKEN (required)

---

## 📈 Output Format

[START] task=support env=support_env model=...
[STEP] step=1 action=reply reward=-0.45 done=false error=null
[END] success=true steps=2 rewards=-0.45,1.75

---

## 🏆 Features

- Real-world tasks
- Context-aware rewards
- Deterministic graders
- OpenEnv compliant
- Docker ready

---

## 👨‍💻 Author

G Dhruvann

---

## ✅ Validation

openenv validate http://localhost:8000

---

## 🏁 Conclusion

A robust RL environment for evaluating customer support agents with realistic decision-making.
