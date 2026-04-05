import asyncio
import os
import requests
from typing import List
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

if API_KEY is None:
    raise ValueError("HF_TOKEN environment variable is required")

BASE_URL = "http://localhost:8000"
MAX_STEPS = 8


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def parse_action(text):
    text = text.lower().strip()
    if text.startswith("escalate"):
        return "escalate"
    elif text.startswith("request"):
        return "request_info"
    return "reply"


def build_prompt(obs):
    return f"""
Customer message: {obs['customer_message']}
Sentiment: {obs['sentiment']}
Urgency: {obs['urgency']}

Rules:
- sentiment < -0.5 → escalate
- billing issues → request_info
- simple queries → reply

Choose one:
reply
request_info
escalate
"""


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards = []
    steps_taken = 0

    log_start("support", "support_env", MODEL_NAME)

    res = requests.post(f"{BASE_URL}/reset", json={})
    result = res.json()

    obs = result["observation"]
    done = result["done"]

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        prompt = build_prompt(obs)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.2,
            )
            text = completion.choices[0].message.content.strip()
        except:
            text = "reply"

        action = parse_action(text)

        res = requests.post(
            f"{BASE_URL}/step",
            json={"action": {"action_type": action}},
        )

        result = res.json()

        reward = result.get("reward", 0.0)
        done = result.get("done", False)

        rewards.append(reward)
        steps_taken = step

        log_step(step, action, reward, done, None)

        obs = result["observation"]

    success = sum(rewards) > 0.5
    log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    asyncio.run(main())