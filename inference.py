import os
import sys
import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

if API_KEY is None:
    raise ValueError("HF_TOKEN environment variable is required")

BASE_URL = "http://localhost:8000"
MAX_STEPS = 6
SUCCESS_THRESHOLD = 0.7

TASK_TYPES = ["easy", "medium", "hard"]


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )


def llm_policy(client, obs, history):
    urgency = obs.get("urgency", "low")
    sentiment = obs.get("sentiment", 0.0)
    message = obs.get("customer_message", "")
    difficulty = obs.get("difficulty", "easy")
    category = obs.get("category", "general")
    step_history = obs.get("history", history)

    system_prompt = (
        "You are a customer support AI agent. Your job is to choose the single "
        "best action for a given support ticket.\n\n"
        "Available actions:\n"
        "  reply         — Send a direct response. Use for simple, informational questions.\n"
        "  request_info  — Ask the customer for more details before acting. Use when the "
        "issue needs investigation (billing errors, missing orders, damaged items, etc.).\n"
        "  escalate      — Hand off to a human agent immediately. Use for high urgency, "
        "very negative sentiment, complaints, fraud, or technical system failures.\n\n"
        "Decision rules (apply in order):\n"
        "  1. If urgency is HIGH → escalate\n"
        "  2. If sentiment is very negative (below -0.5) → escalate\n"
        "  3. If urgency is MEDIUM and you have NOT yet collected info → request_info\n"
        "  4. If urgency is MEDIUM and info is already collected → reply\n"
        "  5. If urgency is LOW → reply\n\n"
        "Respond with ONLY one word: reply, request_info, or escalate. No other text."
    )

    info_collected = len(step_history) > 0 and step_history[0] == "request_info"

    user_prompt = (
        f"Ticket: {message}\n"
        f"Category: {category}\n"
        f"Urgency: {urgency}\n"
        f"Sentiment score: {sentiment:.2f}  "
        f"({'very negative' if sentiment < -0.5 else 'negative' if sentiment < 0 else 'neutral/positive'})\n"
        f"Difficulty: {difficulty}\n"
        f"Steps taken so far: {step_history}\n"
        f"Info already collected: {'yes' if info_collected else 'no'}\n\n"
        "What is your action?"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10,
            temperature=0.7
        )
        raw = response.choices[0].message.content.strip().lower()

        for action in ["request_info", "escalate", "reply"]:
            if action in raw:
                return action

    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)

    return _rule_policy(obs, history)


def _rule_policy(obs, history):
    urgency = obs.get("urgency", "low")
    sentiment = obs.get("sentiment", 0.0)

    if urgency == "high":
        return "escalate"

    if sentiment < -0.5:
        return "escalate"

    if urgency == "medium":
        obs_history = obs.get("history", history)
        if len(obs_history) == 0 or obs_history[0] != "request_info":
            return "request_info"
        return "reply"

    return "reply"


def run_episode(client, task_type):
    rewards = []
    history = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_type, "support_env", MODEL_NAME)

    try:
        obs = {}
        done = True

        for attempt in range(5):
            try:
                res = requests.post(
                    f"{BASE_URL}/reset",
                    json={"task_type": task_type},
                    timeout=10
                )
            except Exception:
                log_end(False, 0, 0.0, [])
                return 0.0

            if res.status_code != 200:
                log_end(False, 0, 0.0, [])
                return 0.0

            reset_result = res.json()

            if reset_result.get("error"):
                continue

            obs = reset_result.get("observation", {})
            done = reset_result.get("done", False)

            if not done:
                break

        if done:
            log_end(False, 0, 0.0, [])
            return 0.0

        for step in range(1, MAX_STEPS + 1):

            if done:
                break

            action = llm_policy(client, obs, history)

            try:
                res = requests.post(
                    f"{BASE_URL}/step_logged",
                    json={"action": {"action_type": action}},
                    timeout=10
                )

                if res.status_code != 200:
                    log_step(step, action, 0.0, True, "api_error")
                    break

                result = res.json()

            except Exception:
                log_step(step, action, 0.0, True, "request_failed")
                break

            if result.get("error"):
                log_step(step, action, 0.0, True, result.get("error"))
                break

            if "observation" not in result:
                log_step(step, action, 0.0, True, "api_error")
                break

            reward = result.get("reward", 0.0)
            done = result.get("done", False)

            history.append(action)
            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, None)

            obs = result.get("observation", {})

        try:
            res = requests.get(f"{BASE_URL}/score", timeout=10)
            if res.status_code == 200:
                score = res.json().get("score", 0.0)
        except Exception:
            score = 0.0

        success = (len(rewards) > 0 and rewards[-1] > 0) or score >= SUCCESS_THRESHOLD

    finally:
        log_end(success, steps_taken, score, rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_type in TASK_TYPES:
        run_episode(client, task_type)


if __name__ == "__main__":
    main()
