# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

from support_env.models import SupportAction, SupportObservation
from support_env.environment import SupportEnvironment
from support_env.tasks import TASKS

from fastapi.responses import HTMLResponse
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import random


app = create_app(
    SupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_env",
    max_concurrent_envs=1,
)


class EnvHolder:
    def __init__(self):
        self.env = SupportEnvironment()
        self.env.reset()


STATE = EnvHolder()


class ResetRequest(BaseModel):
    task_type: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    all_tasks = list(TASKS.values())

    by_type = {"easy": [], "medium": [], "hard": []}
    for t in all_tasks:
        by_type[t["type"]].append(t)

    def pick_example(task_list):
        t = random.choice(task_list)
        return {
            "ticket": t["ticket"],
            "category": t["category"],
            "sentiment": t["sentiment"],
            "urgency": t["urgency"],
            "expected_action": t["expected_action"],
            "context": t.get("context", {})
        }

    return {
        "tasks": [
            {
                "type": "easy",
                "description": (
                    "Simple informational queries with no investigation required. "
                    "Covers FAQs on payments, shipping, account management, product info, "
                    "and policies. Agent must reply directly without requesting more details."
                ),
                "urgency": "low",
                "expected_action": "reply",
                "difficulty": "easy",
                "ticket_count": len(by_type["easy"]),
                "categories": sorted(set(t["category"] for t in by_type["easy"])),
                "scoring_criteria": {
                    "correct (reply at step 1)": 0.95,
                    "wrong action": 0.10
                },
                "reward_range": {
                    "correct_step_1": 1.15,
                    "wrong": -0.3
                },
                "grader": "grade_easy",
                "sample": pick_example(by_type["easy"])
            },
            {
                "type": "medium",
                "description": (
                    "Issues that require investigation before the agent can respond. "
                    "Covers billing discrepancies, wrong/missing deliveries, account access "
                    "problems, and product complaints. Agent must follow a 2-step flow: "
                    "first request_info, then reply. Skipping the first step is penalised."
                ),
                "urgency": "medium",
                "expected_action": "request_info → reply",
                "difficulty": "medium",
                "ticket_count": len(by_type["medium"]),
                "categories": sorted(set(t["category"] for t in by_type["medium"])),
                "scoring_criteria": {
                    "full correct path (request_info → reply)": 0.95,
                    "skipped to reply (wrong flow)": 0.40,
                    "gathered info then wrong step 2": 0.30,
                    "completely wrong": 0.10
                },
                "reward_range": {
                    "step_1_correct (request_info)": 0.45,
                    "step_2_correct (reply)": 1.00,
                    "wrong": -0.3
                },
                "grader": "grade_medium",
                "sample": pick_example(by_type["medium"])
            },
            {
                "type": "hard",
                "description": (
                    "High-urgency tickets requiring immediate escalation to a human agent. "
                    "Includes both clearly distressed customers (very negative sentiment) and "
                    "ambiguous tickets where tone is calm but urgency is high — e.g. fraud, "
                    "system failures, or critical account issues phrased politely. "
                    "Agent must escalate based on urgency and context, not just emotional tone."
                ),
                "urgency": "high",
                "expected_action": "escalate",
                "difficulty": "hard",
                "ticket_count": len(by_type["hard"]),
                "categories": sorted(set(t["category"] for t in by_type["hard"])),
                "scoring_criteria": {
                    "correct (escalate) + sentiment < -0.5": 0.90,
                    "correct (escalate) + sentiment >= -0.5": 0.80,
                    "wrong action": 0.30
                },
                "reward_range": {
                    "correct_step_1": 0.95,
                    "wrong": -0.3
                },
                "grader": "grade_hard",
                "sample": pick_example(by_type["hard"])
            }
        ],
        "total_ticket_pool": len(all_tasks),
        "action_space": ["reply", "request_info", "escalate"],
        "score_range": [0.01, 0.99]
    }


@app.post("/reset")
@app.get("/reset")
async def reset_post(body: ResetRequest = None):
    task_type = None
    if body is not None:
        task_type = body.task_type

    STATE.env = SupportEnvironment()
    result = STATE.env.reset(task_type=task_type)
    return {
        "observation": result.dict(),
        "reward": 0.0,
        "done": False
    }


@app.post("/step_logged")
async def step_logged(action: dict):
    if STATE.env is None:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "error": "Environment not initialized. Please RESET."
        }

    if STATE.env.done:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "error": "Episode finished. Please click RESET."
        }

    act = SupportAction(**action["action"])
    result = STATE.env.step(act)

    return {
        "observation": result.dict(),
        "reward": round(result.reward, 2),
        "done": result.done
    }


@app.get("/score")
def get_score():
    try:
        score = STATE.env.compute_score()
        return {
            "score": round(score, 3),
            "steps": STATE.env.step_count,
            "trajectory": STATE.env.history,
            "task_type": STATE.env.task.get("type"),
            "expected_action": STATE.env.task.get("expected_action")
        }
    except Exception as e:
        return {"score": 0.0, "error": str(e)}


@app.get("/state")
def get_state():
    try:
        return STATE.env.state
    except Exception as e:
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).parent / "frontend.html"
    return html_path.read_text(encoding="utf-8")


def _deduplicate_routes(application):
    seen = set()
    unique = []
    for route in reversed(application.router.routes):
        path = getattr(route, "path", None)
        methods = frozenset(getattr(route, "methods", None) or [])
        key = (path, methods)
        if key not in seen:
            seen.add(key)
            unique.append(route)
    application.router.routes = list(reversed(unique))


_deduplicate_routes(app)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
