# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
FastAPI application for the Support Env Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

from support_env.models import SupportAction, SupportObservation
from server.support_env_environment import SupportEnvironment

import json
from datetime import datetime

from fastapi.responses import HTMLResponse
from pathlib import Path


app = create_app(
    SupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_env",
    max_concurrent_envs=1,
)


GLOBAL_ENV = SupportEnvironment()

EPISODE_ID = None


reset_history = []
step_history = []

LOG_FILE = "logs.json"


def save_log(entry):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reset")
async def reset_get():
    global EPISODE_ID

    result = GLOBAL_ENV.reset()
    EPISODE_ID = datetime.now().isoformat()

    entry = {
        "type": "reset",
        "time": EPISODE_ID,
        "observation": result.dict(),
        "reward": 0.0,
        "done": False
    }

    reset_history.append(entry)
    save_log(entry)

    return entry


@app.post("/step_logged")
async def step_logged(action: dict):

    if GLOBAL_ENV.done:
        return {
            "type": "error",
            "message": "Episode finished. Please click RESET.",
            "done": True
        }

    act = SupportAction(**action["action"])
    result = GLOBAL_ENV.step(act)

    reward = round(result.reward, 2)

    result.reward = reward

    entry = {
        "type": "step",
        "time": datetime.now().isoformat(),
        "action": action,
        "observation": result.dict(),
        "reward": reward,
        "done": result.done
    }

    step_history.append(entry)
    save_log(entry)

    return entry



@app.get("/state")
def get_state():

    return {
        "type": "state",
        "episode_id": EPISODE_ID,
        "step_count": GLOBAL_ENV.step_count,
        "history": GLOBAL_ENV.history,
        "done": GLOBAL_ENV.done,

        "task": GLOBAL_ENV.task,

        "expected_action": GLOBAL_ENV.task.get("expected_action"),
        "decision_hint": (
            "High urgency or negative sentiment → escalate"
            if GLOBAL_ENV.task.get("urgency") == "high" or GLOBAL_ENV.task.get("sentiment") < -0.5
            else "Medium urgency → request_info"
            if GLOBAL_ENV.task.get("urgency") == "medium"
            else "Low urgency → reply"
        )
    }


@app.get("/logs")
def get_logs():
    return {
        "resets": reset_history,
        "steps": step_history
    }


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).parent / "frontend.html"
    return html_path.read_text(encoding="utf-8")


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()