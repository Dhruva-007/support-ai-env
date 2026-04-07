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


# =============================
# CREATE OPENENV APP
# =============================
app = create_app(
    SupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_env",
    max_concurrent_envs=1,
)


# =============================
# 🔥 GLOBAL ENV (FIX)
# =============================
GLOBAL_ENV = SupportEnvironment()


# =============================
# GLOBAL LOG STORAGE
# =============================
reset_history = []
step_history = []

LOG_FILE = "logs.json"


def save_log(entry):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# =============================
# HEALTH
# =============================
@app.get("/health")
def health():
    return {"status": "ok"}


# =============================
# RESET (FIXED)
# =============================
@app.get("/reset")
async def reset_get():
    result = GLOBAL_ENV.reset()

    entry = {
        "type": "reset",
        "time": datetime.now().isoformat(),
        "observation": result.dict(),
        "reward": 0.0,
        "done": False
    }

    reset_history.append(entry)
    save_log(entry)

    return entry


# =============================
# STEP (FIXED)
# =============================
@app.post("/step_logged")
async def step_logged(action: dict):
    act = SupportAction(**action["action"])

    result = GLOBAL_ENV.step(act)

    entry = {
        "type": "step",
        "time": datetime.now().isoformat(),
        "action": action,
        "observation": result.dict(),
        "reward": result.reward,
        "done": result.done
    }

    step_history.append(entry)
    save_log(entry)

    return entry


# =============================
# STATE (FIXED)
# =============================
@app.get("/state")
def get_state():
    return GLOBAL_ENV.state


# =============================
# VIEW HISTORY
# =============================
@app.get("/logs")
def get_logs():
    return {
        "resets": reset_history,
        "steps": step_history
    }


# =============================
# FRONTEND ROUTE
# =============================
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).parent / "frontend.html"
    return html_path.read_text(encoding="utf-8")


# =============================
# MAIN (LOCAL RUN)
# =============================
def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()