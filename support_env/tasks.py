import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.json")


def load_tasks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = {}
    for i, item in enumerate(data):
        tasks[f"task_{i}"] = item

    return tasks


TASKS = load_tasks()