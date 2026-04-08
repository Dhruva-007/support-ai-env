import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.json")


def normalize_task(item):
    text = item["ticket"].lower()

    if "where is my order" in text or "not received" in text:
        item["expected_action"] = "request_info"

    # strong complaint signals → escalate
    if item["sentiment"] < -0.7 or item["urgency"] == "high":
        item["expected_action"] = "escalate"

    return item


def load_tasks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = {}
    for i, item in enumerate(data):
        item = normalize_task(item)
        tasks[f"task_{i}"] = item

    return tasks


TASKS = load_tasks()