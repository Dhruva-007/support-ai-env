import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.json")


REQUIRED_FIELDS = ["ticket", "sentiment", "urgency", "expected_action"]


def validate_task(item, idx):
    for field in REQUIRED_FIELDS:
        if field not in item:
            raise ValueError(f"Task {idx} missing required field: {field}")

    item["urgency"] = item["urgency"].lower().strip()
    item["expected_action"] = item["expected_action"].lower().strip()

    if item["urgency"] not in ["low", "medium", "high"]:
        raise ValueError(f"Invalid urgency in task {idx}")

    if item["expected_action"] not in ["reply", "request_info", "escalate"]:
        raise ValueError(f"Invalid expected_action in task {idx}")

    return item


def assign_task_type(item):
    if item["urgency"] == "low":
        item["type"] = "easy"
    elif item["urgency"] == "medium":
        item["type"] = "medium"
    else:
        item["type"] = "hard"
    return item


def load_tasks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = {}

    for i, item in enumerate(data):
        item = validate_task(item, i)
        item = assign_task_type(item)

        tasks[f"task_{i}"] = item

    if len(tasks) < 3:
        raise ValueError("At least 3 tasks required for validation")

    return tasks


TASKS = load_tasks()