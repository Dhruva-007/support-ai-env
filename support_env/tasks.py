import json
import os
from support_env.graders import grade_easy, grade_medium, grade_hard

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tickets.json")


def normalize_task(item):
    text = item["ticket"].lower()

    if "where is my order" in text or "not received" in text:
        item["expected_action"] = "request_info"

    if item["sentiment"] < -0.7 or item["urgency"] == "high":
        item["expected_action"] = "escalate"

    return item


def load_tasks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = {}

    for i, item in enumerate(data):
        item = normalize_task(item)

        # attach grader (IMPORTANT FIX)
        if item["urgency"] == "low":
            item["grader"] = lambda history, expected=item["expected_action"]: grade_easy(history, expected)

        elif item["urgency"] == "medium":
            item["grader"] = lambda history, expected=item["expected_action"]: grade_medium(history, expected)

        else:
            item["grader"] = lambda history, expected=item["expected_action"], s=item["sentiment"], u=item["urgency"]: grade_hard(history, expected, s, u)

        tasks[f"task_{i}"] = item

    return tasks


TASKS = load_tasks()