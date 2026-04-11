from support_env.models import SupportObservation, SupportAction
from support_env.tasks import TASKS
from support_env.graders import grade_easy, grade_medium, grade_hard
import random

_episode_counter = 0


class SupportEnvironment:
    def __init__(self):
        self._init_task()
        self.step_count = 0
        self.done = False
        self.history = []
        self.episode_id = None
        self.stage = 1
        self.collected_info = False
        self.ticket_id = "0"

    def _init_task(self, task_type=None):
        if task_type in ("easy", "medium", "hard"):
            filtered = [t for t in TASKS.values() if t["type"] == task_type]
            if filtered:
                self.task = random.choice(filtered)
                return
        self.task = random.choice(list(TASKS.values()))

    def reset(self, *args, **kwargs):
        global _episode_counter
        _episode_counter += 1

        self.episode_id = kwargs.get("episode_id", f"episode-{_episode_counter}")
        task_type = kwargs.get("task_type", None)
        self._init_task(task_type=task_type)

        self.step_count = 0
        self.done = False
        self.history = []
        self.stage = 1
        self.collected_info = False
        self.ticket_id = str(_episode_counter)

        return SupportObservation(
            ticket_id=self.ticket_id,
            customer_message=self.task["ticket"],
            history=[],
            sentiment=self.task["sentiment"],
            urgency=self.task["urgency"],
            time_elapsed=0,
            assigned_team=None,
            status="open",
            reward=0.0,
            done=False,
            reason=None,
            difficulty=self.task["type"],
            category=self.task.get("category", "general"),
            context=self.task.get("context", {})
        )

    def step(self, action: SupportAction):
        self.step_count += 1
        self.history.append(action.action_type)

        expected = self.task["expected_action"]
        task_type = self.task["type"]


        if task_type == "easy":
            if action.action_type == expected:
                reward = 1.0
            else:
                reward = -0.3
            self.done = True


        elif task_type == "hard":
            if action.action_type == expected:
                reward = 0.8
            else:
                reward = -0.3
            self.done = True


        else:
            if not self.collected_info:
                if action.action_type == "request_info":
                    reward = 0.3
                    self.collected_info = True
                    self.done = False
                else:
                    reward = -0.2 if action.action_type == "reply" else -0.3
                    self.done = True

            else:
                if action.action_type == "reply":
                    reward = 0.9
                elif action.action_type == "request_info":
                    reward = -0.1
                    self.done = True
                else:
                    reward = -0.3
                self.done = True


        if (
            len(self.history) >= 2
            and self.history[-1] == self.history[-2]
        ):
            reward -= 0.1

        if reward > 0:
            time_bonus = max(0.0, (4 - self.step_count) * 0.05)
            reward += time_bonus


        if self.step_count >= 4:
            self.done = True

        return SupportObservation(
            ticket_id=self.ticket_id,
            customer_message=self.task["ticket"],
            history=self.history,
            sentiment=self.task["sentiment"],
            urgency=self.task["urgency"],
            time_elapsed=self.step_count,
            assigned_team=None,
            status="resolved" if self.done else "open",
            reward=round(reward, 2),
            done=self.done,
            reason=f"Expected {expected}, got {action.action_type}",
            difficulty=self.task["type"],
            category=self.task.get("category", "general"),
            context=self.task.get("context", {})
        )


    def compute_score(self):
        task_type = self.task["type"]

        if task_type == "medium":
            expected = "reply"
        else:
            expected = self.task["expected_action"]

        if task_type == "easy":
            score = grade_easy(self.history, expected)
        elif task_type == "medium":
            score = grade_medium(self.history, expected)
        else:
            score = grade_hard(
                self.history,
                expected,
                self.task["sentiment"]
            )

        return max(0.01, min(score, 0.99))

    def get_score(self):
        return {"score": self.compute_score()}

    @property
    def state(self):
        return {
            "task": self.task,
            "history": self.history,
            "expected_action": self.task["expected_action"],
            "stage": self.stage,
            "collected_info": self.collected_info
        }

    async def reset_async(self, *args, **kwargs):
        return self.reset(*args, **kwargs)

    async def step_async(self, action: SupportAction):
        return self.step(action)

    def close(self):
        pass
