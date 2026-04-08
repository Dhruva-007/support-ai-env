from support_env.models import SupportObservation, SupportAction
from support_env.tasks import TASKS
import random


class SupportEnvironment:
    def __init__(self):
        self._init_task()
        self.step_count = 0
        self.done = False
        self.history = []
        self.episode_id = None

    def _init_task(self):
        self.task = random.choice(list(TASKS.values()))

    def reset(self, *args, **kwargs):
        self.episode_id = kwargs.get("episode_id", "default-episode")
        self._init_task()

        self.step_count = 0
        self.done = False
        self.history = []

        return SupportObservation(
            ticket_id="1",
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
            difficulty=self.task["urgency"]
        )

    def step(self, action: SupportAction):

        self.step_count += 1
        self.history.append(action.action_type)

        expected = self.task["expected_action"]

        if action.action_type == expected:
            base = 1.0

            if self.task["urgency"] == "high":
                base += 0.8
            elif self.task["urgency"] == "medium":
                base += 0.5
            else:
                base += 0.2

            speed_bonus = max(0, 0.3 - 0.05 * self.step_count)
            reward = base + speed_bonus

            if expected == "escalate" and self.task["sentiment"] < -0.5:
                reward += 0.2

            self.done = True

        else:
            if self.task["urgency"] == "high":
                reward = -0.6
            elif self.task["urgency"] == "medium":
                reward = -0.4
            else:
                reward = -0.2

            if self.task["urgency"] == "high" and action.action_type == "reply":
                reward -= 0.3

        if action.action_type == "escalate" and self.task["urgency"] == "low":
            reward -= 0.4

        if action.action_type == "reply" and self.task["urgency"] == "high":
            reward -= 0.4

        if action.action_type == "escalate" and self.task["sentiment"] < -0.5:
            reward += 0.3

        if action.action_type == "reply" and self.task["sentiment"] > 0:
            reward += 0.2

        reward -= 0.05

        if len(self.history) >= 2 and self.history[-1] == self.history[-2]:
            reward -= 0.3

        if len(self.history) >= 2 and self.history[-1] != self.history[-2]:
            reward -= 0.1

        if self.step_count >= 5:
            self.done = True

        reason = (
            f"Expected '{expected}' due to urgency='{self.task['urgency']}' "
            f"and sentiment={self.task['sentiment']}. "
            f"Action taken: '{action.action_type}'."
        )

        obs = SupportObservation(
            ticket_id="1",
            customer_message=self.task["ticket"],
            history=self.history,
            sentiment=self.task["sentiment"],
            urgency=self.task["urgency"],
            time_elapsed=self.step_count,
            assigned_team=None,
            status="resolved" if self.done else "open",
            reward=reward,
            done=self.done,
            reason=reason,
            difficulty=self.task["urgency"]
        )

        return obs

    # ✅ CRITICAL FIX HERE
    def compute_score(self):
        # USE TASK GRADER DIRECTLY
        grader = self.task.get("grader")

        if grader is None:
            return 0.5  # safe fallback (never 0 or 1)

        try:
            score = grader(self.history)
            return max(0.01, min(score, 0.99))
        except Exception:
            return 0.5

    @property
    def state(self):
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "history": self.history,
            "step_count": self.step_count,
            "done": self.done,
            "expected_action": self.task["expected_action"],
        }

    def get_metadata(self):
        return {
            "name": "Support AI RL Environment",
            "description": "RL environment for customer support decision making",
            "version": "1.0",
            "author": "G Dhruvann",
        }

    async def reset_async(self, *args, **kwargs):
        return self.reset(*args, **kwargs)

    async def step_async(self, action: SupportAction):
        return self.step(action)

    def close(self):
        pass