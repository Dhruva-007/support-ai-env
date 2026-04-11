import os
import random

from support_env.models import SupportObservation, SupportAction
from support_env.tasks import TASKS
from support_env.graders import grade_easy, grade_medium, grade_hard

# =========================
# IDEA 6: Optional deterministic seed for reproducible benchmark runs.
# Set RANDOM_SEED=42 (or any integer) to make ticket sampling deterministic.
# When not set, random behavior is completely unchanged.
# =========================
_RANDOM_SEED = os.getenv("RANDOM_SEED")
if _RANDOM_SEED is not None:
    random.seed(int(_RANDOM_SEED))


class SupportEnvironment:
    def __init__(self):
        self._init_task()
        self.step_count = 0
        self.done = False
        self.history = []
        self.episode_id = None
        self.stage = 1
        self.collected_info = False

    def _init_task(self, task_type=None):
        if task_type in ("easy", "medium", "hard"):
            # Filter tasks to only those matching the requested type
            filtered = [t for t in TASKS.values() if t["type"] == task_type]
            if filtered:
                self.task = random.choice(filtered)
                return
        # Default: pick any random task (used by frontend and fallback)
        self.task = random.choice(list(TASKS.values()))

    def reset(self, *args, **kwargs):
        self.episode_id = kwargs.get("episode_id", "default-episode")
        # Accept optional task_type so inference.py can target easy/medium/hard
        task_type = kwargs.get("task_type", None)
        self._init_task(task_type=task_type)

        self.step_count = 0
        self.done = False
        self.history = []
        self.stage = 1
        self.collected_info = False

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
            difficulty=self.task["type"],
            category=self.task.get("category", "general")
        )

    def step(self, action: SupportAction):
        self.step_count += 1
        self.history.append(action.action_type)

        expected = self.task["expected_action"]
        task_type = self.task["type"]

        # =========================
        # EASY
        # =========================
        if task_type == "easy":
            if action.action_type == expected:
                reward = 1.0
            else:
                reward = -0.3
            self.done = True

        # =========================
        # HARD
        # =========================
        elif task_type == "hard":
            if action.action_type == expected:
                reward = 0.8

                # =========================
                # IDEA 3: Context-aware bonus for high-stakes tickets.
                # Fraud and VIP tickets carry real-world priority — correctly
                # escalating them earns an extra +0.05 to reinforce priority triage.
                # Applies only on correct action, so wrong actions are unaffected.
                # =========================
                ctx = self.task.get("context", {})
                if ctx.get("fraud") or ctx.get("vip_user"):
                    reward += 0.05
            else:
                reward = -0.3
            self.done = True

        # =========================
        # MEDIUM → 2-step RL flow
        # =========================
        else:
            # Step 1 → must request_info
            if not self.collected_info:
                if action.action_type == "request_info":
                    reward = 0.3
                    self.collected_info = True
                    self.done = False
                else:
                    # Skipping to reply is less wrong than escalating a medium task
                    reward = -0.2 if action.action_type == "reply" else -0.3
                    self.done = True

            # Step 2 → must reply after collecting info
            else:
                if action.action_type == "reply":
                    reward = 0.9
                elif action.action_type == "request_info":
                    # Asking for info again — mild penalty
                    reward = -0.1
                    self.done = True
                else:
                    reward = -0.3
                self.done = True

        # =========================
        # LOOP PENALTY
        # =========================
        if (
            len(self.history) >= 2
            and self.history[-1] == self.history[-2]
        ):
            reward -= 0.1

        # =========================
        # SLA TIME BONUS
        # Rewards faster correct decisions — mirrors real support SLA targets.
        # Only applied when the action is correct (reward > 0).
        # Each step saved earns +0.05 bonus (max +0.15 at step 1).
        # =========================
        if reward > 0:
            time_bonus = max(0.0, (4 - self.step_count) * 0.05)
            reward += time_bonus

        # =========================
        # SAFETY STOP
        # =========================
        if self.step_count >= 4:
            self.done = True

        return SupportObservation(
            ticket_id="1",
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
            category=self.task.get("category", "general")
        )

    # =========================
    # SCORE
    # =========================
    def compute_score(self):
        task_type = self.task["type"]

        # For medium tasks, the final target action is "reply"
        # (the full correct path is request_info → reply)
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
