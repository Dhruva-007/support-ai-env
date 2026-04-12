def _safe(score: float) -> float:
    return max(0.01, min(score, 0.99))


def grade_easy(history, expected):
    """
    Easy tasks require a direct reply to a simple informational question.

    Grading reflects action severity:
    - Correct (reply):       0.95  ← answered directly, ideal
    - Over-cautious (request_info): 0.25 ← unnecessary investigation, not harmful
    - Over-aggressive (escalate):   0.10 ← completely wrong, alarming for an FAQ
    """
    if not history:
        return 0.10

    if history[0] == expected:         
        return 0.95

    if history[0] == "request_info":   
        return 0.25

    return 0.10                       


def grade_medium(history, expected):
    """
    Medium tasks require a 2-step flow: request_info → reply
    Grading reflects the full trajectory quality:

    - Full correct path (request_info → reply):  0.95  ← best
    - Skipped to reply directly (wrong RL flow):  0.40  ← penalised: skipped required step
    - Wrong first action, then correct reply:     0.30  ← partial
    - Anything else:                              0.10  ← wrong
    """
    if not history:
        return 0.10

    if len(history) >= 2 and history[0] == "request_info" and history[1] == expected:
        return 0.95

    if len(history) >= 2 and history[0] == "request_info" and history[1] != expected:
        return 0.30

    if history[0] == expected:
        return 0.40

    return 0.10


def grade_hard(history, expected, sentiment):
    """
    Hard tasks require escalation. Urgency is hidden — the agent must reason
    from ticket text, sentiment, category and context flags alone.

    Two valid paths:
      Optimal:  escalate immediately (step 1)
      Partial:  request_info → escalate (agent needed more context first)

    Grading:
    - Immediate escalate (step 1 correct):             0.80 or 0.90
      + sentiment bonus (+0.10) if sentiment < -0.5
    - Delayed escalate  (request_info → escalate):     0.65
      Correct final action but missed that urgency=unknown means escalate now
    - Gathered info then wrong action:                 0.20
    - reply as first action (most wrong — ignored severity):  0.10
    - Anything else wrong:                             0.10
    """
    if not history:
        return 0.10

    if history[0] == expected:
        score = 0.80
        if sentiment < -0.5 and history[0] == "escalate":
            score += 0.10
        return _safe(score)

    if len(history) >= 2 and history[0] == "request_info" and history[1] == expected:
        return 0.65

    if len(history) >= 2 and history[0] == "request_info" and history[1] != expected:
        return 0.20

    if history[0] == "reply":
        return 0.10

    return 0.10
