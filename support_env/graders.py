def _safe(score: float) -> float:
    return max(0.01, min(score, 0.99))


def grade_easy(history, expected):
    if not history:
        return 0.1

    if history[0] == expected:
        return 0.95

    return 0.1


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
        return 0.1

    if len(history) >= 2 and history[0] == "request_info" and history[1] == expected:
        return 0.95

    if len(history) >= 2 and history[0] == "request_info" and history[1] != expected:
        return 0.30

    if history[0] == expected:
        return 0.40

    return 0.10


def grade_hard(history, expected, sentiment):
    """
    Hard tasks require immediate escalation.
    Sentiment bonus rewards recognising highly distressed customers.
    """
    if not history:
        return 0.1

    score = 0.3

    if history[0] == expected:
        score += 0.5

    if sentiment < -0.5 and history[0] == "escalate":
        score += 0.1

    return _safe(score)
