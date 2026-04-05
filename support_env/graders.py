def _clamp(score: float) -> float:
    """Ensure score is always between 0 and 1."""
    return max(0.0, min(score, 1.0))


def grade_easy(history, expected):
    """
    Easy:
    - Correct action within first 3 steps
    """
    for i, action in enumerate(history[:3]):
        if action == expected:
            return _clamp(1.0)
    return 0.0


def grade_medium(history, expected):
    """
    Medium:
    - Correct action
    - Penalize delay
    """
    for i, action in enumerate(history):
        if action == expected:
            score = 1.0 - (i * 0.2)
            return _clamp(score)
    return 0.0


def grade_hard(history, expected, sentiment, urgency):
    """
    Hard:
    - Correctness
    - Speed
    - Context awareness
    """
    if not history:
        return 0.0

    score = 0.0

    if expected in history:
        score += 0.5

    if len(history) <= 2:
        score += 0.3

    if sentiment < -0.5 and "escalate" in history:
        score += 0.2

    return _clamp(score)