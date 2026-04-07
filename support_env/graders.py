def _clamp(score: float) -> float:
    return max(0.0, min(score, 1.0))


def grade_easy(history, expected):
    """
    Easy:
    - Strong reward for first-step correctness
    """
    if not history:
        return 0.0

    if history[0] == expected:
        return 1.0

    for i, action in enumerate(history[:3]):
        if action == expected:
            return _clamp(0.7 - i * 0.2)

    return 0.0


def grade_medium(history, expected):
    """
    Medium:
    - Reward correctness
    - Strong penalty for delay
    """
    for i, action in enumerate(history):
        if action == expected:
            score = 1.0 - (i * 0.25)  # 🔥 stronger penalty
            return _clamp(score)

    return 0.0


def grade_hard(history, expected, sentiment, urgency):
    """
    Hard:
    - Correctness
    - Speed
    - Context awareness
    - First-step priority
    """
    if not history:
        return 0.0

    score = 0.0

    # correctness
    if expected in history:
        score += 0.5

    # speed (new stronger signal)
    if history[0] == expected:
        score += 0.4
    elif len(history) <= 2:
        score += 0.2

    # context awareness
    if sentiment < -0.5 and "escalate" in history:
        score += 0.1

    return _clamp(score)