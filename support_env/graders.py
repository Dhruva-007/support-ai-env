def _clamp(score: float) -> float:
    # STRICT: ensure score is always between (0,1)
    return max(0.01, min(score, 0.99))


def grade_easy(history, expected):
    if not history:
        return 0.05

    if history[0] == expected:
        return 0.95

    for i, action in enumerate(history[:3]):
        if action == expected:
            return _clamp(0.7 - i * 0.2)

    return 0.05


def grade_medium(history, expected):
    for i, action in enumerate(history):
        if action == expected:
            score = 0.9 - (i * 0.25)
            return _clamp(score)

    return 0.05


def grade_hard(history, expected, sentiment, urgency):
    if not history:
        return 0.05

    score = 0.05  # base score (never 0)

    # correctness
    if expected in history:
        score += 0.5

    # speed
    if history[0] == expected:
        score += 0.3
    elif len(history) <= 2:
        score += 0.2

    # context awareness
    if sentiment < -0.5 and "escalate" in history:
        score += 0.1

    return _clamp(score)