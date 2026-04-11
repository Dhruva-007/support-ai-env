from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class SupportObservation(BaseModel):
    ticket_id: str
    customer_message: str
    history: List[str]
    sentiment: float
    urgency: str
    time_elapsed: int
    assigned_team: Optional[str]
    status: str

    reward: float = 0.0
    done: bool = False

    reason: Optional[str] = None
    difficulty: Optional[str] = None

    category: Optional[str] = None

    context: Optional[Dict[str, Any]] = None


class SupportAction(BaseModel):
    action_type: str  # "reply", "request_info", "escalate"
