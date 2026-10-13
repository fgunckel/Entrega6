"""LangGraph shared state.

Each node should update only the fields it owns, returning partial state updates.
"""

from typing import Literal, NotRequired

from langgraph.graph import MessagesState
from typing_extensions import TypedDict

AgentName = Literal[
    "researcher",
    "analyst",
    "validator",
    "synthesizer",
    "limited_finish",
    "END",
]


class AgentContribution(TypedDict):
    agent: str
    artifact_type: str
    summary: str
    step: int


class AgentState(MessagesState):
    user_request: str
    system_profile: dict
    criteria: list[dict]

    research_artifact: NotRequired[dict | None]
    analysis_artifact: NotRequired[dict | None]
    validation_result: NotRequired[dict | None]
    contributions: NotRequired[list[AgentContribution]]

    next_agent: AgentName
    routing_reason: str
    steps: int
    retries: int
    task_completed: bool
    final_report: NotRequired[dict | None]
