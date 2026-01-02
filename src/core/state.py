"""Agent state definitions."""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State shared across all agents in the conversation graph.
    
    Attributes:
        messages: The conversation message history
        next: Which agent should handle the next step
    """
    messages: Annotated[Sequence[BaseMessage], "The conversation messages"]
    next: str