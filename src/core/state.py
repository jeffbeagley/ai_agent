from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    '''State shared across all agents in the graph'''
    messages: Annotated[Sequence[BaseMessage], "The conversation messages"]
    next: str  # Which agent to route to next