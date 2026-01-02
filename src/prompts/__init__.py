"""Prompt templates for all agents."""

from .supervisor import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_ROUTING_PROMPT
from .weather import WEATHER_AGENT_SYSTEM_PROMPT

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "SUPERVISOR_ROUTING_PROMPT", 
    "WEATHER_AGENT_SYSTEM_PROMPT"
]