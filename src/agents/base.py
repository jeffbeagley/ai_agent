"""Base agent class for all specialized agents."""

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from config import Config


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    All specialized agents should inherit from this class and implement
    the run() method with their specific logic.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: Optional[List[BaseTool]] = None
    ):
        """
        Initialize a new agent.
        
        Args:
            name: Unique identifier for this agent
            description: Human-readable description of agent's purpose
            system_prompt: System prompt that defines agent behavior
            tools: List of tools available to this agent
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = self._create_model()
    
    def _create_model(self) -> ChatOpenAI:
        """
        Create and configure the LLM model for this agent.
        
        Returns:
            Configured ChatOpenAI model instance
        """
        return ChatOpenAI(
            model=Config.MODEL_NAME,
            base_url=Config.MODEL_BASE_URL,
            temperature=Config.MODEL_TEMPERATURE
        )
    
    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Execute the agent's main logic.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state after agent execution
        """
        pass
    
    def get_description(self) -> str:
        """
        Get formatted description for supervisor routing.
        
        Returns:
            Formatted string describing this agent
        """
        return f"- {self.name}: {self.description}"