from abc import ABC, abstractmethod
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from config import Config

class BaseAgent(ABC):
    '''Base class for all agents'''
    
    def __init__(self, name: str, description: str, system_prompt: str, tools: List[BaseTool] = None):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = self._create_model()
    
    def _create_model(self):
        '''Create the LLM model for this agent'''
        return ChatOpenAI(
            model=Config.MODEL_NAME,
            base_url=Config.MODEL_BASE_URL,
            temperature=Config.MODEL_TEMPERATURE
        )
    
    @abstractmethod
    def run(self, state):
        '''Execute the agent's logic'''
        pass
    
    def get_description(self) -> str:
        '''Get agent description for supervisor routing'''
        return f"- {self.name}: {self.description}"