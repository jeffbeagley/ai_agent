"""Weather specialist agent."""

from langchain_core.messages import SystemMessage

from agents.base import BaseAgent
from tools.weather_tools import WEATHER_TOOLS
from prompts.weather import WEATHER_AGENT_SYSTEM_PROMPT


class WeatherAgent(BaseAgent):
    """
    Specialized agent for handling weather-related queries.
    
    This agent has access to weather tools and is optimized for
    providing accurate, helpful weather information to users.
    """
    
    def __init__(self):
        """Initialize the weather agent with appropriate tools and prompts."""
        super().__init__(
            name="weather_agent",
            description="Handles all weather-related queries including current conditions, forecasts, and weather advice",
            system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
            tools=WEATHER_TOOLS
        )
    
    def run(self, state: dict) -> dict:
        """
        Process weather-related requests using available tools.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with agent's response
        """
        messages = state["messages"]
        
        # Bind tools to the model
        model_with_tools = self.model.bind_tools(self.tools)
        
        # Create system message with agent's prompt
        system_message = SystemMessage(content=self.system_prompt)
        
        # Invoke model with tools
        response = model_with_tools.invoke([system_message] + messages)
        
        return {"messages": [response]}