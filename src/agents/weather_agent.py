from langchain_core.messages import SystemMessage
from agents.base import BaseAgent
from tools.weather_tools import WEATHER_TOOLS
from prompts.weather import WEATHER_AGENT_SYSTEM_PROMPT

class WeatherAgent(BaseAgent):
    '''Specialized agent for weather-related queries'''
    
    def __init__(self):
        super().__init__(
            name="weather_agent",
            description="Handles all weather-related queries including current conditions, forecasts, and weather advice",
            system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
            tools=WEATHER_TOOLS
        )
    
    def run(self, state):
        '''Process weather-related requests'''
        messages = state["messages"]
        
        # Bind tools to model
        model_with_tools = self.model.bind_tools(self.tools)
        
        # Add system prompt
        system_message = SystemMessage(content=self.system_prompt)
        
        # Invoke with tools
        response = model_with_tools.invoke([system_message] + messages)
        
        return {"messages": [response]}