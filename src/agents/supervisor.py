"""Supervisor agent for routing requests."""

from typing import Dict
from langchain_core.messages import SystemMessage

from agents.base import BaseAgent
from prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_ROUTING_PROMPT


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that analyzes requests and routes to specialized agents.
    
    The supervisor is the entry point for all user requests and decides
    which specialized agent should handle each request, or whether to
    handle it directly.
    """
    
    def __init__(self, available_agents: Dict[str, BaseAgent]):
        """
        Initialize the supervisor with available sub-agents.
        
        Args:
            available_agents: Dictionary mapping agent names to agent instances
        """
        self.available_agents = available_agents
        
        # Build agent descriptions for the system prompt
        agent_descriptions = "\n".join([
            agent.get_description() 
            for agent in available_agents.values()
        ])
        
        system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
            agent_descriptions=agent_descriptions
        )
        
        super().__init__(
            name="supervisor",
            description="Routes requests to specialized agents or handles directly",
            system_prompt=system_prompt
        )
    
    def run(self, state: dict) -> dict:
        """
        Analyze the request and route to the appropriate agent.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with routing decision
        """
        messages = state["messages"]
        last_message = messages[-1].content
        
        # Create routing prompt with the user's request
        routing_prompt = SUPERVISOR_ROUTING_PROMPT.format(
            request=last_message
        )
        
        prompt_message = SystemMessage(
            content=self.system_prompt + "\n\n" + routing_prompt
        )
        
        # Get routing decision from LLM
        response = self.model.invoke([prompt_message])
        decision = response.content.strip().lower()
        
        # Check if routing to any available agent
        for agent_name in self.available_agents.keys():
            if agent_name in decision:
                return {"next": agent_name, "messages": messages}
        
        # Supervisor handles the request directly
        direct_answer = self.model.invoke(messages)
        return {"next": "FINISH", "messages": messages + [direct_answer]}