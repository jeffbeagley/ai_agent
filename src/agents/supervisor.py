from typing import Dict, List
from langchain_core.messages import SystemMessage
from agents.base import BaseAgent
from prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_ROUTING_PROMPT

class SupervisorAgent(BaseAgent):
    '''Supervisor agent that routes requests to specialized agents'''
    
    def __init__(self, available_agents: Dict[str, BaseAgent]):
        self.available_agents = available_agents
        
        # Build agent descriptions for the prompt
        agent_descriptions = "\\n".join([
            agent.get_description() for agent in available_agents.values()
        ])
        
        system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
            agent_descriptions=agent_descriptions
        )
        
        super().__init__(
            name="supervisor",
            description="Routes requests to specialized agents",
            system_prompt=system_prompt
        )
    
    def run(self, state):
        '''Analyze request and route to appropriate agent'''
        messages = state["messages"]
        last_message = messages[-1].content
        
        # Create routing prompt
        agent_names = list(self.available_agents.keys())
        
        # Simple routing logic - check for agent keywords
        routing_prompt = SystemMessage(
            content=self.system_prompt + "\\n\\n" + 
                   f"Request: {last_message}\\n\\nRoute to (agent name or FINISH):"
        )
        
        response = self.model.invoke([routing_prompt])
        decision = response.content.strip().lower()
        
        # Determine next agent
        if any(agent_name in decision for agent_name in agent_names):
            for agent_name in agent_names:
                if agent_name in decision:
                    return {"next": agent_name, "messages": messages}
        
        # Supervisor handles directly
        answer = self.model.invoke(messages)
        return {"next": "FINISH", "messages": messages + [answer]}