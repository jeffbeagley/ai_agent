"""Graph construction for the agent system."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, AIMessage

from core.state import AgentState
from agents.supervisor import SupervisorAgent
from agents.weather_agent import WeatherAgent
from tools.weather_tools import WEATHER_TOOLS
from config import Config


def create_agent_graph():
    """
    Build and compile the agent conversation graph.
    
    The graph defines how requests flow through the system:
    1. Start at supervisor
    2. Route to specialized agents as needed
    3. Execute tools when required
    4. Format and return responses
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Initialize all agents
    weather_agent = WeatherAgent()
    
    available_agents = {
        "weather_agent": weather_agent,
        # Add new agents here
    }
    
    supervisor = SupervisorAgent(available_agents)
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("supervisor", supervisor.run)
    workflow.add_node("weather_agent", weather_agent.run)
    
    # Tool execution node
    def execute_tools(state: AgentState) -> dict:
        """
        Execute any tool calls from the last message.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with tool results
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_messages = []
        
        # Execute each tool call
        for tool_call in last_message.tool_calls:
            # Find matching tool
            for tool in WEATHER_TOOLS:  # Extend with more tool registries
                if tool.name == tool_call["name"]:
                    result = tool.invoke(tool_call["args"])
                    tool_messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        )
                    )
                    break
        
        return {"messages": tool_messages}
    
    workflow.add_node("execute_tools", execute_tools)
    
    # Response formatting node
    def format_response(state: AgentState) -> dict:
        """
        Format the final response after tool execution.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with formatted response
        """
        messages = state["messages"]
        
        # Use the weather agent to format the response
        model_with_tools = weather_agent.model.bind_tools(weather_agent.tools)
        response = model_with_tools.invoke(messages)
        
        return {"messages": [response], "next": "FINISH"}
    
    workflow.add_node("format_response", format_response)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Define routing logic
    def route_supervisor(state: AgentState) -> Literal["weather_agent", "__end__"]:
        """Route from supervisor to agent or end."""
        next_step = state.get("next", "FINISH")
        if next_step == "weather_agent":
            return "weather_agent"
        return "__end__"
    
    def route_agent(state: AgentState) -> Literal["execute_tools", "__end__"]:
        """Route from agent to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "execute_tools"
        return "__end__"
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "weather_agent": "weather_agent",
            "__end__": END
        }
    )
    
    workflow.add_conditional_edges(
        "weather_agent",
        route_agent,
        {
            "execute_tools": "execute_tools",
            "__end__": END
        }
    )
    
    # Add direct edges
    workflow.add_edge("execute_tools", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()