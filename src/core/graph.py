from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, AIMessage
from core.state import AgentState
from agents.supervisor import SupervisorAgent
from agents.weather_agent import WeatherAgent
from tools.weather_tools import WEATHER_TOOLS

def create_agent_graph():
    '''Build the agent graph with supervisor and sub-agents'''
    
    # Initialize agents
    weather_agent = WeatherAgent()
    
    available_agents = {
        "weather_agent": weather_agent
    }
    
    supervisor = SupervisorAgent(available_agents)
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add supervisor node
    workflow.add_node("supervisor", supervisor.run)
    
    # Add agent nodes
    workflow.add_node("weather_agent", weather_agent.run)
    
    # Add tool execution node
    def execute_tools(state: AgentState):
        '''Execute any tool calls from agents'''
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_messages = []
        for tool_call in last_message.tool_calls:
            # Find and execute the tool
            for tool in WEATHER_TOOLS:
                if tool.name == tool_call["name"]:
                    result = tool.invoke(tool_call["args"])
                    tool_messages.append(
                        ToolMessage(content=result, tool_call_id=tool_call["id"])
                    )
        
        return {"messages": tool_messages}
    
    workflow.add_node("execute_tools", execute_tools)
    
    # Add formatting node
    def format_response(state: AgentState):
        '''Format final response after tool execution'''
        messages = state["messages"]
        
        # Get the weather agent's model with tools
        model_with_tools = weather_agent.model.bind_tools(weather_agent.tools)
        response = model_with_tools.invoke(messages)
        
        return {"messages": [response], "next": "FINISH"}
    
    workflow.add_node("format_response", format_response)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Define routing functions
    def route_supervisor(state: AgentState) -> Literal["weather_agent", "__end__"]:
        next_step = state.get("next", "FINISH")
        if next_step == "weather_agent":
            return "weather_agent"
        return "__end__"
    
    def route_agent(state: AgentState) -> Literal["execute_tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "execute_tools"
        return "__end__"
    
    # Add edges
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
    
    workflow.add_edge("execute_tools", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()