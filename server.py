"""
Fast Supervisor Agent System with Sub-Agents
LangGraph + FastAPI + Streaming for Open-WebUI
"""

import os
import json
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import functools

# ============= WEATHER TOOL =============
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'
    """
    return f"The weather in {location} is sunny and 72°F with clear skies."


# ============= AGENT STATE =============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: str  # Which agent to call next


# ============= MODEL =============
def get_model():
    return ChatOpenAI(
        model="llama-3.1-storm-8b",
        base_url=os.getenv("OPENAI_HOST", "http://localhost:11434/v1"),
        temperature=0.7
    )


# ============= SUPERVISOR NODE =============
def supervisor_node(state: AgentState):
    """Supervisor decides which agent to route to"""
    messages = state["messages"]
    
    supervisor_prompt = """You are a supervisor routing assistant requests to specialized agents.

Available agents:
- weather_agent: Handles all weather-related queries (current weather, forecasts, etc.)
- FINISH: Direct questions you can answer without specialized agents

Analyze this request and respond with ONLY ONE WORD:
- "weather_agent" if it's about weather
- "FINISH" if you should answer directly

Request: {request}

Your decision (one word):"""
    
    last_message = messages[-1].content
    
    model = get_model()
    response = model.invoke([
        SystemMessage(content=supervisor_prompt.format(request=last_message))
    ])
    
    decision = response.content.strip().lower()
    
    if "weather" in decision:
        return {"next": "weather_agent", "messages": messages}
    else:
        # Supervisor answers directly
        answer = model.invoke(messages)
        return {"next": "FINISH", "messages": messages + [answer]}


# ============= WEATHER AGENT NODE =============
def weather_agent_node(state: AgentState):
    """Weather specialist agent with tools"""
    messages = state["messages"]
    
    model = get_model()
    model_with_tools = model.bind_tools([get_weather])
    
    weather_system = SystemMessage(content="""You are a weather specialist. 
Use the get_weather tool to provide accurate weather information.
Be concise and helpful.""")
    
    response = model_with_tools.invoke([weather_system] + messages)
    
    return {"messages": [response]}


# ============= TOOL EXECUTION NODE =============
def execute_tools(state: AgentState):
    """Execute any tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    from langchain_core.messages import ToolMessage
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "get_weather":
            result = get_weather.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )
    
    return {"messages": tool_messages}


# ============= CALL AGENT AFTER TOOLS =============
def call_weather_agent_after_tools(state: AgentState):
    """After tools execute, call weather agent again to format response"""
    messages = state["messages"]
    
    model = get_model()
    model_with_tools = model.bind_tools([get_weather])
    
    response = model_with_tools.invoke(messages)
    
    return {"messages": [response], "next": "FINISH"}


# ============= ROUTERS =============
def route_supervisor(state: AgentState) -> Literal["weather_agent", "__end__"]:
    """Route from supervisor"""
    next_step = state.get("next", "FINISH")
    if next_step == "weather_agent":
        return "weather_agent"
    return "__end__"


def route_weather_agent(state: AgentState) -> Literal["execute_tools", "format_response"]:
    """Route from weather agent - check if tools needed"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return "format_response"


def route_after_tools(state: AgentState) -> Literal["call_agent", "__end__"]:
    """After tools, call agent to format or finish"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # After tool execution, always call agent once more to format
    if isinstance(last_message, type(last_message)) and hasattr(last_message, "tool_call_id"):
        return "call_agent"
    
    return "__end__"


# ============= BUILD GRAPH =============
def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("weather_agent", weather_agent_node)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("call_agent", call_weather_agent_after_tools)
    
    # Entry point
    workflow.set_entry_point("supervisor")
    
    # Supervisor routes to weather_agent or ends
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "weather_agent": "weather_agent",
            "__end__": END
        }
    )
    
    # Weather agent either uses tools or formats response
    workflow.add_conditional_edges(
        "weather_agent",
        route_weather_agent,
        {
            "execute_tools": "execute_tools",
            "format_response": END
        }
    )
    
    # After tools execute, call agent to format
    workflow.add_edge("execute_tools", "call_agent")
    
    # After formatting, end
    workflow.add_edge("call_agent", END)
    
    return workflow.compile()


# ============= FASTAPI APP =============
app = FastAPI(title="Supervisor Agent API")

graph = create_graph()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint"""
    body = await request.json()
    
    print("="*80)
    print("REQUEST BODY:")
    print(json.dumps(body, indent=2))
    print("="*80)
    
    messages = body.get("messages", [])
    conversation_id = body.get("conversation_id", "default")
    stream = body.get("stream", True)
    
    # Convert to LangChain messages
    lc_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "system":
            lc_messages.append(SystemMessage(content=content))
    
    # Check if this is a metadata request (non-streaming)
    last_msg = lc_messages[-1].content if lc_messages else ""
    is_metadata = "### Task:" in last_msg or "### Guidelines:" in last_msg
    
    if is_metadata and not stream:
        print("DETECTED NON-STREAMING METADATA REQUEST")
        
        model = get_model()
        metadata_system = SystemMessage(content="""You must respond with ONLY valid JSON. 
No markdown, no code blocks, no extra text. Just pure JSON starting with { and ending with }.""")
        
        response = model.invoke([metadata_system] + lc_messages)
        content = response.content.strip()
        
        print(f"RAW METADATA RESPONSE: {content}")
        
        # Clean up markdown
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            content = content.replace("```json", "").replace("```", "").strip()
        
        print(f"CLEANED METADATA RESPONSE: {content}")
        
        # Return non-streaming OpenAI format
        return {
            "id": conversation_id,
            "object": "chat.completion",
            "created": int(__import__('time').time()),
            "model": "supervisor-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    
    # For streaming requests
    async def generate():
        """Stream responses"""
        full_response = ""
        
        try:
            # Run graph for regular requests
            print("RUNNING GRAPH FOR REGULAR REQUEST")
            config = {"configurable": {"thread_id": conversation_id}}
            
            async for event in graph.astream(
                {"messages": lc_messages, "next": ""},
                config=config,
                stream_mode="values"
            ):
                msgs = event.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if isinstance(last, AIMessage) and last.content:
                        content = last.content
                        
                        if content != full_response:
                            chunk = content[len(full_response):]
                            full_response = content
                            
                            data = {
                                "id": conversation_id,
                                "object": "chat.completion.chunk",
                                "created": int(__import__('time').time()),
                                "model": "supervisor-agent",
                                "choices": [{
                                    "delta": {"content": chunk},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                            await asyncio.sleep(0.01)
            
            # Finish
            data = {
                "id": conversation_id,
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": "supervisor-agent",
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"Error: {str(e)}"
            data = {
                "id": conversation_id,
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": "supervisor-agent",
                "choices": [{
                    "delta": {"content": error_msg},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)