"""
Fast Supervisor Agent System with Sub-Agents
LangGraph + FastAPI + Streaming for Open-WebUI
"""

import os
import json
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import asyncio

# Import from organized modules (in production these would be separate files)
# from config import Config
# from core.graph import create_agent_graph
# from core.state import AgentState

# Inline minimal imports for this demo
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

# ============= CONFIG =============
class Config:
    MODEL_NAME = "llama-3.1-storm-8b"
    MODEL_BASE_URL = os.getenv("OPENAI_HOST", "http://localhost:11434/v1")
    MODEL_TEMPERATURE = 0.7
    API_HOST = "0.0.0.0"
    API_PORT = 8000

# ============= STATE =============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The conversation messages"]
    next: str

# ============= TOOLS =============
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific location.
    
    Provides real-time weather information including temperature and conditions.
    
    Args:
        location: City and state/country (e.g., 'San Francisco, CA')
    
    Returns:
        Weather information string
    """
    return f"The weather in {location} is sunny and 72°F with clear skies."

WEATHER_TOOLS = [get_weather]

# ============= PROMPTS =============
SUPERVISOR_PROMPT = """You are a supervisor routing requests to specialized agents.

Available agents:
- weather_agent: Handles weather queries, forecasts, and meteorological information

Analyze the request and respond with ONLY the agent name or "FINISH":
- "weather_agent" for weather-related requests
- "FINISH" if you'll answer directly

Request: {request}

Decision (one word):"""

WEATHER_AGENT_PROMPT = """You are a weather specialist assistant.

Use the get_weather tool to provide accurate weather information.
Be concise, helpful, and always cite the specific location.

Guidelines:
- Use tools for current weather data
- Provide clear, actionable information
- Ask for clarification if needed"""

# ============= AGENT NODES =============
def get_model():
    return ChatOpenAI(
        model=Config.MODEL_NAME,
        base_url=Config.MODEL_BASE_URL,
        temperature=Config.MODEL_TEMPERATURE
    )

def supervisor_node(state: AgentState):
    """Supervisor routes to specialized agents"""
    messages = state["messages"]
    last_message = messages[-1].content
    
    model = get_model()
    prompt = SystemMessage(content=SUPERVISOR_PROMPT.format(request=last_message))
    
    response = model.invoke([prompt])
    decision = response.content.strip().lower()
    
    if "weather" in decision:
        return {"next": "weather_agent", "messages": messages}
    
    # Supervisor answers directly
    answer = model.invoke(messages)
    return {"next": "FINISH", "messages": messages + [answer]}

def weather_agent_node(state: AgentState):
    """Weather specialist with tools"""
    messages = state["messages"]
    
    model = get_model()
    model_with_tools = model.bind_tools(WEATHER_TOOLS)
    
    system_msg = SystemMessage(content=WEATHER_AGENT_PROMPT)
    response = model_with_tools.invoke([system_msg] + messages)
    
    return {"messages": [response]}

def execute_tools(state: AgentState):
    """Execute tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        for tool in WEATHER_TOOLS:
            if tool.name == tool_call["name"]:
                result = tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )
    
    return {"messages": tool_messages}

def format_response(state: AgentState):
    """Format final response"""
    messages = state["messages"]
    model = get_model()
    model_with_tools = model.bind_tools(WEATHER_TOOLS)
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "next": "FINISH"}

# ============= ROUTERS =============
def route_supervisor(state: AgentState) -> Literal["weather_agent", "__end__"]:
    return "weather_agent" if state.get("next") == "weather_agent" else "__end__"

def route_agent(state: AgentState) -> Literal["execute_tools", "__end__"]:
    last = state["messages"][-1]
    return "execute_tools" if hasattr(last, "tool_calls") and last.tool_calls else "__end__"

# ============= BUILD GRAPH =============
def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("weather_agent", weather_agent_node)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("format_response", format_response)
    
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges("supervisor", route_supervisor, 
                                   {"weather_agent": "weather_agent", "__end__": END})
    workflow.add_conditional_edges("weather_agent", route_agent, 
                                   {"execute_tools": "execute_tools", "__end__": END})
    workflow.add_edge("execute_tools", "format_response")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# ============= FASTAPI APP =============
app = FastAPI(title="Supervisor Agent API")
graph = create_graph()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible endpoint"""
    body = await request.json()
    
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
    
    # Handle metadata requests (non-streaming)
    last_msg = lc_messages[-1].content if lc_messages else ""
    is_metadata = "### Task:" in last_msg or "### Guidelines:" in last_msg
    
    if is_metadata and not stream:
        model = get_model()
        metadata_system = SystemMessage(content="Respond with ONLY valid JSON. No markdown, no extra text.")
        
        response = model.invoke([metadata_system] + lc_messages)
        content = response.content.strip()
        
        # Clean markdown
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            content = content.replace("```json", "").replace("```", "").strip()
        
        return JSONResponse(content={
            "id": conversation_id,
            "object": "chat.completion",
            "created": int(__import__('time').time()),
            "model": "supervisor-agent",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
    
    # Streaming conversation
    async def generate():
        full_response = ""
        try:
            config = {"configurable": {"thread_id": conversation_id}}
            
            async for event in graph.astream({"messages": lc_messages, "next": ""}, 
                                            config=config, stream_mode="values"):
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
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
