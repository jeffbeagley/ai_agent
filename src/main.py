"""
FastAPI server for the Supervisor Agent System.

This module provides an OpenAI-compatible API endpoint for interacting
with the multi-agent system through Open-WebUI or other compatible clients.
"""

import json
import asyncio
import logging
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import Config
from core.graph import create_agent_graph

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate configuration
Config.validate()

# Initialize FastAPI app
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description="Multi-agent system with supervisor routing"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent graph
graph = create_agent_graph()
logger.info("Agent graph initialized successfully")


def convert_to_langchain_messages(messages: List[dict]) -> List:
    """
    Convert OpenAI format messages to LangChain format.
    
    Args:
        messages: List of messages in OpenAI format
        
    Returns:
        List of LangChain message objects
    """
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
    
    return lc_messages


def is_metadata_request(content: str) -> bool:
    """
    Check if a request is for metadata generation.
    
    Args:
        content: Message content to check
        
    Returns:
        True if this is a metadata request
    """
    return "### Task:" in content or "### Guidelines:" in content


async def handle_metadata_request(lc_messages: List, conversation_id: str) -> JSONResponse:
    """
    Handle metadata generation requests (follow-ups, title, tags).
    
    Args:
        lc_messages: LangChain format messages
        conversation_id: Unique conversation identifier
        
    Returns:
        JSONResponse with metadata
    """
    logger.info("Processing metadata request")
    
    model = ChatOpenAI(
        model=Config.MODEL_NAME,
        base_url=Config.MODEL_BASE_URL,
        temperature=Config.MODEL_TEMPERATURE
    )
    
    metadata_system = SystemMessage(
        content="Respond with ONLY valid JSON. No markdown, no extra text."
    )
    
    response = model.invoke([metadata_system] + lc_messages)
    content = response.content.strip()
    
    # Clean markdown formatting
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        content = content.replace("```json", "").replace("```", "").strip()
    
    logger.debug(f"Metadata response: {content}")
    
    return JSONResponse(content={
        "id": conversation_id,
        "object": "chat.completion",
        "created": int(__import__('time').time()),
        "model": Config.MODEL_NAME,
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
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming and non-streaming responses.
    Handles regular conversations and metadata generation.
    """
    body = await request.json()
    
    messages = body.get("messages", [])
    conversation_id = body.get("conversation_id", "default")
    stream = body.get("stream", True)
    
    logger.info(f"Request - ID: {conversation_id}, Stream: {stream}, Messages: {len(messages)}")
    
    # Convert to LangChain format
    lc_messages = convert_to_langchain_messages(messages)
    
    # Handle metadata requests (non-streaming)
    last_msg = lc_messages[-1].content if lc_messages else ""
    if is_metadata_request(last_msg) and not stream:
        return await handle_metadata_request(lc_messages, conversation_id)
    
    # Handle streaming conversation
    async def generate():
        """Generate streaming response."""
        full_response = ""
        
        try:
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
                                "model": Config.MODEL_NAME,
                                "choices": [{
                                    "delta": {"content": chunk},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                            await asyncio.sleep(0.01)
            
            # Send finish token
            data = {
                "id": conversation_id,
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": Config.MODEL_NAME,
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
            
            logger.info(f"Response completed - ID: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            error_data = {
                "id": conversation_id,
                "object": "chat.completion.chunk",
                "created": int(__import__('time').time()),
                "model": Config.MODEL_NAME,
                "choices": [{
                    "delta": {"content": f"Error: {str(e)}"},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": Config.API_VERSION,
        "model": Config.MODEL_NAME
    }


@app.get("/api/agents")
async def list_agents():
    """List available agents."""
    return {
        "agents": Config.AVAILABLE_AGENTS
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {Config.API_TITLE} v{Config.API_VERSION}")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Host: {Config.API_HOST}:{Config.API_PORT}")
    
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level=Config.LOG_LEVEL.lower()
    )