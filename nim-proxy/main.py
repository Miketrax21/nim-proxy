from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import os
from typing import AsyncGenerator

app = FastAPI()

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NIM_API_KEY", "")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    auth_header = request.headers.get("Authorization", "")
    body = await request.json()
    
    model = body.get("model", "meta/llama-3.1-405b-instruct")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 1024)
    top_p = body.get("top_p", 1.0)
    
    nim_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": stream
    }
    
    headers = {
        "Authorization": f"Bearer {NIM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        if stream:
            return StreamingResponse(
                stream_nim_response(client, nim_payload, headers),
                media_type="text/event-stream"
            )
        else:
            try:
                response = await client.post(
                    f"{NIM_BASE_URL}/chat/completions",
                    json=nim_payload,
                    headers=headers
                )
                response.raise_for_status()
                return JSONResponse(content=response.json())
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=str(e))

async def stream_nim_response(client, payload, headers):
    try:
        async with client.stream(
            "POST",
            f"{NIM_BASE_URL}/chat/completions",
            json=payload,
            headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield f"{line}\n\n"
    except httpx.HTTPError as e:
        error_data = {
            "error": {
                "message": str(e),
                "type": "proxy_error",
                "code": "nim_api_error"
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "meta/llama-3.1-405b-instruct",
                "object": "model",
                "created": 1677610602,
                "owned_by": "nvidia"
            }
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
