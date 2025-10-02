from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "NIM Proxy is running"}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    
    headers = {
        "Authorization": f"Bearer {os.getenv('NIM_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            json=body,
            headers=headers
        )
        return JSONResponse(content=resp.json())

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "meta/llama-3.1-405b-instruct"}]}        ) as response:
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
