from fastapi import Request
from fastapi.responses import JSONResponse
import httpx
import os

async def handler(request: Request):
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
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
        return JSONResponse(
            content=resp.json(),
            headers={"Access-Control-Allow-Origin": "*"}
        )
