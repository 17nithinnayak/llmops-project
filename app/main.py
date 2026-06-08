import httpx
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings
from metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, TOKENS_GENERATED,
    LatencyTimer, metrics_endpoint, record_model_info
)

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

SYSTEM_PROMPT = """You are an expert cloud infrastructure and DevOps engineer.
Review the code submitted and respond in this exact structure:

## Summary
One sentence describing what the code does.

## Issues Found
List each issue with severity: [CRITICAL] [WARNING] [INFO]
If no issues, write "No issues found."

## Suggestions
Concrete, actionable improvements.

## Revised Snippet (if applicable)
Only include if you have a meaningful fix to show.

Be concise, technical, and direct. Focus on security, best practices, and cloud-readiness."""

class CodeReviewRequest(BaseModel):
    code: str = Field(..., min_length=5)
    language: Optional[str] = None
    context: Optional[str] = None

class CodeReviewResponse(BaseModel):
    review: str
    model: str
    tokens_generated: int
    latency_seconds: float

class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_reachable: bool
    version: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} with model {settings.model_name}")
    record_model_info(settings.model_name, settings.version)
    yield
    logger.info("Shutting down")

app = FastAPI(
    title=settings.app_name,
    description="LLMOps project — Cloud infra code review powered by Qwen2.5-Coder",
    version=settings.version,
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", tags=["ops"])
async def root():
    return {"service": settings.app_name, "model": settings.model_name, "docs": "/docs"}

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ollama_host}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        model=settings.model_name,
        ollama_reachable=ollama_ok,
        version=settings.version
    )

@app.get("/metrics", tags=["ops"])
async def metrics():
    return metrics_endpoint()

@app.post("/review", response_model=CodeReviewResponse, tags=["inference"])
@limiter.limit(settings.rate_limit)
async def review_code(request: Request, body: CodeReviewRequest):
    lang_hint = f"\nLanguage: {body.language}" if body.language else ""
    ctx_hint = f"\nContext: {body.context}" if body.context else ""
    user_message = f"Review this code:{lang_hint}{ctx_hint}\n\n```\n{body.code}\n```"

    payload = {
        "model": settings.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "stream": False,
        "options": {"num_predict": settings.max_tokens, "temperature": 0.2}
    }

    start = time.time()
    with LatencyTimer(REQUEST_LATENCY, {"endpoint": "/review"}):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.TimeoutException:
            REQUEST_COUNT.labels(endpoint="/review", status="timeout").inc()
            raise HTTPException(status_code=504, detail="Model timed out. Try a shorter snippet.")
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/review", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

    data = response.json()
    tokens = data.get("eval_count", 0)
    TOKENS_GENERATED.observe(tokens)
    REQUEST_COUNT.labels(endpoint="/review", status="success").inc()
    logger.info(f"Review done | tokens={tokens} | latency={round(time.time()-start,3)}s")

    return CodeReviewResponse(
        review=data["message"]["content"],
        model=settings.model_name,
        tokens_generated=tokens,
        latency_seconds=round(time.time() - start, 3)
    )
