import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*")

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Cookie, Response, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from AiAgent import ThryAgent
from config import validateEnv, get_uuid
from pydantic import BaseModel, field_validator
import logging
import hmac

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryID(BaseModel):
    query: str
    chat_id: str

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 5000:
            raise ValueError('Query too long (max 5000 characters)')
        return v.strip()

    @field_validator('chat_id')
    @classmethod
    def validate_chat_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Chat ID cannot be empty')
        return v.strip()

validateEnv()

app = FastAPI()

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Thry-Api-Key", "Content-Type"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

async def verify_api_key(api_key: str = Header(None, alias="Thry-Api-Key")):
    expected = os.getenv("THRY_API_KEY", "")
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("60/minute", key_func=lambda r: "global")
@limiter.limit("5/minute")
async def send_message(message: QueryID,
                       response: Response,
                       request: Request,
                       session_id: str = Cookie(None),
                       authorized: str = Depends(verify_api_key)):

    try:
        if not session_id:
            session_id = str(get_uuid())
            response.set_cookie(
                key="session_id",
                value=session_id,
                max_age=30 * 24 * 60 * 60,
                httponly=True,
                secure=True,
            )

        thread_id = f"{session_id}:{message.chat_id}"

        # Create a new agent instance per request for Vercel serverless compatibility
        agent = ThryAgent()

        # Run the synchronous agent.run() in a thread pool to avoid blocking
        result = await asyncio.to_thread(agent.run, message.query, thread_id)

        if not result or 'messages' not in result or not result["messages"]:
            logger.error("Agent returned invalid result")
            raise HTTPException(status_code=500, detail="Failed to generate response")

        return {"response": result['messages'][-1].content}

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in /chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later."
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Vercel serverless handler
handler = app
