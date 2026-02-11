import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*")

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Cookie, Response, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from AiAgent import ThryAgent
from pydantic import BaseModel, field_validator
import uuid
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
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

async def verify_api_key(api_key: str = Header(None)):
    expected = os.getenv("THRY_API_KEY", "")
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

# Initialize agent
agent = ThryAgent()
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
            session_id = str(uuid.uuid4())
            response.set_cookie(
                key="session_id",
                value=session_id,
                max_age=30 * 24 * 60 * 60,
                httponly=True,
                secure=os.getenv("ENVIRONMENT", "development") == "production",
                samesite="lax"
            )

        thread_id = f"{session_id}:{message.chat_id}"
        result = agent.run(message.query, thread_id)

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
