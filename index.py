import warnings

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*")

import re
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Cookie, Response, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from upstash_redis import Redis
from upstash_ratelimit import Ratelimit, FixedWindow
from contextlib import asynccontextmanager


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
        v = v.strip()
        if not v:
            raise ValueError('Chat ID cannot be empty')
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', v):  # only safe characters
            raise ValueError('Chat ID contains invalid characters')
        return v

validateEnv()

redis = Redis.from_env()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    yield
    logger.info("Shutting down, closing database connections...")
    agent._ThryAgent__db_manager.close()
    logger.info("Database connections closed.")

app = FastAPI(lifespan=lifespan)
agent = ThryAgent()


ratelimit = Ratelimit(
    redis=redis,
    limiter=FixedWindow(max_requests=5, window=60),  # 5 req/min per IP
)

global_ratelimit = Ratelimit(
    redis=redis,
    limiter=FixedWindow(max_requests=60, window=60),  # 60 req/min globally
)


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

async def check_rate_limit(request: Request):
    ip = (
        request.headers.get("x-vercel-forwarded-for") or
        request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    )
    # Check per-IP limit
    per_ip_response = ratelimit.limit(ip)
    if not per_ip_response.allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": "60"}
        )
    
    # Check global limit
    global_response = global_ratelimit.limit("global")
    if not global_response.allowed:
        raise HTTPException(
            status_code=429,
            detail="Service is busy. Please try again shortly.",
            headers={"Retry-After": "60"}
        )

UUID_REGEX = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
)

def is_valid_uuid(value: str) -> bool:
        return bool(UUID_REGEX.match(value))    

@app.post("/chat")
async def send_message(message: QueryID,
                       response: Response,
                       request: Request,
                       session_id: str = Cookie(None),
                       authorized: str = Depends(verify_api_key),
                       _: None = Depends(check_rate_limit)):
    

    try:
        if not session_id or not is_valid_uuid(session_id):
            session_id = str(get_uuid())
            response.set_cookie(
                key="session_id",
                value=session_id,
                max_age=30 * 24 * 60 * 60,
                httponly=True,
                secure=True,
                samesite="none"
            )

        thread_id = f"{session_id}:{message.chat_id}"


        result = await asyncio.wait_for(
            asyncio.to_thread(agent.run, message.query, thread_id),25
        )

        if not result or 'messages' not in result or not result["messages"]:
            logger.error("Agent returned invalid result")
            raise HTTPException(status_code=500, detail="Failed to generate response")

        return {"response": result['messages'][-1].content}


    except asyncio.TimeoutError:
        # ✅ Catch timeout specifically and return clean 504
        logger.error("Agent timed out after 50 seconds")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try again."
        )
    
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
    """Health check endpoint for Render and monitoring.""" 
    return {"status": "ok", "service": "thry-backend"}

