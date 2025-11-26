from dotenv import load_dotenv
from functools import lru_cache

from redis.asyncio import Redis
import redis.asyncio as aioredis

from qdrant_client import QdrantClient
from pydantic_settings import BaseSettings
import os

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "THRY RAG"

    REDIS_URL: str
    
    QDRANT_URL: str
    QDRANT_API_KEY: str
    

    GOOGLE_API_KEY: str

    SESSION_EXPIRE_SECONDS: int = 900  # 15 mins
    CHAT_HISTORY_MAX_MESSAGES: int = 50
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

class RedisManager:
    """Redis connection manager"""
    def __init__(self):
        self._client: Redis | None = None

    async def get_client(self) -> Redis:
        """Get or create Redis client"""
        if self._client is None:
            settings = get_settings()

            self._client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
                )
            
            return self._client
    
    async def close(self):
        """Closes Redis Connection"""

        if self._client:
            await self._client.close()
            self._client = None


class QdrantManager:
    """Qdrant connection Manager"""

    def __init__(self):
        self._client: QdrantClient | None = None
    
    def get_client(self) -> QdrantClient:
        """Initialize Qdrant Connection"""
        if self._client is None:
            settings = get_settings()
            Qdrant_Client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            )

    def close(self):
        """Closes Qdrant Connection"""

        if self._client:
            self._client.close()
            self._client = None

redis_manager = RedisManager()
qdrant_manager = QdrantManager()

async def get_redis_manager() -> RedisManager:
    """Returns the Redis client from the instance"""
    return redis_manager.get_client()

def get_Qdrant_manager()-> QdrantManager:
    """Returns the Redis client from the instance"""
    return qdrant_manager.get_client()




