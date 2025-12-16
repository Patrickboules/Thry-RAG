from dotenv import load_dotenv
from functools import lru_cache


from qdrant_client import QdrantClient
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "THRY RAG"
    
    QDRANT_URL: str
    QDRANT_API_KEY: str
    
    DATABASE_URL: str
    GOOGLE_API_KEY: str
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

class QdrantManager:
    """Qdrant connection Manager"""
    def __init__(self):
        self._client: QdrantClient | None = None
    
    def get_client(self) -> QdrantClient:
        """Initialize Qdrant Connection"""
        if self._client is None:
            settings = get_settings()
            self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            )
        return self._client

    def close(self):
        """Closes Qdrant Connection"""

        if self._client:
            self._client.close()
            self._client = None


qdrant_manager = QdrantManager()

def get_Qdrant_Client()-> QdrantClient:
    """Returns the Qdrant client from the instance"""
    return qdrant_manager.get_client()




