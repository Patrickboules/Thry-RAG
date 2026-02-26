import logging
import os
import warnings
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

warnings.filterwarnings("ignore", message=".*PyTorch.*")
warnings.filterwarnings("ignore", message=".*TensorFlow.*")

load_dotenv()

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.__dbconnection_string = os.getenv("DATABASE_URL")
        if not self.__dbconnection_string:
            raise ValueError("DATABASE_URL environment variable is not set")

        self.__connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }

        sa_url = (
            self.__dbconnection_string
            .replace("postgresql://", "postgresql+psycopg2://")
            .replace("postgresql+psycopg://", "postgresql+psycopg2://")
        )
        self.__sa_engine = create_engine(sa_url, poolclass=NullPool)

        # Open the pool explicitly so it's ready before first request.
        self.__sync_connection_pool = AsyncConnectionPool(
            conninfo=self.__dbconnection_string,
            max_size=10,
            min_size=1,            # keep at least 1 connection warm on Render
            reconnect_timeout=30,
            reconnect_failed=None,
            timeout=10,
            kwargs=self.__connection_kwargs,
            open=False,             
        )

        self.__embeddings: Optional[HuggingFaceEndpointEmbeddings] = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.__vector_db: Optional[PGVector] = PGVector(
            connection=self.__sa_engine,
            embeddings=self.__embeddings,
            collection_name='thry_rag'
        )

        # Use the SYNC PostgresSaver — our agent runs in asyncio.to_thread
        # (a plain thread with no event loop), so async savers won't work.
        self.__checkpointer: Optional[AsyncPostgresSaver] = AsyncPostgresSaver(self.__sync_connection_pool)


    def get_pgvector(self) -> Optional[PGVector]:
        return self.__vector_db

    def get_embeddings(self) -> Optional[HuggingFaceEndpointEmbeddings]:
        return self.__embeddings

    def get_PostgresSaver(self) -> Optional[AsyncPostgresSaver]:
        return self.__checkpointer

    def get_pool(self) -> Optional[AsyncConnectionPool]:
        return self.__sync_connection_pool

    async def close(self):
        try:
            if hasattr(self, '_Database__sync_connection_pool') and self.__sync_connection_pool:
                if not self.__sync_connection_pool.closed:
                    await self.__sync_connection_pool.close()
        except Exception:
            logger.warning("Error closing psycopg pool", exc_info=True)

        try:
            if hasattr(self, '_Database__sa_engine') and self.__sa_engine:
                self.__sa_engine.dispose()
        except Exception:
            logger.warning("Error disposing SQLAlchemy engine", exc_info=True)

    def __enter__(self):
        return self

    async def __exit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False


def get_database() -> Database:
    return Database()