import logging
import os
import warnings
from typing import Optional

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.checkpoint.postgres import PostgresSaver  # sync saver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
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
        self.__sync_connection_pool = ConnectionPool(
            conninfo=self.__dbconnection_string,
            max_size=10,
            min_size=1,            # keep at least 1 connection warm on Render
            reconnect_timeout=30,
            reconnect_failed=None,
            timeout=10,
            kwargs=self.__connection_kwargs,
            open=True,             # open immediately, fail fast at startup
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
        self.__checkpointer: Optional[PostgresSaver] = PostgresSaver(self.__sync_connection_pool)

        # Create checkpoint tables if they don't exist yet.
        self.__checkpointer.setup()

    def get_pgvector(self) -> Optional[PGVector]:
        return self.__vector_db

    def get_embeddings(self) -> Optional[HuggingFaceEndpointEmbeddings]:
        return self.__embeddings

    def get_PostgresSaver(self) -> Optional[PostgresSaver]:
        return self.__checkpointer

    def get_pool(self) -> Optional[ConnectionPool]:
        return self.__sync_connection_pool

    def close(self):
        try:
            if hasattr(self, '_Database__sync_connection_pool') and self.__sync_connection_pool:
                if not self.__sync_connection_pool.closed:
                    self.__sync_connection_pool.close()
        except Exception:
            logger.warning("Error closing psycopg pool", exc_info=True)

        try:
            if hasattr(self, '_Database__sa_engine') and self.__sa_engine:
                self.__sa_engine.dispose()
        except Exception:
            logger.warning("Error disposing SQLAlchemy engine", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def get_database() -> Database:
    return Database()