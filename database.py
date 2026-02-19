import os
import warnings
from contextlib import contextmanager
from typing import Optional, Generator

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# Suppress transformers/pytorch warnings before importing
warnings.filterwarnings("ignore", message=".*PyTorch.*")
warnings.filterwarnings("ignore", message=".*TensorFlow.*")

load_dotenv()


class Database:
    """
    Serverless-compatible database manager for Neon Postgres.

    Designed for Vercel serverless functions with:
    - Short-lived connection pools (created per request in serverless)
    - Automatic cleanup via context manager (no atexit - doesn't work in serverless)
    - Lazy initialization to reduce cold start latency
    - Singleton pattern disabled for serverless cold starts
    - Neon pooled connection string support (use the -pooler endpoint)
    """

    def __init__(self):
        # Initialize database connection string
        self.__dbconnection_string = os.getenv("DATABASE_URL")
        if not self.__dbconnection_string:
            raise ValueError("DATABASE_URL environment variable is not set")

        self.__connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,  # Required for PgBouncer transaction mode
            "row_factory": dict_row,
        }

        # Initialize connection pool with serverless-optimized settings
        self.__sync_connection_pool = ConnectionPool(
            conninfo=self.__dbconnection_string,
            max_size=1,
            min_size=0,  
            timeout=15,  
            max_lifetime=120,  
            max_idle=60,  
            kwargs=self.__connection_kwargs,
            open=False
        )

        # Lazy initialization placeholders (initialized on first access)
        self.__embeddings: Optional[HuggingFaceEndpointEmbeddings] = None
        self.__vector_db: Optional[PGVector] = None
        self.__checkpointer: Optional[PostgresSaver] = None

    # Lazy property for Embeddings - reduces cold start latency
    @property
    def __embeddings_lazy(self) -> HuggingFaceEndpointEmbeddings:
        if self.__embeddings is None:
            self.__embeddings = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
            )
        return self.__embeddings

    # Lazy property for PGVector - reduces cold start latency
    @property
    def __vector_db_lazy(self) -> PGVector:
        if self.__vector_db is None:
            self.__vector_db = PGVector(
                connection=self.__dbconnection_string,
                embeddings=self.__embeddings_lazy,
                collection_name='thry_rag'
            )
        return self.__vector_db

    # Lazy property for PostgresSaver - reduces cold start latency
    @property
    def __checkpointer_lazy(self) -> PostgresSaver:
        if self.__checkpointer is None:
            self.__checkpointer = PostgresSaver(self.__sync_connection_pool)
        return self.__checkpointer

    def get_pgvector(self) -> PGVector:
        return self.__vector_db_lazy

    def get_embeddings(self) -> HuggingFaceEndpointEmbeddings:
        return self.__embeddings_lazy

    def get_PostgresSaver(self) -> PostgresSaver:
        return self.__checkpointer_lazy
    
    def get_pool(self) -> ConnectionPool:
        return self.__sync_connection_pool

    def close(self):
        if hasattr(self, '_Database__sync_connection_pool') and self.__sync_connection_pool:
            try:    
                self.__sync_connection_pool.close()
            except Exception as e:
                pass
    # Context manager support for guaranteed cleanup
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Don't suppress exceptions

def get_database() -> Database:
    """
    Get database instance.

    For Vercel serverless: Create a new instance per request to avoid
    connection leaks across function invocations.
    """
    return Database()