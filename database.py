import atexit
import os
import warnings

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
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
    def __init__(self)-> Database:
        #initialize database string
        self.__dbconnection_string = os.getenv("DATABASE_URL")
        if not self.__dbconnection_string:
            raise ValueError("DATABASE_URL environment variable is not set")
        #intialize connection kawrgs
        self.__connection_kwargs = {
                            "autocommit": True,      
                            "prepare_threshold": 0,
                            "row_factory": dict_row,
                            }
        
        #initialize connection pool to database URL
        self.__sync_connection_pool = ConnectionPool(
                    conninfo=self.__dbconnection_string,
                    max_size=20,
                    min_size=5,
                    timeout=30,
                    kwargs=self.__connection_kwargs
                )
        #initialize Embedding Model
        self.__embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",provider="hf-inference")
        
        #initialize Reranker
        self.__reranker = InferenceClient(model="BAAI/bge-reranker-v2-m3")

        #initialize PGVector
        self.__vector_db = PGVector(
                    connection=self.__dbconnection_string,
                    embeddings=self.__embeddings,
                    collection_name='thry_rag'
                )
        
        self.__checkpointer = PostgresSaver(self.__sync_connection_pool)
        self.__checkpointer.setup()

    def get_pgvector(self)-> PGVector:
        return self.__vector_db
    
    def get_embeddings(self) -> HuggingFaceEndpointEmbeddings:
        return self.__embeddings
    
    def get_reranker(self)-> InferenceClient:
        return self.__reranker
    
    def get_PostgresSaver(self)-> PostgresSaver:
        return self.__checkpointer

    def close(self):
        """Close the connection pool to prevent shutdown warnings."""
        if self.__sync_connection_pool:
            self.__sync_connection_pool.close()


database = Database()

# Register cleanup to avoid PythonFinalizationError on exit
atexit.register(database.close)

def get_database()->Database:
    return database