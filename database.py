from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.checkpoint.postgres import PostgresSaver

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from dotenv import load_dotenv
import os

load_dotenv()

class Database:
    def __init__(self)-> Database:
        #initialize database string
        self.__dbconnection_string = os.getenv("DATABASE_URL")

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
                    kwargs=self.__connection_kwargs
                )
        #initialize Embedding Model
        self.__embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-m3",provider="hf-inference")
        
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
    

database = Database()

def get_database()->Database:
    return database