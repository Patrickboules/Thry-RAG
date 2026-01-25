from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

#intialize Google Gemma Embeddings Model
embeddings = embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#intialize Semantic Chunking
chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

pdf_path = "./Thndr-Learn.pdf"

pdf_loader = PyPDFLoader(pdf_path)

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
except Exception as e:
    raise

pages_split = chunker.split_documents(pages)

try:
    vector_space = PGVector(
        connection=os.getenv("DATABASE_URL"),
        embeddings=embeddings,
        collection_name='thry_rag'
    )

    vector_space.add_documents(pages_split)
except Exception as e:
    print(f"Error setting up Supabase PgVectorDB: {str(e)}")
    raise