from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from database import get_database
from dotenv import load_dotenv

load_dotenv()
#initialize VectorDatabase
vector_space = get_database().get_pgvector()

#intialize Semantic Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

pdf_path = "./Thndr-Learn.pdf"

pdf_loader = PyPDFLoader(pdf_path)

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
except Exception as e:
    raise

pages_split = text_splitter.split_documents(pages)

try:
    vector_space.add_documents(pages_split)
except Exception as e:
    print(f"Error setting up PgVectorDB: {str(e)}")
    raise