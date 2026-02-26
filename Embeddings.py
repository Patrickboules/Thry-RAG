import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from database import get_database
from dotenv import load_dotenv


load_dotenv()

# Initialize Semantic Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

async def process_pdf(pdf_path: str) -> None:
    """
    Process a PDF file and add its embeddings to the vector database.

    Args:
        pdf_path: Path to the PDF file to process
    """
    # Use context manager to ensure database cleanup
    async with get_database() as db:
        vector_space = db.get_pgvector()

        pdf_loader = PyPDFLoader(pdf_path)

        # Load and split the PDF
        try:
            pages = pdf_loader.load()
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {str(e)}")
            raise

        pages_split = text_splitter.split_documents(pages)

        try:
            vector_space.add_documents(pages_split)
            print(f"Successfully processed {len(pages_split)} chunks from {pdf_path}")
        except Exception as e:
            print(f"Error adding documents to PgVectorDB: {str(e)}")
            raise


def main():
    """Main entry point for processing the default PDF."""
    pdf_path = "./Thndr-Learn.pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    process_pdf(pdf_path)


if __name__ == "__main__":
    main()
