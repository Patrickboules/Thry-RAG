from langchain_core.tools import tool
from functools import partial
from langchain_postgres.vectorstores import PGVector
import httpx
import os

url = "https://api.langsearch.com/v1/rerank"

def _retriever_with_reranker(query: str,vector_space:PGVector) -> str:
    """
    This tool searches and returns the information from the Thndr Learn Content.
    """
    retriever = vector_space.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
    docs = retriever.invoke(query)

    if not docs:
            return "I found no relevant information in the Thndr Learn Content"

    payload = {
        "model": "langsearch-reranker-v1",
        "query": query,
        "top_n": 3,
        "return_documents": True,
        "documents": [doc.page_content for doc in docs]
    }

    headers = {
        'Authorization': f'Bearer {os.getenv("LANGSEARCH_API_KEY")}',
        'Content-Type': 'application/json'
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()    

    results = response.json()["results"]

    return "\n\n".join([
    f"Document {i+1}:\n{result['document']['text']}"
    for i, result in enumerate(results)
    ])


class MyTools:
    """
    Serverless-compatible tool manager.

    Each instance creates its own InferenceClient to avoid shared state
    across function invocations in Vercel serverless.
    """

    def __init__(self,vector_space:PGVector):
        self.__vectordb = vector_space

    def get_tools(self):
        # Create tool with reranker bound using functools.partial
        return [tool(partial(_retriever_with_reranker,vector_space = self.__vectordb))]


# Helper function to get tools instance
def get_tools(vectordb:PGVector):
    """Get a new MyTools instance for serverless compatibility."""
    return MyTools(vectordb).get_tools()