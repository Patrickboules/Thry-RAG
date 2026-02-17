from langchain_core.tools import tool
from typing import Optional
from functools import partial
from huggingface_hub import InferenceClient
from database import use_db


def _retriever_with_reranker(query: str, reranker: InferenceClient) -> str:
    """
    This tool searches and returns the information from the Thndr Learn Content.
    """
    with use_db() as database:
        vector_space = database.get_pgvector()

        retriever = vector_space.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        docs = retriever.invoke(query)

        if not docs:
            return "I found no relevant information in the Thndr Learn Content"

        scores = []
        for doc in docs:
            result = reranker.text_classification(f"{query} [SEP] {doc.page_content}")
            scores.append(result[0]['score'])

        # Sort by score and get top 3
        ranked_docs = sorted(zip(scores, docs), reverse=True)[:3]

        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, (score, doc) in enumerate(ranked_docs)])


class MyTools:
    """
    Serverless-compatible tool manager.

    Each instance creates its own InferenceClient to avoid shared state
    across function invocations in Vercel serverless.
    """

    def __init__(self):
        # Create reranker instance per tools instance
        self.__reranker = InferenceClient(model="BAAI/bge-reranker-v2-m3")

    def get_tools(self):
        # Create tool with reranker bound using functools.partial
        return [tool(partial(_retriever_with_reranker, reranker=self.__reranker))]


# Helper function to get tools instance
def get_tools():
    """Get a new MyTools instance for serverless compatibility."""
    return MyTools().get_tools()