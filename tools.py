from langchain_core.tools import tool
from langchain_postgres.vectorstores import PGVector
import httpx
import os

RERANK_URL = "https://api.langsearch.com/v1/rerank"


class MyTools:

    def __init__(self, vector_space: PGVector):
        self._vector_space = vector_space

    def get_tools(self):
        """Create tool with vector_space bound via closure."""

        @tool
        async def retriever_with_reranker(query: str) -> str:
            """
            This tool searches and returns the information from the Thndr Learn Content.
            """
            retriever = self._vector_space.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )

            docs = await retriever.ainvoke(query)

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

            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                try:
                    response = await client.post(RERANK_URL, headers=headers, json=payload)
                    response.raise_for_status()
                except httpx.TimeoutException:
                    # Fallback: return top-3 docs without reranking
                    return "\n\n".join([
                        f"Document {i+1}:\n{doc.page_content}"
                        for i, doc in enumerate(docs[:3])
                    ])
                except httpx.HTTPStatusError as e:
                    return f"Reranking service error ({e.response.status_code})."

            try:
                results = response.json()["results"]
            except (KeyError, ValueError):
                # Fallback if response parsing fails
                return "\n\n".join([
                    f"Document {i+1}:\n{doc.page_content}"
                    for i, doc in enumerate(docs[:3])
                ])

            return "\n\n".join([
                f"Document {i+1}:\n{result['document']['text']}"
                for i, result in enumerate(results)
            ])

        return [retriever_with_reranker]


def get_tools(vectordb: PGVector):
    """Get a new MyTools instance."""
    return MyTools(vectordb).get_tools()