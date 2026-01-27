from langchain_core.tools import tool
from database import get_database

vector_space = get_database().get_pgvector()

retriever = vector_space.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 7}
)

@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Thndr Learn Content.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Thndr Learn Content"
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

def get_tools():
    return [retriever_tool]
