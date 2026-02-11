from langchain_core.tools import tool
from database import use_db



@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Thndr Learn Content.
    """
    with use_db() as database:
        vector_space = database.get_pgvector()
        reranker_model = database.get_reranker()

        retriever = vector_space.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
            )    
        docs = retriever.invoke(query)

        if not docs:
            return "I found no relevant information in the Thndr Learn Content"

        scores = []
        for doc in docs:
            result = reranker_model.text_classification(f"{query} [SEP] {doc.page_content}")
            scores.append(result[0]['score'])

        # Sort by score and get top 3
        ranked_docs = sorted(zip(scores, docs), reverse=True)[:3]

        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, (score, doc) in enumerate(ranked_docs)])

class MyTools:
    def __init__(self):
        self.__retriever_tool = retriever_tool

    def get_tools(self):
        return [self.__retriever_tool]


my_tools = MyTools()

