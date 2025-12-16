from typing import TypedDict,Sequence,Annotated
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,ToolMessage,AIMessage
from operator import add as add_messages
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_postgres import PostgresChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
# from config import get_Qdrant_Client
import os

load_dotenv()

# intialize Google LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True,
)

# intialize PostgresDatabase
history = PostgresChatMessageHistory(
    connection_string=os.getenv("DATABASE_URL"),
    session_id="user_123"
)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="thry_rag",
    url=os.getenv("QDRANT_URL"),
)

retriever = qdrant.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} 
)

@tool
def retriever_tool(query:str) -> str:
    """This function retrieves from the Thndr-Learn"""
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Thndr Learn documents."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages:Sequence[Annotated[BaseMessage,add_messages]]

def should_continue():
    pass