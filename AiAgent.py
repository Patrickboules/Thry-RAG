from typing import TypedDict,Sequence,Annotated
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,ToolMessage,AIMessage
from operator import add as add_messages
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from config import RedisManager,QdrantManager

Redis = RedisManager().get_client()
Qdrant = QdrantManager().get_client()
load_dotenv()

tools = []

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True,
).bind_tools(tools)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


class AgentState(TypedDict):
    messages:Sequence[Annotated[BaseMessage,add_messages]]