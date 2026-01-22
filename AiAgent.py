from typing import TypedDict,Sequence,Annotated
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,ToolMessage,AIMessage
from operator import add as add_messages

from langchain_core.tools import tool
from langchain_postgres import PostgresChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import SupabaseVectorStore

import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

# intialize Google LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    convert_system_message_to_human=True,
)

#intialize Google Gemini Embeddings Model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


#intialize Postgres Chat DB
connection_string = os.getenv("DATABASE_URL")
sync_connection = psycopg.connect(connection_string)

#intialize Semantic Chunking
chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")


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
    session_id:str
    messages:Sequence[Annotated[BaseMessage,add_messages]]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant specialized in investment education, trained on the Thndr Learn educational materials.

Your role:
- Help users learn about investing concepts, strategies, and best practices from the Thndr Learn curriculum
- Explain investment terminology, market principles, and financial concepts in a clear, educational manner
- Guide beginners through their investment learning journey using the knowledge from Thndr's educational resources
- Make multiple retrieval calls when needed to provide comprehensive investment education

Guidelines:
- Always cite specific sections from the Thndr Learn materials when explaining concepts
- Break down complex investment topics into easy-to-understand explanations
- Provide practical examples and scenarios when teaching investment principles
- If a concept is not covered in the Thndr Learn materials, clearly state this
- Encourage sound investment practices and learning-first approach
- When explaining strategies, reference the specific lessons or chapters from the curriculum

Your goal is to be a knowledgeable investment education assistant, helping users understand investing through the Thndr Learn framework. Focus on education, clarity, and building foundational investment knowledge.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} 


def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages']) 

    message = llm.invoke(messages)
    history = PostgresChatMessageHistory(
        table_name='UserMessages',
        session_id=state['session_id'],
        sync_connection=sync_connection
    )

    history.add_message(message)
    return {'messages': [message]}

