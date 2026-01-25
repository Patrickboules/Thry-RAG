from langgraph.graph import StateGraph,START,END
from operator import add as add_messages
from typing import TypedDict,Sequence,Annotated

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,ToolMessage,AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_postgres.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq


from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from dotenv import load_dotenv
import os

load_dotenv()

# intialize Google LLM 
llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0)

#intialize Gemini Embeddings Model 
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#intialize Postgres Chat DB
DBconnection_string = os.getenv("DATABASE_URL")
connection_kwargs = {
    "autocommit": True,      
    "prepare_threshold": 0,
    "row_factory": dict_row,
}

# Initialize your pool with the kwargs
sync_connection_pool = ConnectionPool(
    conninfo=DBconnection_string,
    max_size=20,
    kwargs=connection_kwargs
)
checkpointer = PostgresSaver(sync_connection_pool)

#intialize VectorSpace
vector_space = PGVector(
        connection=DBconnection_string,
        embeddings=embeddings,
        collection_name='thry_rag'
    )

retriever = vector_space.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

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

class AgentState(TypedDict):
    session_id:str
    messages:Sequence[Annotated[BaseMessage,add_messages]]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages']) 

    message = llm.invoke(messages)

    return {'messages': [message]}

tools_dict = {our_tool.name: our_tool for our_tool in tools} 

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}    

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()
