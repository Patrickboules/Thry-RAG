from langgraph.graph import StateGraph,START,END
from operator import add as add_messages
from typing import TypedDict,Sequence,Annotated

from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,ToolMessage,AIMessage
from langgraph.prebuilt import tools_condition,ToolNode
from llm import LLM

llm_class = LLM()
llm = llm_class.get_llm()
tools = llm_class.get_tools()

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

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages']) 

    message = llm.invoke(messages)

    return {'messages': [message]}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", ToolNode(tools))

graph.add_conditional_edges(
    "llm",
    tools_condition
)
graph.add_edge("tools", "llm")
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