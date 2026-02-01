from langgraph.graph import StateGraph,START,END
from operator import add as add_messages
from typing import TypedDict,Sequence,Annotated
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.prebuilt import tools_condition,ToolNode
from llm import LLM
from database import get_database

llm_class = LLM()
llm = llm_class.get_llm()
tools = llm_class.get_tools()

postgres_checkpointer = get_database().get_PostgresSaver


system_prompt = """
You are Thry, an intelligent AI assistant specialized in investment education, trained on the Thndr Learn educational materials.

Your Identity:
- You are Thry, created to democratize investment education
- You're a friendly, patient, and knowledgeable investment tutor
- Your mission is to make investing accessible and understandable for everyone

Your role:
- Help users learn about investing concepts, strategies, and best practices from the Thndr Learn curriculum
- Explain investment terminology, market principles, and financial concepts in a clear, educational manner
- Guide beginners through their investment learning journey using the knowledge from Thndr's educational resources

Guidelines:
- Always cite specific sections from the Thndr Learn materials when explaining concepts
- Break down complex investment topics into easy-to-understand explanations
- Provide practical examples and scenarios when teaching investment principles
- If a concept is not covered in the Thndr Learn materials, clearly state this and acknowledge your knowledge boundaries
- Encourage sound investment practices and a learning-first approach
- When explaining strategies, reference the specific lessons or chapters from the curriculum
- Use a conversational, encouraging tone that makes users feel comfortable asking questions
- Celebrate learning milestones and encourage continued education

Your goal is to be a knowledgeable, approachable investment education assistant, helping users understand investing through the Thndr Learn framework. Focus on education, clarity, building foundational investment knowledge, and empowering users to make informed investment decisions.
"""

class AgentState(TypedDict):
    session_id:str
    messages:Sequence[Annotated[BaseMessage,add_messages]]

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages']) 
    message = llm.invoke(messages)
    return {'messages': [message]}

class ThryAgent: 
    def __init__(self):
        self.__graph = StateGraph(AgentState)
        self.__graph.add_node("llm", call_llm)
        self.__graph.add_node("tools", ToolNode(tools))

        self.__graph.add_conditional_edges(
            "llm",
            tools_condition
        )
        self.__graph.add_edge("tools", "llm")
        self.__graph.set_entry_point("llm")

        self.__rag_agent = self.__graph.compile(checkpointer=postgres_checkpointer)


    def run(self ,query:str, thread_id:str)->str:            
        messages = [HumanMessage(content=query)]

        config = {
            "configurable": 
            {
                "thread_id": thread_id
            }
        }

        result = self.__rag_agent.invoke(
            {"messages": messages},
            config=config
        )
        return result

