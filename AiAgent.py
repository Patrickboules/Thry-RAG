from langgraph.graph import StateGraph
from langgraph.errors import GraphRecursionError
from operator import add as add_messages
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from llm import LLM
from database import get_database
from functools import partial


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
    messages: Sequence[Annotated[BaseMessage, add_messages]]


def call_llm(state: AgentState, llm) -> AgentState:
    """Call the LLM with the current state."""
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm.invoke(messages)
    return {'messages': [message]}


class ThryAgent:
    def __init__(self):
        self.__db_manager = get_database()

        self.__llm_class = LLM(self.__db_manager.get_pgvector())
        self.__llm = self.__llm_class.get_llm()
        self.__tools = self.__llm_class.get_tools()

        graph = StateGraph(AgentState)

        graph.add_node("llm", partial(call_llm, llm=self.__llm))
        graph.add_node("tools", ToolNode(self.__tools))

        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")
        graph.set_entry_point("llm")

        # This graph is invoked via .invoke() (sync) inside asyncio.to_thread(),
        # which gives it its own thread with no running event loop — correct.
        self.__rag_agent = graph.compile(checkpointer=self.__db_manager.get_PostgresSaver())

    async def initialize(self):
        await self.__db_manager.get_pool().open()

    async def run(self, query: str, thread_id: str) -> dict:
        """
        Run the agent asynchronously.

        """
        try:
            config = {
                "recursion_limit": 25,
                "configurable": {
                    "thread_id": thread_id,
                }
            }
            result = await self.__rag_agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config=config
            )
            return result

        except GraphRecursionError:
            return {
                "messages": [
                    AIMessage(content="I'm sorry, your question required too many steps to process. Could you try rephrasing or simplifying it?")
                ]
            }
        except Exception:
            raise

    async def close(self):
        """Clean up database connections."""
        await self.__db_manager.close()