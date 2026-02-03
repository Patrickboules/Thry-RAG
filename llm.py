from langchain_groq import ChatGroq
from langchain_core.tools import BaseTool
from tools import my_tools
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self) -> None: 
        # Initialize the endpoint
        llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.7,
            max_tokens=2000,
            max_retries=2,
        )        
        # Fetch and bind tools
        self.__tools = self._fetch_tools() 
        self.__llm = llm.bind_tools(self.__tools)

    def _fetch_tools(self) -> list[BaseTool]:
        return my_tools.get_tools()

    def get_llm(self)-> ChatGroq:
        return self.__llm
    
    def get_tools(self) -> list[BaseTool]:
        return self.__tools