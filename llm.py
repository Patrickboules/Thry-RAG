from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.tools import BaseTool
from tools import get_tools
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self) -> None: 
        # Initialize the endpoint
        chat_model = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.0
        )
        
        # Wrap in ChatHuggingFace
        llm = ChatHuggingFace(llm=chat_model)
        
        # Fetch and bind tools
        self.__tools = self._fetch_tools() 
        self.__llm = llm.bind_tools(self.__tools)

    def _fetch_tools(self) -> list[BaseTool]:
        return get_tools()

    def get_llm(self)-> ChatHuggingFace:
        return self.__llm
    
    def get_tools(self) -> list[BaseTool]:
        return self.__tools