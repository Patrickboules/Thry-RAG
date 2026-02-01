import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*")

from fastapi import FastAPI,Cookie,Response
from AiAgent import ThryAgent
from pydantic import BaseModel
import uuid

class QueryID(BaseModel):
    query:str
    chatID:str

app = FastAPI()

app.post("/chat")
async def send_message(message:QueryID,response:Response,session_id:str = Cookie(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session id",
            value=session_id,
            max_age=30*60*60*24,
            httponly=True,
            secure=True,
            samesite="lax"
        )

    thread_id = f"{session_id}:{message.chatID}"
    agent = ThryAgent()
    result = agent.run(message.query,thread_id)
    return {"response":result['messages'][-1].content}