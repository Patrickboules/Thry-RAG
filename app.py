import warnings
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*")

from fastapi import FastAPI,Cookie,Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from AiAgent import ThryAgent
from pydantic import BaseModel
import uuid

class QueryID(BaseModel):
    query:str
    chat_id:str


limiter = Limiter(
    key_func=get_remote_address, 
    default_limits=["60/minute"]
)

app = FastAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

agent = ThryAgent()

@app.post("/chat")
async def send_message(message:QueryID ,response:Response ,session_id:str = Cookie(None)):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key= "session_id",
            value= session_id,
            max_age= 30 * 60 * 60 * 24,
            httponly= True,
            secure= True,
            samesite= "lax"
        )

    thread_id = f"{session_id}:{message.chat_id}"
    result = agent.run(message.query,thread_id)
    return {"response":result['messages'][-1].content}