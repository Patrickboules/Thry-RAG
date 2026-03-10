from pydantic import BaseModel, field_validator
import re

class QueryID(BaseModel):
    query: str
    chat_id: str

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        if len(v) > 5000:
            raise ValueError('Query too long (max 5000 characters)')
        return v.strip()

    @field_validator('chat_id')
    @classmethod
    def validate_chat_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Chat ID cannot be empty')
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', v):  # only safe characters
            raise ValueError('Chat ID contains invalid characters')
        return v
