from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    document_id: str
    question: str


class SourceItem(BaseModel):
    page: int
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
