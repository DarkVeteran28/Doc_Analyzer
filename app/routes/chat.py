from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from rag.rag_pipeline import ask_question


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = ask_question(
            document_id=request.document_id,
            question=request.question,
            n_results=3,
            persist_directory="chroma_db",
            retrieval_mode="hybrid",
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )
