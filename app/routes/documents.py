from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.document import DocumentResponse
from app.services.document_service import (
    get_all_documents,
    get_document,
    delete_document,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


@router.get(
    "",
    response_model=list[DocumentResponse]
)
def list_documents(
    db: Session = Depends(get_db)
):
    return get_all_documents(db)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse
)
def get_single_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    document = get_document(db, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return document


@router.delete(
    "/{document_id}",
    response_model=DocumentResponse
)
def remove_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    document = delete_document(db, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return document