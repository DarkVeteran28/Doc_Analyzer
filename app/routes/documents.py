import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.document import DocumentResponse
from app.services.document_service import (
    create_document,
    get_all_documents,
    get_document,
    delete_document,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"]
)


UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post(
    "/upload",
    response_model=DocumentResponse
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # 2. Generate unique document ID
    document_id = str(uuid.uuid4())

    # 3. Create file path
    file_path = os.path.join(
        UPLOAD_DIR,
        f"{document_id}.pdf"
    )

    # 4. Save PDF
    try:
        contents = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to save the uploaded file"
        )

    # 5. Save metadata in database
    document = create_document(
        db=db,
        document_id=document_id,
        filename=file.filename,
        file_path=file_path,
        status="uploaded"
    )

    return document


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