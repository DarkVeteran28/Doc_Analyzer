from sqlalchemy.orm import Session

from app.models.document import Document


def get_all_documents(db: Session):
    return db.query(Document).all()


def get_document(db: Session, document_id: str):
    return db.query(Document).filter(
        Document.id == document_id
    ).first()


def delete_document(db: Session, document_id: str):
    document = get_document(db, document_id)

    if document is None:
        return None

    db.delete(document)
    db.commit()

    return document