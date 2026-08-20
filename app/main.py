from fastapi import FastAPI
from app.routes.documents import router as documents_router
from app.database.database import Base, engine
from app.models.document import Document


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Document Q&A API",
    version="1.0.0"
)

app.include_router(documents_router)

@app.get("/")
def root():
    return {
        "message": "AI Document Q&A API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }