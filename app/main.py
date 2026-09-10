import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.documents import router as documents_router
from app.routes.chat import router as chat_router
from app.database.database import Base, engine
from app.models.document import Document


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Document Q&A API",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# CORS
# Allow the deployed Vercel frontend and localhost dev server.
# Set FRONTEND_URL on Render to https://<your-app>.vercel.app
# ---------------------------------------------------------------------------
_frontend_url = os.environ.get("FRONTEND_URL", "").strip()

_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if _frontend_url:
    _allowed_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)


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
