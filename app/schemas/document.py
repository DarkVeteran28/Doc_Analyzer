from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_path: str
    upload_date: datetime
    status: str

    class Config:
        from_attributes = True