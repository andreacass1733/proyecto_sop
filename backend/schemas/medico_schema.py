from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class MedicoBase(BaseModel):
    nombre: str
    primer_apellido: str
    segundo_apellido: Optional[str] = None
    especialidad: Optional[str] = None
    telefono: Optional[str] = None

class MedicoResponse(MedicoBase):
    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
