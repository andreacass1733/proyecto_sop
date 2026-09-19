from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, date

class CitaBase(BaseModel):
    paciente_id: Optional[str] = None
    medico_id: Optional[str] = None
    fecha_atencion: Optional[datetime] = None
    fecha_proxima_cita: Optional[str] = None
    motivo: Optional[str] = None
    observaciones: Optional[str] = None
    estado: Optional[str] = "programada"

class CitaCreate(CitaBase):
    pass

class CitaUpdate(BaseModel):
    paciente_id: Optional[str] = None
    medico_id: Optional[str] = None
    fecha_atencion: Optional[datetime] = None
    fecha_proxima_cita: Optional[str] = None
    motivo: Optional[str] = None
    observaciones: Optional[str] = None
    estado: Optional[str] = None

class CitaResponse(CitaBase):
    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
