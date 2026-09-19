from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ConsultaBase(BaseModel):
    cita_id: Optional[str] = None
    paciente_id: Optional[str] = None
    medico_id: Optional[str] = None
    motivo: Optional[str] = None
    observaciones: Optional[str] = None
    estado: Optional[str] = "en_proceso"
    reentrenamiento_id: Optional[str] = None
    fecha_proxima_cita: Optional[str] = None

class ConsultaCreate(ConsultaBase):
    pass

class ConsultaUpdate(BaseModel):
    cita_id: Optional[str] = None
    paciente_id: Optional[str] = None
    medico_id: Optional[str] = None
    motivo: Optional[str] = None
    observaciones: Optional[str] = None
    estado: Optional[str] = None
    reentrenamiento_id: Optional[str] = None
    fecha_proxima_cita: Optional[str] = None

class ConsultaResponse(ConsultaBase):
    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
