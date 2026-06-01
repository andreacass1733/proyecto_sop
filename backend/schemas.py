from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Criterio3Response(BaseModel):
    id: int
    imagen_path: str
    prob_sop: float
    resultado: str
    validado: bool
    etiqueta_real: Optional[str]
    fecha: datetime

    class Config:
        from_attributes = True

class ValidarRequest(BaseModel):
    etiqueta_real: str  # "Normal" o "SOP"