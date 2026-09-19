from fastapi import APIRouter
from typing import List
from schemas.medico_schema import MedicoResponse
from services import medico_service

router = APIRouter(prefix="/medicos", tags=["Médicos"])

@router.get("", response_model=List[MedicoResponse])
def listar_medicos_endpoint():
    return medico_service.obtener_medicos()
