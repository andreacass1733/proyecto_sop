from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional
from schemas.cita_schema import CitaCreate, CitaUpdate, CitaResponse
from services import cita_service

router = APIRouter(prefix="/citas", tags=["Citas Médicas"])

@router.post("", response_model=CitaResponse, status_code=status.HTTP_201_CREATED)
def crear_cita_endpoint(cita: CitaCreate):
    try:
        return cita_service.crear_cita(cita)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=List[CitaResponse])
def listar_citas(
    paciente_id: Optional[str] = Query(None, description="Filtrar por paciente"),
    medico_id: Optional[str] = Query(None, description="Filtrar por médico")
):
    return cita_service.obtener_citas(paciente_id, medico_id)

@router.get("/{cita_id}", response_model=CitaResponse)
def obtener_cita(cita_id: str):
    cita = cita_service.obtener_cita_por_id(cita_id)
    if not cita:
        raise HTTPException(status_code=404, detail="Cita no encontrada")
    return cita

@router.put("/{cita_id}", response_model=CitaResponse)
def actualizar_cita_endpoint(cita_id: str, cita: CitaUpdate):
    actualizada = cita_service.actualizar_cita(cita_id, cita)
    if not actualizada:
        raise HTTPException(status_code=404, detail="Cita no encontrada para actualizar")
    return actualizada

@router.delete("/{cita_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_cita_endpoint(cita_id: str):
    exito = cita_service.eliminar_cita(cita_id)
    if not exito:
        raise HTTPException(status_code=404, detail="Cita no encontrada para eliminar")
    return None
