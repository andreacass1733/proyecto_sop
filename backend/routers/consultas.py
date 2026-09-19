from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional
from schemas.consulta_schema import ConsultaCreate, ConsultaUpdate, ConsultaResponse
from services import consulta_service

router = APIRouter(prefix="/consultas", tags=["Consultas Médicas"])

@router.post("", response_model=ConsultaResponse, status_code=status.HTTP_201_CREATED)
def crear_consulta_endpoint(consulta: ConsultaCreate):
    try:
        return consulta_service.crear_consulta(consulta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=List[ConsultaResponse])
def listar_consultas(
    paciente_id: Optional[str] = Query(None, description="Filtrar por paciente"),
    medico_id: Optional[str] = Query(None, description="Filtrar por médico")
):
    return consulta_service.obtener_consultas(paciente_id, medico_id)

@router.get("/{consulta_id}", response_model=ConsultaResponse)
def obtener_consulta(consulta_id: str):
    consulta = consulta_service.obtener_consulta_por_id(consulta_id)
    if not consulta:
        raise HTTPException(status_code=404, detail="Consulta no encontrada")
    return consulta

@router.put("/{consulta_id}", response_model=ConsultaResponse)
def actualizar_consulta_endpoint(consulta_id: str, consulta: ConsultaUpdate):
    actualizada = consulta_service.actualizar_consulta(consulta_id, consulta)
    if not actualizada:
        raise HTTPException(status_code=404, detail="Consulta no encontrada para actualizar")
    return actualizada

@router.delete("/{consulta_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_consulta_endpoint(consulta_id: str):
    exito = consulta_service.eliminar_consulta(consulta_id)
    if not exito:
        raise HTTPException(status_code=404, detail="Consulta no encontrada para eliminar")
    return None
