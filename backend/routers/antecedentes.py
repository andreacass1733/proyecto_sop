"""
Router — Antecedentes Médicos de Pacientes.

Rutas disponibles:
    GET    /antecedentes/{paciente_id}   → Obtener antecedentes de un paciente
    POST   /antecedentes                 → Crear antecedentes (primera vez)
    PUT    /antecedentes/{paciente_id}   → Actualizar antecedentes existentes
    POST   /antecedentes/guardar         → Crear o actualizar (upsert simplificado)
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from schemas.paciente_schema import (
    AntecedentesCreate,
    AntecedentesUpdate,
    AntecedentesOut,
)
from services.paciente_service import (
    obtener_antecedentes,
    crear_antecedentes,
    actualizar_antecedentes,
    crear_o_actualizar_antecedentes,
    obtener_paciente,
)

router = APIRouter(prefix="/antecedentes", tags=["Antecedentes Medicos"])


# ── GET /antecedentes/{paciente_id} ───────────────────────────────────────
@router.get("/{paciente_id}", response_model=AntecedentesOut)
def get_antecedentes(paciente_id: UUID):
    """
    Devuelve los antecedentes médicos de un paciente.
    Retorna 404 si el paciente no tiene antecedentes registrados aún.
    """
    try:
        antecedentes = obtener_antecedentes(str(paciente_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not antecedentes:
        raise HTTPException(
            status_code=404,
            detail="Este paciente aun no tiene antecedentes registrados."
        )

    return antecedentes


# ── POST /antecedentes ─────────────────────────────────────────────────────
@router.post("", response_model=AntecedentesOut, status_code=201)
def post_antecedentes(body: AntecedentesCreate):
    """
    Crea los antecedentes médicos de un paciente.
    Solo se puede crear una vez por paciente (restricción UNIQUE en paciente_id).
    Para actualizar, usar el endpoint PUT.
    """
    # Verificar que el paciente existe
    paciente = obtener_paciente(str(body.paciente_id))
    if not paciente:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    try:
        datos = body.model_dump(exclude_none=True)
        if "paciente_id" in datos:
            datos["paciente_id"] = str(datos["paciente_id"])
        return crear_antecedentes(datos)
    except Exception as e:
        error_str = str(e)
        if "duplicate" in error_str.lower() or "unique" in error_str.lower():
            raise HTTPException(
                status_code=409,
                detail="Este paciente ya tiene antecedentes registrados. Use PUT para actualizarlos."
            )
        raise HTTPException(status_code=500, detail=error_str)


# ── PUT /antecedentes/{paciente_id} ───────────────────────────────────────
@router.put("/{paciente_id}", response_model=AntecedentesOut)
def put_antecedentes(paciente_id: UUID, body: AntecedentesUpdate):
    """
    Actualiza los antecedentes médicos de un paciente.
    Solo se modifican los campos enviados en el body.
    """
    try:
        datos = body.model_dump(exclude_none=True)
        if not datos:
            raise HTTPException(status_code=400, detail="No se enviaron campos para actualizar")

        actualizado = actualizar_antecedentes(str(paciente_id), datos)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not actualizado:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron antecedentes para este paciente. Use POST para crearlos."
        )

    return actualizado


# ── POST /antecedentes/guardar (upsert) ────────────────────────────────────
@router.post("/guardar", response_model=AntecedentesOut)
def guardar_antecedentes(body: AntecedentesCreate):
    """
    Crea o actualiza los antecedentes médicos de un paciente (upsert).
    Si ya existen, los actualiza. Si no, los crea.
    Ideal para el botón 'Guardar' del formulario del frontend.
    """
    paciente = obtener_paciente(str(body.paciente_id))
    if not paciente:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    try:
        datos = body.model_dump(exclude_none=True)
        paciente_id_str = str(body.paciente_id)
        datos.pop("paciente_id", None)  # La función lo agrega internamente
        return crear_o_actualizar_antecedentes(paciente_id_str, datos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
