"""
Router — Gestión de Pacientes.

Rutas disponibles:
    GET    /pacientes              → Listar pacientes (con paginación y búsqueda)
    GET    /pacientes/total        → Total de pacientes activos
    GET    /pacientes/{id}         → Detalle de un paciente
    POST   /pacientes              → Crear nuevo paciente
    PUT    /pacientes/{id}         → Editar paciente
    DELETE /pacientes/{id}         → Desactivar paciente (soft delete)
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from schemas.paciente_schema import (
    PacienteCreate,
    PacienteUpdate,
    PacienteOut,
    PacienteListItem,
)
from services.paciente_service import (
    listar_pacientes,
    obtener_paciente,
    crear_paciente,
    actualizar_paciente,
    desactivar_paciente,
    contar_pacientes,
)

router = APIRouter(prefix="/pacientes", tags=["Pacientes"])


# ── GET /pacientes ─────────────────────────────────────────────────────────
@router.get("", response_model=list[PacienteListItem])
def get_pacientes(
    busqueda: Optional[str] = Query(None, description="Filtrar por nombre, apellido, CI o email"),
    activo: Optional[bool] = Query(None, description="Listar activos (true), inactivos (false) o todos (omitir)"),
    pagina: int = Query(1, ge=1),
    por_pagina: int = Query(20, ge=1, le=100),
):
    """
    Lista todos los pacientes con paginación y filtro opcional.
    Por defecto devuelve pacientes activos ordenados por apellido.
    """
    try:
        return listar_pacientes(
            activo=activo,
            busqueda=busqueda,
            pagina=pagina,
            por_pagina=por_pagina,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /pacientes/total ───────────────────────────────────────────────────
@router.get("/total")
def get_total_pacientes():
    """Devuelve el total de pacientes activos registrados."""
    try:
        total = contar_pacientes(activo=True)
        return {"total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /pacientes/{id} ────────────────────────────────────────────────────
@router.get("/{paciente_id}", response_model=PacienteOut)
def get_paciente(paciente_id: UUID):
    """Devuelve el detalle completo de un paciente por su UUID."""
    try:
        paciente = obtener_paciente(str(paciente_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not paciente:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    return paciente


# ── POST /pacientes ────────────────────────────────────────────────────────
@router.post("", response_model=PacienteOut, status_code=201)
def post_paciente(body: PacienteCreate):
    """
    Registra un nuevo paciente en el sistema.
    El campo CI es único — si ya existe, Supabase retorna error de duplicado.
    """
    try:
        datos = body.model_dump(exclude_none=True)
        return crear_paciente(datos)
    except Exception as e:
        error_str = str(e)
        if "duplicate" in error_str.lower() or "unique" in error_str.lower():
            raise HTTPException(
                status_code=409,
                detail="Ya existe un paciente con esa Cédula de Identidad (CI)."
            )
        raise HTTPException(status_code=500, detail=error_str)


# ── PUT /pacientes/{id} ────────────────────────────────────────────────────
@router.put("/{paciente_id}", response_model=PacienteOut)
def put_paciente(paciente_id: UUID, body: PacienteUpdate):
    """
    Actualiza los datos de un paciente.
    Solo se modifican los campos incluidos en el body (los demás permanecen intactos).
    """
    try:
        datos = body.model_dump(exclude_none=True)
        if not datos:
            raise HTTPException(status_code=400, detail="No se enviaron campos para actualizar")

        actualizado = actualizar_paciente(str(paciente_id), datos)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not actualizado:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    return actualizado


# ── DELETE /pacientes/{id} (soft delete) ───────────────────────────────────
@router.delete("/{paciente_id}")
def delete_paciente(paciente_id: UUID):
    """
    Desactiva un paciente (soft delete).
    No se elimina de la base de datos para preservar el historial clínico.
    """
    try:
        resultado = desactivar_paciente(str(paciente_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not resultado:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")

    return {"mensaje": "Paciente desactivado correctamente", "id": str(paciente_id)}
