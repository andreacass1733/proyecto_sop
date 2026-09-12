"""
Servicio — Módulo de Pacientes y Antecedentes Médicos.

Toda la lógica de negocio que interactúa con Supabase:
  - CRUD de pacientes (con soft-delete)
  - CRUD de antecedentes médicos
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from db.supabase_client import supabase

TABLA_PACIENTE = "paciente"
TABLA_ANTECEDENTES = "antecedentes_medicos"


# ══════════════════════════════════════════════════════════════════════════════
# PACIENTES
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MEDICO_ID = "0a8eaba2-be5b-4dbd-b22b-fc35c5616c12"


def listar_pacientes(
    activo: Optional[bool] = True,
    busqueda: Optional[str] = None,
    pagina: int = 1,
    por_pagina: int = 20,
) -> list[dict]:
    """
    Devuelve la lista de pacientes con paginación y filtros.

    Args:
        activo    : Filtrar por estado activo (True), inactivo (False) o None (todos).
        busqueda  : Filtro opcional por nombre, apellido, CI o email.
        pagina    : Número de página (desde 1).
        por_pagina: Cantidad de resultados por página.

    Returns:
        Lista de registros de pacientes.
    """
    offset = (pagina - 1) * por_pagina
    query = (
        supabase.table(TABLA_PACIENTE)
        .select("id, nombre, primer_apellido, segundo_apellido, ci, telefono, email, fecha_nacimiento, activo, medico_id, created_at")
        .order("created_at", desc=True)
        .range(offset, offset + por_pagina - 1)
    )

    if activo is not None:
        query = query.eq("activo", activo)

    if busqueda:
        query = query.or_(
            f"nombre.ilike.%{busqueda}%,primer_apellido.ilike.%{busqueda}%,segundo_apellido.ilike.%{busqueda}%,ci.ilike.%{busqueda}%,email.ilike.%{busqueda}%"
        )

    resultado = query.execute()
    return resultado.data or []


def obtener_paciente(paciente_id: str) -> Optional[dict]:
    """
    Devuelve el detalle completo de un paciente por su ID.

    Returns:
        Diccionario con todos los campos, o None si no existe.
    """
    resultado = (
        supabase.table(TABLA_PACIENTE)
        .select("*")
        .eq("id", paciente_id)
        .execute()
    )
    datos = resultado.data
    return datos[0] if datos else None


def crear_paciente(datos: dict) -> dict:
    """
    Inserta un nuevo paciente en la base de datos vinculado al médico principal.

    Args:
        datos: Diccionario con los campos del paciente.

    Returns:
        El registro creado con todos sus campos.
    """
    # Convertir fechas a string si vienen como date
    if "fecha_nacimiento" in datos and datos["fecha_nacimiento"] is not None:
        if hasattr(datos["fecha_nacimiento"], "isoformat"):
            datos["fecha_nacimiento"] = datos["fecha_nacimiento"].isoformat()

    # Si no se proporciona medico_id, usar el ID del Dr. Rodrigo Espinoza
    if not datos.get("medico_id"):
        datos["medico_id"] = DEFAULT_MEDICO_ID

    resultado = supabase.table(TABLA_PACIENTE).insert(datos).execute()
    return resultado.data[0]


def actualizar_paciente(paciente_id: str, datos: dict) -> Optional[dict]:
    """
    Actualiza los campos de un paciente.

    Solo actualiza los campos que vienen en 'datos' (no None).

    Returns:
        El registro actualizado, o None si el paciente no existe.
    """
    # Eliminar campos None para no sobreescribir datos existentes
    datos_limpios = {k: v for k, v in datos.items() if v is not None}

    if "fecha_nacimiento" in datos_limpios and datos_limpios["fecha_nacimiento"] is not None:
        if hasattr(datos_limpios["fecha_nacimiento"], "isoformat"):
            datos_limpios["fecha_nacimiento"] = datos_limpios["fecha_nacimiento"].isoformat()

    datos_limpios["updated_at"] = datetime.now(timezone.utc).isoformat()

    resultado = (
        supabase.table(TABLA_PACIENTE)
        .update(datos_limpios)
        .eq("id", paciente_id)
        .execute()
    )
    datos = resultado.data
    return datos[0] if datos else None


def desactivar_paciente(paciente_id: str) -> Optional[dict]:
    """
    Soft-delete: marca al paciente como inactivo en lugar de eliminarlo.

    Esto preserva el historial clínico sin borrar datos de la base de datos.

    Returns:
        El registro actualizado, o None si no existe.
    """
    return actualizar_paciente(paciente_id, {"activo": False})


def contar_pacientes(activo: bool = True) -> int:
    """Devuelve el total de pacientes activos/inactivos."""
    resultado = (
        supabase.table(TABLA_PACIENTE)
        .select("id", count="exact")
        .eq("activo", activo)
        .execute()
    )
    return resultado.count or 0


# ══════════════════════════════════════════════════════════════════════════════
# ANTECEDENTES MÉDICOS
# ══════════════════════════════════════════════════════════════════════════════

def obtener_antecedentes(paciente_id: str) -> Optional[dict]:
    """
    Devuelve los antecedentes médicos de un paciente.

    Cada paciente tiene exactamente un registro de antecedentes (1-a-1).

    Returns:
        Diccionario con los antecedentes, o None si no se han registrado aún.
    """
    resultado = (
        supabase.table(TABLA_ANTECEDENTES)
        .select("*")
        .eq("paciente_id", paciente_id)
        .execute()
    )
    datos = resultado.data
    return datos[0] if datos else None


def crear_antecedentes(datos: dict) -> dict:
    """
    Crea el registro de antecedentes médicos para un paciente.

    Si ya existe un registro para ese paciente_id, lanza un error (restricción UNIQUE).

    Returns:
        El registro creado.
    """
    # Convertir UUID a string si es necesario
    if "paciente_id" in datos and isinstance(datos["paciente_id"], UUID):
        datos["paciente_id"] = str(datos["paciente_id"])

    try:
        resultado = supabase.table(TABLA_ANTECEDENTES).insert(datos).execute()
        return resultado.data[0]
    except Exception as e:
        error_msg = str(e).lower()
        # Si falla por columna faltante en Supabase (ej. hiperprolactinemia o iniciada_vida_sexual), reintentar sin ella
        modificado = False
        for col in ["hiperprolactinemia", "iniciada_vida_sexual", "historial"]:
            if col in datos and (col in error_msg or "column" in error_msg or "schema" in error_msg):
                datos.pop(col, None)
                modificado = True
        if modificado:
            resultado = supabase.table(TABLA_ANTECEDENTES).insert(datos).execute()
            return resultado.data[0]
        raise e


def actualizar_antecedentes(paciente_id: str, datos: dict) -> Optional[dict]:
    """
    Actualiza los antecedentes médicos de un paciente.

    Si no existe el registro, devuelve None.

    Returns:
        El registro actualizado.
    """
    datos_limpios = {k: v for k, v in datos.items() if v is not None}
    datos_limpios["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        resultado = (
            supabase.table(TABLA_ANTECEDENTES)
            .update(datos_limpios)
            .eq("paciente_id", paciente_id)
            .execute()
        )
        datos_res = resultado.data
        return datos_res[0] if datos_res else None
    except Exception as e:
        error_msg = str(e).lower()
        modificado = False
        for col in ["hiperprolactinemia", "iniciada_vida_sexual", "historial"]:
            if col in datos_limpios and (col in error_msg or "column" in error_msg or "schema" in error_msg):
                datos_limpios.pop(col, None)
                modificado = True
        if modificado:
            resultado = (
                supabase.table(TABLA_ANTECEDENTES)
                .update(datos_limpios)
                .eq("paciente_id", paciente_id)
                .execute()
            )
            datos_res = resultado.data
            return datos_res[0] if datos_res else None
        raise e


def crear_o_actualizar_antecedentes(paciente_id: str, datos: dict) -> dict:
    """
    Crea los antecedentes si no existen, o los actualiza si ya existen.
    Esto simplifica el flujo del frontend (un solo botón "Guardar").

    Returns:
        El registro creado o actualizado.
    """
    existente = obtener_antecedentes(paciente_id)
    if existente:
        res = actualizar_antecedentes(paciente_id, datos)
        if res is not None:
            return res
        datos["paciente_id"] = paciente_id
        return crear_antecedentes(datos)
    else:
        datos["paciente_id"] = paciente_id
        return crear_antecedentes(datos)
