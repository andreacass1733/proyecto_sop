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
    Actualiza los campos de un paciente y guarda automáticamente la versión anterior
    en el campo 'historial' (JSONB) con la fecha y hora del cambio (HU-004).

    Returns:
        El registro actualizado, o None si el paciente no existe.
    """
    # 1. Obtener registro actual antes de actualizar
    actual = obtener_paciente(paciente_id)
    if not actual:
        return None

    # Eliminar campos None para no sobreescribir datos existentes
    datos_limpios = {k: v for k, v in datos.items() if v is not None}

    if "fecha_nacimiento" in datos_limpios and datos_limpios["fecha_nacimiento"] is not None:
        if hasattr(datos_limpios["fecha_nacimiento"], "isoformat"):
            datos_limpios["fecha_nacimiento"] = datos_limpios["fecha_nacimiento"].isoformat()

    # 2. Construir historial de versiones anteriores en formato JSONB (HU-004)
    historial_previo = actual.get("historial")
    if not isinstance(historial_previo, list):
        historial_previo = []

    # Extraer versión previa de los campos modificables
    snapshot = {
        "telefono": actual.get("telefono"),
        "email": actual.get("email"),
        "nombre": actual.get("nombre"),
        "primer_apellido": actual.get("primer_apellido"),
        "segundo_apellido": actual.get("segundo_apellido"),
        "ci": actual.get("ci"),
        "fecha_nacimiento": str(actual.get("fecha_nacimiento")) if actual.get("fecha_nacimiento") else None,
        "activo": actual.get("activo"),
    }

    # Solo agregamos al historial si hay cambios efectivos
    hubo_cambios = any(
        k in datos_limpios and datos_limpios[k] != actual.get(k)
        for k in ["telefono", "email", "nombre", "primer_apellido", "segundo_apellido", "ci", "fecha_nacimiento", "activo"]
    )

    if hubo_cambios:
        nuevo_registro_historial = {
            "fecha_modificacion": datetime.now(timezone.utc).isoformat(),
            "version_anterior": snapshot,
        }
        historial_previo.append(nuevo_registro_historial)
        datos_limpios["historial"] = historial_previo

    datos_limpios["updated_at"] = datetime.now(timezone.utc).isoformat()

    resultado = (
        supabase.table(TABLA_PACIENTE)
        .update(datos_limpios)
        .eq("id", paciente_id)
        .execute()
    )
    datos_res = resultado.data
    return datos_res[0] if datos_res else None


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
    if not datos:
        return None

    registro = datos[0]

    # Evaluar iniciada_vida_sexual si no viene explícita o si tiene antecedentes obstétricos
    tiene_obstetricos = any(
        (registro.get(k) or 0) > 0
        for k in ["gestas", "partos", "cesareas", "abortos", "hijos_vivos"]
    )
    meta_sex = False
    historial = registro.get("historial")
    if isinstance(historial, list):
        for h in historial:
            if isinstance(h, dict) and "__iniciada_vida_sexual__" in h:
                meta_sex = bool(h["__iniciada_vida_sexual__"])
    elif isinstance(historial, dict):
        meta_sex = bool(historial.get("__iniciada_vida_sexual__", False))

    registro["iniciada_vida_sexual"] = bool(
        registro.get("iniciada_vida_sexual") or tiene_obstetricos or meta_sex
    )
    return registro


def crear_antecedentes(datos: dict) -> dict:
    """
    Crea el registro de antecedentes médicos para un paciente.

    Si ya existe un registro para ese paciente_id, lanza un error (restricción UNIQUE).

    Returns:
        El registro creado.
    """
    if "paciente_id" in datos and isinstance(datos["paciente_id"], UUID):
        datos["paciente_id"] = str(datos["paciente_id"])

    iniciada_sex = datos.pop("iniciada_vida_sexual", None)
    
    # Si historial no existe o es nulo, inicializar
    historial = datos.get("historial")
    if not isinstance(historial, list):
        historial = []
    if iniciada_sex is not None:
        historial.append({"__iniciada_vida_sexual__": bool(iniciada_sex)})
        datos["historial"] = historial

    try:
        resultado = supabase.table(TABLA_ANTECEDENTES).insert(datos).execute()
        res = resultado.data[0]
        tiene_obs = any((res.get(k) or 0) > 0 for k in ["gestas", "partos", "cesareas", "abortos", "hijos_vivos"])
        res["iniciada_vida_sexual"] = bool(iniciada_sex if iniciada_sex is not None else tiene_obs)
        return res
    except Exception as e:
        error_msg = str(e).lower()
        modificado = False
        for col in ["hiperprolactinemia", "iniciada_vida_sexual", "historial"]:
            if col in datos and (col in error_msg or "column" in error_msg or "schema" in error_msg):
                datos.pop(col, None)
                modificado = True
        if modificado:
            resultado = supabase.table(TABLA_ANTECEDENTES).insert(datos).execute()
            res = resultado.data[0]
            tiene_obs = any((res.get(k) or 0) > 0 for k in ["gestas", "partos", "cesareas", "abortos", "hijos_vivos"])
            res["iniciada_vida_sexual"] = bool(iniciada_sex if iniciada_sex is not None else tiene_obs)
            return res
        raise e


def actualizar_antecedentes(paciente_id: str, datos: dict) -> Optional[dict]:
    """
    Actualiza los antecedentes médicos de un paciente y conserva la versión anterior
    en el historial JSONB con la fecha y hora del cambio (HU-006).

    Returns:
        El registro actualizado.
    """
    existente = obtener_antecedentes(paciente_id)
    if not existente:
        return None

    iniciada_sex = datos.pop("iniciada_vida_sexual", None)
    datos_limpios = {k: v for k, v in datos.items() if v is not None}

    # Conservar versión anterior en historial JSONB (HU-006)
    historial_previo = existente.get("historial")
    if not isinstance(historial_previo, list):
        historial_previo = []

    # Extraer copia de los campos clínicos existentes
    snapshot = {
        k: v for k, v in existente.items()
        if k not in ["id", "paciente_id", "created_at", "updated_at", "historial", "iniciada_vida_sexual"]
    }

    hubo_cambios = any(
        k in datos_limpios and datos_limpios[k] != existente.get(k)
        for k in snapshot.keys()
    )

    if hubo_cambios:
        nuevo_historial = {
            "fecha_modificacion": datetime.now(timezone.utc).isoformat(),
            "version_anterior": snapshot,
        }
        historial_previo.append(nuevo_historial)

    if iniciada_sex is not None:
        # Actualizar o agregar marcador en historial
        marcador_encontrado = False
        for entry in historial_previo:
            if isinstance(entry, dict) and "__iniciada_vida_sexual__" in entry:
                entry["__iniciada_vida_sexual__"] = bool(iniciada_sex)
                marcador_encontrado = True
                break
        if not marcador_encontrado:
            historial_previo.append({"__iniciada_vida_sexual__": bool(iniciada_sex)})

    datos_limpios["historial"] = historial_previo
    datos_limpios["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        resultado = (
            supabase.table(TABLA_ANTECEDENTES)
            .update(datos_limpios)
            .eq("paciente_id", paciente_id)
            .execute()
        )
        datos_res = resultado.data
        if not datos_res:
            return None
        res = datos_res[0]
        tiene_obs = any((res.get(k) or 0) > 0 for k in ["gestas", "partos", "cesareas", "abortos", "hijos_vivos"])
        res["iniciada_vida_sexual"] = bool(iniciada_sex if iniciada_sex is not None else tiene_obs)
        return res
    except Exception as e:
        error_msg = str(e).lower()
        modificado = False
        for col in ["hiperprolactinemia", "iniciada_vida_sexual"]:
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
            if not datos_res:
                return None
            res = datos_res[0]
            tiene_obs = any((res.get(k) or 0) > 0 for k in ["gestas", "partos", "cesareas", "abortos", "hijos_vivos"])
            res["iniciada_vida_sexual"] = bool(iniciada_sex if iniciada_sex is not None else tiene_obs)
            return res
        raise e


def crear_o_actualizar_antecedentes(paciente_id: str, datos: dict) -> dict:
    """
    Crea los antecedentes si no existen, o los actualiza si ya existen.
    Esto simplifica el flujo del frontend (un solo botón "Guardar").

    Returns:
        El registro creado o actualizado.
    """
    sex_val = datos.get("iniciada_vida_sexual")
    existente = obtener_antecedentes(paciente_id)
    if existente:
        res = actualizar_antecedentes(paciente_id, datos)
        if res is not None:
            if sex_val is not None:
                res["iniciada_vida_sexual"] = bool(sex_val)
            return res
        datos["paciente_id"] = paciente_id
        res = crear_antecedentes(datos)
    else:
        datos["paciente_id"] = paciente_id
        res = crear_antecedentes(datos)

    if res and sex_val is not None:
        res["iniciada_vida_sexual"] = bool(sex_val)
    return res
