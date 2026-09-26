from typing import List, Optional
from datetime import datetime, timezone
from db.supabase_client import supabase
from schemas.cita_schema import CitaCreate, CitaUpdate

from services import medico_service

def crear_cita(cita: CitaCreate) -> dict:
    datos = cita.model_dump(exclude_unset=True)
    if not datos.get("medico_id"):
        datos["medico_id"] = medico_service.obtener_medico_predeterminado_id()

    if not datos.get("fecha_atencion"):
        datos["fecha_atencion"] = datetime.now(timezone.utc).isoformat()
    elif isinstance(datos["fecha_atencion"], datetime):
        fecha = datos["fecha_atencion"]
        # Basic validation (hours 7-19) on backend (using UTC hour could be tricky depending on timezone, so we assume frontend sends the right constraint, but we block exactly past times easily)
        if fecha < datetime.now(timezone.utc):
            raise Exception("No se puede agendar una cita en el pasado.")
        datos["fecha_atencion"] = fecha.isoformat()
        
    res = supabase.table("cita").insert(datos).execute()
    if res.data:
        return res.data[0]
    raise Exception("No se pudo crear la cita")

def obtener_citas(paciente_id: Optional[str] = None, medico_id: Optional[str] = None) -> List[dict]:
    query = supabase.table("cita").select("*")
    if paciente_id:
        query = query.eq("paciente_id", paciente_id)
    if medico_id:
        query = query.eq("medico_id", medico_id)
    res = query.order("created_at", desc=True).execute()
    return res.data or []

def obtener_cita_por_id(cita_id: str) -> Optional[dict]:
    res = supabase.table("cita").select("*").eq("id", cita_id).execute()
    if res.data:
        return res.data[0]
    return None

def actualizar_cita(cita_id: str, cita: CitaUpdate) -> Optional[dict]:
    datos = cita.model_dump(exclude_none=True)
    datos["updated_at"] = datetime.now(timezone.utc).isoformat()
    if "fecha_atencion" in datos and isinstance(datos["fecha_atencion"], datetime):
        datos["fecha_atencion"] = datos["fecha_atencion"].isoformat()

    res = supabase.table("cita").update(datos).eq("id", cita_id).execute()
    if res.data:
        return res.data[0]
    return None

def eliminar_cita(cita_id: str) -> bool:
    res = supabase.table("cita").delete().eq("id", cita_id).execute()
    return bool(res.data)
