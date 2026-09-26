from typing import List, Optional
from datetime import datetime, timezone, timedelta
from db.supabase_client import supabase
from schemas.cita_schema import CitaCreate, CitaUpdate

from services import medico_service

def auto_cancelar_citas_vencidas():
    """
    Cancela automáticamente cualquier cita en estado 'programada' si ha transcurrido
    más de 1 hora respecto a su 'fecha_atencion'.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        res_programadas = supabase.table("cita").select("*").eq("estado", "programada").execute()
        if res_programadas.data:
            for c in res_programadas.data:
                fecha_str = c.get("fecha_atencion")
                if fecha_str:
                    try:
                        # Normalizar ISO string
                        clean_str = str(fecha_str).replace("Z", "+00:00")
                        dt_atencion = datetime.fromisoformat(clean_str)
                        if dt_atencion.tzinfo is None:
                            dt_atencion = dt_atencion.replace(tzinfo=timezone.utc)
                        
                        # Si han pasado más de 60 minutos desde la hora pautada
                        if now_utc - dt_atencion > timedelta(hours=1):
                            obs_prev = c.get("observaciones") or ""
                            motivo_cancel = "[Cancelada automáticamente por retraso superado a 1 hora]"
                            nueva_obs = f"{obs_prev} {motivo_cancel}".strip() if obs_prev else motivo_cancel
                            
                            supabase.table("cita").update({
                                "estado": "cancelada",
                                "observaciones": nueva_obs,
                                "updated_at": now_utc.isoformat()
                            }).eq("id", c["id"]).execute()
                    except Exception as e_parse:
                        print("Error al evaluar fecha de cita vencida:", e_parse)
    except Exception as e:
        print("Error al ejecutar auto_cancelar_citas_vencidas:", e)

def crear_cita(cita: CitaCreate) -> dict:
    datos = cita.model_dump(exclude_unset=True)
    if not datos.get("medico_id"):
        datos["medico_id"] = medico_service.obtener_medico_predeterminado_id()

    if not datos.get("fecha_atencion"):
        datos["fecha_atencion"] = datetime.now(timezone.utc).isoformat()
    elif isinstance(datos["fecha_atencion"], datetime):
        fecha = datos["fecha_atencion"]
        if fecha < datetime.now(timezone.utc):
            raise Exception("No se puede agendar una cita en el pasado.")
        datos["fecha_atencion"] = fecha.isoformat()
        
    res = supabase.table("cita").insert(datos).execute()
    if res.data:
        return res.data[0]
    raise Exception("No se pudo crear la cita")

def obtener_citas(paciente_id: Optional[str] = None, medico_id: Optional[str] = None) -> List[dict]:
    # Primero auto-cancelar cualquier cita vencida (+1 hora de tolerancia)
    auto_cancelar_citas_vencidas()

    query = supabase.table("cita").select("*")
    if paciente_id:
        query = query.eq("paciente_id", paciente_id)
    if medico_id:
        query = query.eq("medico_id", medico_id)
    res = query.order("created_at", desc=True).execute()
    return res.data or []

def obtener_cita_por_id(cita_id: str) -> Optional[dict]:
    auto_cancelar_citas_vencidas()
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
