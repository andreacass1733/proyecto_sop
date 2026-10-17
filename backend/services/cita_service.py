from typing import List, Optional
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
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

def _normalizar_estado_db(estado: Optional[str]) -> Optional[str]:
    if not estado:
        return estado
    est = estado.strip().lower()
    if est in ["no asistió", "no asistio", "no_asistio"]:
        return "no_asistio"
    return est

def _formatear_cita(cita: Optional[dict]) -> Optional[dict]:
    if not cita:
        return cita
    if cita.get("estado") == "no_asistio":
        cita["estado"] = "no asistió"
    return cita

def crear_cita(cita: CitaCreate) -> dict:
    """
    Registra una cita médica con fecha de atención y fecha tentativa de próxima cita (HU-007).
    Valida que la fecha tentativa de próxima cita sea posterior a la fecha de atención.
    El estado inicial es 'programada' por defecto.
    """
    datos = cita.model_dump(exclude_unset=True)
    if not datos.get("medico_id"):
        datos["medico_id"] = medico_service.obtener_medico_predeterminado_id()

    if not datos.get("estado"):
        datos["estado"] = "programada"
    else:
        datos["estado"] = _normalizar_estado_db(datos["estado"])

    if not datos.get("fecha_atencion"):
        datos["fecha_atencion"] = datetime.now(timezone.utc).isoformat()
    elif isinstance(datos["fecha_atencion"], datetime):
        fecha = datos["fecha_atencion"]
        if fecha < datetime.now(timezone.utc):
            raise HTTPException(status_code=400, detail="No se puede agendar una cita en el pasado.")
        datos["fecha_atencion"] = fecha.isoformat()

    # Validación Criterio HU-007: La fecha tentativa de próxima cita debe ser posterior a la de atención
    if datos.get("fecha_proxima_cita") and datos.get("fecha_atencion"):
        try:
            str_atencion = str(datos["fecha_atencion"]).replace("Z", "+00:00")
            dt_atencion = datetime.fromisoformat(str_atencion)
            if dt_atencion.tzinfo is None:
                dt_atencion = dt_atencion.replace(tzinfo=timezone.utc)

            str_prox = str(datos["fecha_proxima_cita"]).replace("Z", "+00:00")
            if "T" in str_prox:
                dt_prox = datetime.fromisoformat(str_prox)
            else:
                dt_prox = datetime.fromisoformat(f"{str_prox}T23:59:59+00:00")
            if dt_prox.tzinfo is None:
                dt_prox = dt_prox.replace(tzinfo=timezone.utc)

            if dt_prox <= dt_atencion:
                raise HTTPException(
                    status_code=400,
                    detail="La fecha tentativa de próxima cita debe ser posterior a la fecha de atención."
                )
        except HTTPException:
            raise
        except Exception as e_date:
            pass

    res = supabase.table("cita").insert(datos).execute()
    if res.data:
        return _formatear_cita(res.data[0])
    raise HTTPException(status_code=500, detail="No se pudo crear la cita")

def obtener_citas(paciente_id: Optional[str] = None, medico_id: Optional[str] = None) -> List[dict]:
    # Primero auto-cancelar cualquier cita vencida (+1 hora de tolerancia)
    auto_cancelar_citas_vencidas()

    query = supabase.table("cita").select("*")
    if paciente_id:
        query = query.eq("paciente_id", paciente_id)
    if medico_id:
        query = query.eq("medico_id", medico_id)
    res = query.order("created_at", desc=True).execute()
    return [_formatear_cita(c) for c in (res.data or [])]

def obtener_cita_por_id(cita_id: str) -> Optional[dict]:
    auto_cancelar_citas_vencidas()
    res = supabase.table("cita").select("*").eq("id", cita_id).execute()
    if res.data:
        return _formatear_cita(res.data[0])
    return None

def actualizar_cita(cita_id: str, cita: CitaUpdate) -> Optional[dict]:
    datos = cita.model_dump(exclude_none=True)
    datos["updated_at"] = datetime.now(timezone.utc).isoformat()
    if "fecha_atencion" in datos and isinstance(datos["fecha_atencion"], datetime):
        datos["fecha_atencion"] = datos["fecha_atencion"].isoformat()

    if "estado" in datos and datos["estado"]:
        datos["estado"] = _normalizar_estado_db(datos["estado"])

    res = supabase.table("cita").update(datos).eq("id", cita_id).execute()
    if res.data:
        return _formatear_cita(res.data[0])
    return None

def eliminar_cita(cita_id: str) -> bool:
    res = supabase.table("cita").delete().eq("id", cita_id).execute()
    return bool(res.data)
