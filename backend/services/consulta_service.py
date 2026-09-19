from typing import List, Optional
from datetime import datetime, timezone
from db.supabase_client import supabase
from schemas.consulta_schema import ConsultaCreate, ConsultaUpdate
from services import medico_service

def _formatear_fecha_atencion(fecha_str: Optional[str]) -> str:
    if not fecha_str:
        return datetime.now(timezone.utc).isoformat()
    if "T" in str(fecha_str):
        return str(fecha_str)
    return f"{fecha_str}T09:00:00Z"

def crear_consulta(consulta: ConsultaCreate) -> dict:
    datos = consulta.model_dump(exclude_unset=True)
    fecha_prox = datos.pop("fecha_proxima_cita", None)
    
    if not datos.get("medico_id"):
        datos["medico_id"] = medico_service.obtener_medico_predeterminado_id()
        
    cita_id = datos.get("cita_id")
    
    # Si viene cita_id, verificar si realmente existe en la tabla cita de forma segura
    if cita_id:
        try:
            res_cita = supabase.table("cita").select("*").eq("id", cita_id).execute()
            if not res_cita.data or len(res_cita.data) == 0:
                datos["cita_id"] = None
            else:
                # Si existe la cita, verificar si ya tiene consulta vinculada para evitar duplicados
                res_exist = supabase.table("consulta").select("*").eq("cita_id", cita_id).execute()
                if res_exist.data and len(res_exist.data) > 0:
                    if fecha_prox:
                        supabase.table("cita").insert({
                            "paciente_id": res_exist.data[0].get("paciente_id"),
                            "medico_id": res_exist.data[0].get("medico_id"),
                            "fecha_atencion": _formatear_fecha_atencion(fecha_prox),
                            "fecha_proxima_cita": str(fecha_prox).split("T")[0],
                            "motivo": "Próximo control de seguimiento SOP",
                            "observaciones": "Programado desde consulta médica",
                            "estado": "programada"
                        }).execute()
                    return res_exist.data[0]
        except Exception as e_cita:
            print("Cita ID invalido o no encontrado, se creara una nueva cita:", e_cita)
            datos["cita_id"] = None

    # Si no tiene cita_id o la cita previa no existía, crear la cita de atención actual con estado 'atendida'
    if not datos.get("cita_id"):
        paciente_id = datos.get("paciente_id")
        if not paciente_id:
            raise Exception("Debe seleccionar una paciente para registrar la consulta médica")
        
        cita_datos = {
            "paciente_id": paciente_id,
            "medico_id": datos.get("medico_id"),
            "motivo": datos.get("motivo") or "Consulta médica directa",
            "observaciones": datos.get("observaciones"),
            "estado": "atendida"
        }
        res_cita_nueva = supabase.table("cita").insert(cita_datos).execute()
        if res_cita_nueva.data:
            datos["cita_id"] = res_cita_nueva.data[0]["id"]
        else:
            raise Exception("No se pudo generar el registro de cita requerido para la consulta")

    res = supabase.table("consulta").insert(datos).execute()
    if res.data:
        # Asegurar que la cita de esta consulta quede marcada como atendida
        if datos.get("cita_id"):
            supabase.table("cita").update({"estado": "atendida"}).eq("id", datos["cita_id"]).execute()

        # Generar la próxima cita futura de control en estado 'programada'
        if fecha_prox:
            try:
                supabase.table("cita").insert({
                    "paciente_id": datos.get("paciente_id"),
                    "medico_id": datos.get("medico_id"),
                    "fecha_atencion": _formatear_fecha_atencion(fecha_prox),
                    "fecha_proxima_cita": str(fecha_prox).split("T")[0],
                    "motivo": "Próximo control de seguimiento SOP",
                    "observaciones": "Programado desde consulta médica",
                    "estado": "programada"
                }).execute()
            except Exception as e_prox:
                print("Error al agendar fecha_proxima_cita automática:", e_prox)

        return res.data[0]
    raise Exception("No se pudo crear la consulta médica")

def obtener_consultas(paciente_id: Optional[str] = None, medico_id: Optional[str] = None) -> List[dict]:
    query = supabase.table("consulta").select("*")
    if paciente_id:
        query = query.eq("paciente_id", paciente_id)
    if medico_id:
        query = query.eq("medico_id", medico_id)
    res = query.order("created_at", desc=True).execute()
    return res.data or []

def obtener_consulta_por_id(consulta_id: str) -> Optional[dict]:
    res = supabase.table("consulta").select("*").eq("id", consulta_id).execute()
    if res.data:
        return res.data[0]
    return None

def actualizar_consulta(consulta_id: str, consulta: ConsultaUpdate) -> Optional[dict]:
    datos = consulta.model_dump(exclude_none=True)
    datos["updated_at"] = datetime.now(timezone.utc).isoformat()
    res = supabase.table("consulta").update(datos).eq("id", consulta_id).execute()
    if res.data and len(res.data) > 0:
        consulta_actualizada = res.data[0]
        cita_id = consulta_actualizada.get("cita_id")
        nuevo_estado = datos.get("estado")
        if cita_id and nuevo_estado:
            if nuevo_estado in ["completada", "finalizada"]:
                supabase.table("cita").update({"estado": "atendida"}).eq("id", cita_id).execute()
            elif nuevo_estado == "cancelada":
                supabase.table("cita").update({"estado": "cancelada"}).eq("id", cita_id).execute()
        return consulta_actualizada
    return None

def eliminar_consulta(consulta_id: str) -> bool:
    res = supabase.table("consulta").delete().eq("id", consulta_id).execute()
    return bool(res.data)
