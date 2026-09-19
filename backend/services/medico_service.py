from typing import List, Optional
from db.supabase_client import supabase

def obtener_medicos() -> List[dict]:
    res = supabase.table("medico").select("*").execute()
    return res.data or []

def obtener_medico_predeterminado_id() -> Optional[str]:
    try:
        res = supabase.table("medico").select("id").limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["id"]
    except Exception as e:
        print("Error al obtener medico predeterminado:", e)
    return None
