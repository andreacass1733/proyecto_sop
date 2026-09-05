# test_conexion.py
from db.supabase_client import supabase

def test_conexion():
    try:
        respuesta = supabase.table("medico").select("*").limit(1).execute()
        print("✅ Conexión exitosa con Supabase")
        print(f"   Datos recibidos: {respuesta.data}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")

if __name__ == "__main__":
    test_conexion()