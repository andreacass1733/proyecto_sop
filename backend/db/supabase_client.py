import os
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "ecografias")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Faltan credenciales de Supabase. Define SUPABASE_URL y SUPABASE_KEY en .env"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase_admin_client() -> Client:
    """Retorna un cliente limpio con service_role sin contaminación de sesión."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)