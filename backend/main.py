"""Punto de entrada de la API de SOP (FastAPI).

Configura la aplicación, el middleware de CORS, la precarga del modelo IA
y registra todos los routers de la arquitectura modular.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from routers import criterio3, pacientes, antecedentes, citas, consultas, medicos
from services.criterio3_service import get_model
from db.supabase_client import supabase

# ── Importar routers que ya existen ───────────────────────────────────────────
# criterio1, criterio2 y clasificacion se agregan cuando estén listos
# from routers import criterio1, criterio2, clasificacion


# ── Lifespan: se ejecuta al arrancar y al apagar el servidor ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Precarga el modelo IA al iniciar para evitar latencia en la primera petición."""
    print("[INFO] Iniciando servidor SOP...")
    try:
        get_model()
        print("[OK] Modelo EfficientNet-B0 precargado en memoria")
    except FileNotFoundError as e:
        print(f"[WARN] Advertencia: {e}")
    yield
    print("[INFO] Servidor SOP detenido")


# ── Aplicación principal ──────────────────────────────────────────────────────
app = FastAPI(
    title="SOP — API de Criterios de Rotterdam",
    description="Backend para la detección y clasificación del Síndrome de Ovario Poliquístico",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware de autenticación JWT (Supabase) ─────────────────────────────────
RUTAS_PUBLICAS = {"/", "/docs", "/openapi.json", "/redoc"}

# @app.middleware("http")
''' async def verificar_jwt(request: Request, call_next):
    """Valida el token JWT de Supabase en cada petición protegida."""
    if request.url.path in RUTAS_PUBLICAS:
        return await call_next(request)

    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        raise HTTPException(status_code=401, detail="Token de autenticación requerido")

    try:
        supabase.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

    return await call_next(request)
 '''

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(criterio3.router)
app.include_router(pacientes.router)
app.include_router(antecedentes.router)
app.include_router(citas.router)
app.include_router(consultas.router)
app.include_router(medicos.router)

# Se agregan cuando estén listos:
# app.include_router(criterio1.router)
# app.include_router(criterio2.router)
# app.include_router(clasificacion.router)


# ── Ruta de salud ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"mensaje": "API SOP funcionando OK", "version": "1.0.0"}