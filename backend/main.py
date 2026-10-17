"""Punto de entrada de la API de SOP (FastAPI).

Configura la aplicación, el middleware de CORS, la precarga del modelo IA
y registra todos los routers de la arquitectura modular.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers import auth, criterio3, pacientes, antecedentes, citas, consultas, medicos
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware de autenticación JWT (Supabase Auth) ───────────────────────────
RUTAS_PUBLICAS = {"/", "/docs", "/openapi.json", "/redoc"}

@app.middleware("http")
async def verificar_jwt(request: Request, call_next):
    """
    Valida el token JWT de Supabase Auth en cada petición protegida.
    Garantiza que únicamente el médico especialista autenticado accede a la API.
    """
    # Permitir solicitudes preflight CORS
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if (
        path in RUTAS_PUBLICAS
        or path.startswith("/auth/login")
        or path.startswith("/auth/verify-2fa")
        or path.startswith("/auth/resend-2fa")
        or path.startswith("/auth/forgot-password")
        or path.startswith("/auth/reset-password")
        or path.startswith("/docs")
        or path.startswith("/openapi")
        or path.startswith("/redoc")
    ):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"detail": "Token de autenticación requerido"}
        )

    token = auth_header.replace("Bearer ", "").strip()
    if not token:
        return JSONResponse(
            status_code=401,
            content={"detail": "Token de autenticación requerido"}
        )

    try:
        user_info = supabase.auth.get_user(token)
        if not user_info or not user_info.user:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token inválido o expirado"}
            )
        request.state.user = user_info.user
    except Exception:
        return JSONResponse(
            status_code=401,
            content={"detail": "Token inválido o expirado"}
        )

    return await call_next(request)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
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