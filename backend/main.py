from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routers import criterio3

# Crear tablas automáticamente
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SOP - Criterio Ecográfico API",
    description="Backend para clasificación de imágenes de ovario poliquístico",
    version="1.0.0"
)

# CORS para conectar con React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # puerto por defecto de Vite/React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(criterio3.router)

@app.get("/")
def root():
    return {"mensaje": "API SOP funcionando ✅"}