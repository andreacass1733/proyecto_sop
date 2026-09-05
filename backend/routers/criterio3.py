"""
Router del Criterio 3 — Criterio ecográfico (imágenes de ovario).

Este archivo solo define las rutas HTTP y coordina el flujo de datos.
Toda la lógica de IA, almacenamiento y base de datos vive en:
    services/criterio3_service.py

Rutas disponibles:
    POST   /criterio3/predecir         → Analiza una ecografía con IA
    GET    /criterio3/stats            → Estadísticas del modelo
    GET    /criterio3/estudios         → Lista todos los estudios
    GET    /criterio3/estudios/{id}    → Detalle de un estudio
    PUT    /criterio3/validar/{id}     → El médico valida el resultado
    POST   /criterio3/reentrenar       → Inicia reentrenamiento del modelo
"""

import os
import tempfile
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from schemas.criterio3_schema import (
    EstadisticasResponse,
    EstudioEcograficoOut,
    PrediccionResponse,
    ValidacionResponse,
    ValidarRequest,
)
from services.criterio3_service import (
    contar_foliculos,
    generar_mapa_calor,
    guardar_estudio,
    obtener_estadisticas,
    run_prediction,
    subir_imagen_storage,
    subir_mapa_calor_storage,
    superponer_mapa_calor,
    validar_estudio,
    UMBRAL_REENTRENAMIENTO,
    VERSION_MODELO,
)
from db.supabase_client import supabase

# Nombre del router y prefijo de URL para todas las rutas de este criterio
router = APIRouter(prefix="/criterio3", tags=["Criterio 3 — Ecografía"])

# Tabla en Supabase que almacena los estudios ecográficos
TABLA = "estudio_ecografico"

# Solo se aceptan imágenes en formato JPG o PNG
FORMATOS_PERMITIDOS = {"image/jpeg", "image/png"}


# ══════════════════════════════════════════════════════════════════════════════
# POST /criterio3/predecir
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/predecir", response_model=PrediccionResponse)
async def predecir(
    file: UploadFile = File(...),
    consulta_id: Optional[UUID] = Form(None),  # Opcional para pruebas sin BD completa
):
    """
    Recibe una imagen de ecografía ovárica y realiza el análisis completo:

    1. Valida que el archivo sea JPG o PNG
    2. Lee los bytes de la imagen
    3. Sube la imagen original a Supabase Storage
    4. Guarda la imagen temporalmente en disco para procesarla con IA
    5. Predice si cumple el criterio ecográfico de Rotterdam (EfficientNet-B0)
    6. Estima el número de folículos visibles (OpenCV HoughCircles)
    7. Genera un mapa de calor para visualizar zonas de activación
    8. Sube el mapa de calor a Supabase Storage
    9. Guarda todos los resultados en la tabla estudio_ecografico
    10. Devuelve el resultado al frontend

    El archivo temporal se elimina siempre al finalizar (paso 10), incluso si hay errores.
    """

    # ── Paso 1: Validar formato del archivo ───────────────────────────────────
    if file.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no permitido: '{file.content_type}'. Use JPG o PNG.",
        )

    # ── Paso 2: Leer todos los bytes de la imagen en memoria ──────────────────
    contenido = await file.read()

    # ── Paso 3: Subir ecografía original a Supabase Storage ───────────────────
    try:
        imagen_url, imagen_nombre = subir_imagen_storage(contenido, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al subir la imagen a Storage: {str(e)}"
        )

    # ── Paso 4: Crear archivo temporal en disco para procesar con OpenCV/TF ───
    # Se necesita una ruta de archivo porque cv2.imread y keras requieren ruta, no bytes
    extension = file.filename.split(".")[-1].lower()
    archivo_temp = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=f".{extension}", delete=False
        ) as tmp:
            tmp.write(contenido)
            ruta_temp = tmp.name

        # ── Paso 5: Predicción con EfficientNet-B0 ────────────────────────────
        try:
            prob_sop, prob_normal, resultado = run_prediction(ruta_temp)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error en la predicción del modelo: {str(e)}"
            )

        # ── Paso 6: Estimar número de folículos con OpenCV ────────────────────
        # Si falla, se guarda None y el médico puede completarlo manualmente
        try:
            num_foliculos = contar_foliculos(ruta_temp)
        except Exception:
            num_foliculos = None  # No bloquear el flujo si el conteo falla

        # ── Paso 7 y 8: Generar y subir mapa de calor ─────────────────────────
        mapa_calor_url = None
        try:
            mapa = generar_mapa_calor(ruta_temp)
            if mapa is not None:
                # Superponer el mapa sobre la imagen original
                mapa_bytes = superponer_mapa_calor(ruta_temp, mapa)
                # Subir el mapa de calor a la subcarpeta 'mapas_calor/' del bucket
                mapa_calor_url = subir_mapa_calor_storage(mapa_bytes, imagen_nombre)
        except Exception as e:
            print(f"[WARN] Mapa de calor no disponible: {e}")
            # El mapa de calor es opcional, no se detiene el proceso si falla

    finally:
        # ── Paso 10: Eliminar el archivo temporal del disco siempre ───────────
        # Esto garantiza que no queden archivos huérfanos aunque ocurra un error
        if archivo_temp and os.path.exists(ruta_temp):
            os.remove(ruta_temp)
        elif "ruta_temp" in locals() and os.path.exists(ruta_temp):
            os.remove(ruta_temp)

    # ── Paso 9: Guardar resultado completo en la base de datos ────────────────
    registro = None
    try:
        registro = guardar_estudio(
            consulta_id=str(consulta_id) if consulta_id else None,
            imagen_url=imagen_url,
            imagen_nombre=imagen_nombre,
            prob_sop=prob_sop,
            prob_normal=prob_normal,
            resultado=resultado,
            num_foliculos=num_foliculos,
        )
    except Exception as e:
        print(f"[WARN] No se guardo en BD (modo prueba sin consulta_id): {e}")

    DUMMY_UUID = UUID("00000000-0000-0000-0000-000000000000")
    
    # Construir y devolver la respuesta al frontend
    return PrediccionResponse(
        id=(registro.get("id") if registro and registro.get("id") else DUMMY_UUID),
        consulta_id=(registro.get("consulta_id") if registro and registro.get("consulta_id") else DUMMY_UUID),
        imagen_url=imagen_url,
        imagen_nombre=imagen_nombre,
        prob_sop=prob_sop,
        prob_normal=prob_normal,
        prob_sop_porcentaje=round(prob_sop * 100, 2),
        prob_normal_porcentaje=round(prob_normal * 100, 2),
        resultado=resultado,
        num_foliculos=num_foliculos if num_foliculos is not None else 0,
        mapa_calor_url=mapa_calor_url or imagen_url,
        version_modelo=registro["version_modelo"] if (registro and "version_modelo" in registro) else VERSION_MODELO,
        created_at=registro["created_at"] if (registro and "created_at" in registro) else datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /criterio3/stats
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/stats", response_model=EstadisticasResponse)
def get_stats():
    """
    Devuelve estadísticas actualizadas del modelo y del proceso de validación:
      - Total de estudios realizados
      - Cuántos fueron validados por el médico
      - Precisión real calculada sobre los estudios validados
      - Si ya hay suficientes validaciones para lanzar un reentrenamiento
    """
    try:
        return obtener_estadisticas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# GET /criterio3/estudios
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/estudios", response_model=list[EstudioEcograficoOut])
def listar_estudios():
    """
    Lista todos los estudios ecográficos registrados en el sistema,
    ordenados del más reciente al más antiguo.

    Útil para el historial de análisis del médico.
    """
    try:
        datos = (
            supabase.table(TABLA)
            .select("*")
            .order("created_at", desc=True)  # Más recientes primero
            .execute()
            .data
        )
        return datos or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# GET /criterio3/estudios/{estudio_id}
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/estudios/{estudio_id}", response_model=EstudioEcograficoOut)
def obtener_estudio(estudio_id: UUID):
    """
    Devuelve el detalle completo de un estudio ecográfico específico.
    Incluye la imagen, probabilidades, resultado, folículos y estado de validación.
    """
    datos = (
        supabase.table(TABLA)
        .select("*")
        .eq("id", str(estudio_id))
        .execute()
        .data
    )

    if not datos:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    return datos[0]


# ══════════════════════════════════════════════════════════════════════════════
# PUT /criterio3/validar/{estudio_id}
# ══════════════════════════════════════════════════════════════════════════════

@router.put("/validar/{estudio_id}", response_model=ValidacionResponse)
def validar(estudio_id: UUID, body: ValidarRequest):
    """
    Permite al médico validar (confirmar o corregir) el resultado del modelo.

    El médico indica si la imagen es realmente 'SOP' o 'Normal'.
    Esta información se acumula para reentrenar el modelo con datos reales
    del contexto clínico local.

    También puede agregar observaciones de texto (opcional).
    """
    try:
        registro = validar_estudio(
            estudio_id=str(estudio_id),
            etiqueta_real=body.etiqueta_real,
            observacion=body.observacion_medico,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not registro:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    return ValidacionResponse(
        id=registro["id"],
        validado=registro["validado"],
        etiqueta_real=registro["etiqueta_real"],
        observacion_medico=registro.get("observacion_medico"),
        updated_at=registro["updated_at"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /criterio3/reentrenar
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/reentrenar")
def reentrenar():
    """
    Inicia el proceso de reentrenamiento del modelo EfficientNet-B0.

    Condición para reentrenar: debe haber al menos 50 imágenes
    validadas por el médico (configurado en UMBRAL_REENTRENAMIENTO).

    El reentrenamiento se lanza en segundo plano (subprocess) para no
    bloquear la API mientras dura el proceso (puede tardar horas).
    El script de entrenamiento usa las imágenes del dataset local,
    no las de Supabase Storage.
    """
    import subprocess

    # Consultar cuántos estudios ya fueron validados por el médico
    datos = (
        supabase.table(TABLA)
        .select("id")
        .eq("validado", True)
        .execute()
        .data
    )
    total_validadas = len(datos) if datos else 0

    # Verificar que se alcanzó el umbral mínimo para reentrenar
    if total_validadas < UMBRAL_REENTRENAMIENTO:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Solo hay {total_validadas} imágenes validadas. "
                f"Se necesitan al menos {UMBRAL_REENTRENAMIENTO} para reentrenar."
            ),
        )

    # Lanzar el script de entrenamiento en un proceso separado (en segundo plano)
    # Popen no bloquea: la API responde inmediatamente y el entrenamiento corre aparte
    subprocess.Popen(["python", "training/entrenar_modelo.py"])

    return {
        "mensaje":        f"Reentrenamiento iniciado con {total_validadas} imágenes ✅",
        "total_validadas": total_validadas,
    }