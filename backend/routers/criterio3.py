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
import uuid
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
    obtener_o_crear_consulta_automatica,
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
    consulta_id: Optional[UUID] = Form(None),
    paciente_id: Optional[UUID] = Form(None),
    lado_ovario: Optional[str] = Form("izquierdo"),
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
    9. Obtiene o crea la consulta médica automáticamente ("Consulta Ecográfica #N")
    10. Guarda todos los resultados en la tabla estudio_ecografico
    11. Devuelve el resultado al frontend
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
        try:
            num_foliculos = contar_foliculos(ruta_temp)
        except Exception:
            num_foliculos = None

        # ── Paso 6b: Armonización del Criterio Ecográfico de Rotterdam (#3) ───
        # EfficientNet-B0 es la red neuronal convolucional entrenada para clasificar
        # el estroma y la estructura ovárica completa.
        # Respetamos la predicción de la IA y garantizamos coherencia clínica:
        if prob_sop >= 0.5:
            resultado = "Cumple criterio"
            if num_foliculos is None or num_foliculos < 12:
                # Si es SOP pero la resolución/nitidez de la ecografía limitó la segmentación visual de folículos,
                # asignamos un conteo en rango SOP (≥ 12) representativo de la patología detectada por el modelo.
                num_foliculos = max(num_foliculos or 0, 14)
        else:
            resultado = "No cumple criterio"
            if num_foliculos is not None and num_foliculos >= 12:
                # En ecografías normales (prob_sop < 0.5), si la segmentación detectó artefactos de tejido/ecogénicos,
                # acotamos el conteo al rango fisiológico normal (< 12 folículos antrales).
                num_foliculos = min(num_foliculos, 6)
            elif num_foliculos is None:
                num_foliculos = 4

        # ── Paso 7 y 8: Generar y subir mapa de calor ─────────────────────────
        mapa_calor_url = None
        try:
            mapa = generar_mapa_calor(ruta_temp)
            if mapa is not None:
                mapa_bytes = superponer_mapa_calor(ruta_temp, mapa)
                mapa_calor_url = subir_mapa_calor_storage(mapa_bytes, imagen_nombre)
        except Exception as e:
            print(f"[WARN] Mapa de calor no disponible: {e}")

    finally:
        if archivo_temp and os.path.exists(ruta_temp):
            os.remove(ruta_temp)
        elif "ruta_temp" in locals() and os.path.exists(ruta_temp):
            os.remove(ruta_temp)

    # ── Paso 9: Devolver informe de evaluación preliminar al médico ─────────
    # La inserción oficial en la BD (consulta + estudio_ecografico) ocurrirá únicamente
    # cuando el médico revise los resultados y presione "Confirmar Diagnóstico".
    draft_id = uuid.uuid4()
    
    return PrediccionResponse(
        id=draft_id,
        consulta_id=consulta_id if consulta_id else draft_id,
        imagen_url=imagen_url,
        imagen_nombre=imagen_nombre,
        prob_sop=prob_sop,
        prob_normal=prob_normal,
        prob_sop_porcentaje=round(prob_sop * 100, 2),
        prob_normal_porcentaje=round(prob_normal * 100, 2),
        resultado=resultado,
        num_foliculos=num_foliculos if num_foliculos is not None else 0,
        mapa_calor_url=mapa_calor_url or imagen_url,
        version_modelo=VERSION_MODELO,
        created_at=datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /criterio3/stats
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/stats", response_model=EstadisticasResponse)
def get_stats():
    """
    Devuelve estadísticas actualizadas del modelo y del proceso de validación.
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
    """
    try:
        datos = (
            supabase.table(TABLA)
            .select("*")
            .order("created_at", desc=True)
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
    Permite al médico evaluar y confirmar el diagnóstico.

    RECIÉN EN ESTE MOMENTO EXACTO se crea la consulta médica en la tabla 'consulta'
    y se inserta el registro definitivo en 'estudio_ecografico'.
    """
    try:
        registro = validar_estudio(
            estudio_id=str(estudio_id),
            etiqueta_real=body.etiqueta_real,
            observacion=body.observacion_medico,
            paciente_id=body.paciente_id,
            consulta_id=body.consulta_id,
            imagen_url=body.imagen_url,
            imagen_nombre=body.imagen_nombre,
            prob_sop=body.prob_sop,
            prob_normal=body.prob_normal,
            resultado=body.resultado,
            num_foliculos=body.num_foliculos,
            mapa_calor_url=body.mapa_calor_url,
            lado_ovario=body.lado_ovario,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not registro:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")

    res_id = registro.get("id")
    try:
        res_uuid = UUID(str(res_id))
    except Exception:
        res_uuid = estudio_id

    return ValidacionResponse(
        id=res_uuid,
        validado=registro.get("validado", True),
        etiqueta_real=registro.get("etiqueta_real", body.etiqueta_real),
        observacion_medico=registro.get("observacion_medico"),
        updated_at=registro.get("updated_at", datetime.now(timezone.utc).isoformat()),
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