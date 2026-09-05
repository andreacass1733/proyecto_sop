"""
Servicio del Criterio 3 — Criterio ecográfico (imágenes de ovario).

Responsabilidades de este módulo:
  - Cargar y reutilizar el modelo EfficientNet-B0 en memoria
  - Predecir si una ecografía muestra morfología ovárica poliquística
  - Generar mapas de calor (saliency map) para visualizar las zonas de interés
  - Estimar el número de folículos visibles usando OpenCV
  - Subir imágenes y mapas de calor a Supabase Storage
  - Insertar, actualizar y consultar registros en la tabla estudio_ecografico
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from db.supabase_client import supabase


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GENERAL
# ══════════════════════════════════════════════════════════════════════════════

# Rutas donde se buscará el modelo entrenado (en orden de prioridad)
_RUTAS_MODELO = ["models_ia/best_fase1.keras", "models/best_fase1.keras"]

# Etiquetas de las dos clases que puede predecir el modelo
CLASES = ["Normal", "SOP"]

# Versión del modelo actualmente en producción (se guarda en cada registro)
VERSION_MODELO = "v1.0"

# Nombre del bucket de Supabase Storage donde se almacenan las ecografías
BUCKET = "ecografias"

# Cantidad mínima de imágenes validadas por el médico para poder reentrenar
UMBRAL_REENTRENAMIENTO = 50

# Variable global que almacena el modelo ya cargado (evita recargarlo en cada petición)
_model = None


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DEL MODELO (SINGLETON)
# ══════════════════════════════════════════════════════════════════════════════

def _resolver_ruta_modelo() -> str:
    """
    Busca el archivo .keras del modelo en las rutas configuradas.
    Lanza un error si no se encuentra en ninguna de ellas.
    """
    for ruta in _RUTAS_MODELO:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(
        f"No se encontró 'best_fase1.keras'. Rutas buscadas: {_RUTAS_MODELO}"
    )


def get_model() -> tf.keras.Model:
    """
    Devuelve el modelo EfficientNet-B0 listo para predecir.

    Usa el patrón singleton: carga el modelo del disco solo la primera vez
    que se llama; en llamadas posteriores reutiliza el mismo objeto en memoria.
    Esto evita el costo de leer un archivo de ~25 MB en cada petición.
    """
    global _model
    if _model is None:
        ruta = _resolver_ruta_modelo()
        _model = tf.keras.models.load_model(ruta)
        print(f"[OK] Modelo EfficientNet-B0 cargado desde: {ruta}")
    return _model


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO DE IMAGEN
# ══════════════════════════════════════════════════════════════════════════════

def _preprocesar_imagen(img_path: str) -> np.ndarray:
    """
    Prepara una imagen para que el modelo pueda procesarla:

    1. Carga la imagen desde disco con PIL (soporta JPG y PNG)
    2. Redimensiona a 224×224 px (tamaño fijo que espera EfficientNet-B0)
    3. Convierte a array numérico de numpy
    4. Agrega la dimensión de batch: (224, 224, 3) → (1, 224, 224, 3)
    5. Aplica preprocess_input de EfficientNet:
       resta la media de ImageNet por canal y normaliza los valores

    Retorna un array listo para pasarle directamente al modelo.
    """
    img = keras_image.load_img(img_path, target_size=(224, 224))  # Carga y redimensiona
    img_array = keras_image.img_to_array(img)                      # PIL → numpy float32
    img_array = np.expand_dims(img_array, axis=0)                  # Agrega dimensión de batch
    img_array = preprocess_input(img_array)                        # Normalización EfficientNet
    return img_array


# ══════════════════════════════════════════════════════════════════════════════
# PREDICCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_prediction(img_path: str) -> tuple[float, float, str]:
    """
    Analiza una ecografía ovárica con el modelo EfficientNet-B0 y determina
    si muestra morfología de ovario poliquístico (criterio ecográfico de Rotterdam).

    Proceso:
      1. Preprocesa la imagen (224×224, normalización)
      2. Pasa la imagen por el modelo → obtiene [prob_normal, prob_sop]
      3. Si prob_sop ≥ 0.5 → "Cumple criterio"; de lo contrario → "No cumple criterio"

    Retorna:
        (prob_sop, prob_normal, resultado)
        Ejemplo: (0.87, 0.13, "Cumple criterio")
    """
    modelo = get_model()
    img_array = _preprocesar_imagen(img_path)

    # El modelo devuelve una fila por imagen: [prob_clase_0, prob_clase_1]
    # Clase 0 = Normal, Clase 1 = SOP (según el orden de CLASES)
    pred = modelo.predict(img_array, verbose=0)[0]
    prob_normal = float(pred[0])
    prob_sop    = float(pred[1])

    # Umbral de decisión: 50%
    resultado = "Cumple criterio" if prob_sop >= 0.5 else "No cumple criterio"

    return prob_sop, prob_normal, resultado


# ══════════════════════════════════════════════════════════════════════════════
# MAPA DE CALOR — SALIENCY MAP (VISUALIZACIÓN DE ZONAS ACTIVAS)
# ══════════════════════════════════════════════════════════════════════════════

def generar_mapa_calor(img_path: str, clase_idx: int = 1) -> Optional[np.ndarray]:
    """
    Genera un mapa de calor que indica qué zonas de la ecografía
    influyeron más en la predicción del modelo.

    Técnica usada: Saliency Map por gradientes
    ──────────────────────────────────────────
    Se calcula el gradiente de la puntuación de la clase objetivo (SOP)
    respecto a cada píxel de la imagen de entrada.
    Un píxel con gradiente alto significa que cambiar su valor
    afectaría mucho la predicción → el modelo lo considera importante.

    En ecografías ováricas, las zonas con mayor activación suelen
    corresponder a los folículos (estructuras circulares oscuras),
    que son el patrón que el modelo aprendió a reconocer.

    Args:
        img_path  : Ruta local de la imagen a analizar
        clase_idx : Clase objetivo (1 = SOP por defecto, 0 = Normal)

    Retorna:
        ndarray 2D de forma (224, 224) con valores entre 0.0 y 1.0,
        donde 1.0 = zona de máxima activación.
        Devuelve None si ocurre algún error.
    """
    try:
        modelo = get_model()

        # Preprocesar la imagen y convertirla a Variable de TensorFlow
        # GradientTape puede calcular derivadas respecto a tf.Variable automáticamente
        img_procesada = _preprocesar_imagen(img_path)
        img_var = tf.Variable(img_procesada, dtype=tf.float32)

        # Registrar las operaciones del forward pass dentro del contexto del tape
        with tf.GradientTape() as tape:
            predicciones = modelo(img_var, training=False)
            # Puntuación de la clase objetivo (SOP = índice 1)
            puntuacion = predicciones[:, clase_idx]

        # Derivada de la puntuación respecto a cada píxel de la imagen de entrada
        # Forma del gradiente: (1, 224, 224, 3) — igual que la imagen
        gradientes = tape.gradient(puntuacion, img_var)

        if gradientes is None:
            print("[WARN] No se pudo calcular el gradiente (el tape no registro operaciones)")
            return None

        # Promediar los gradientes de los 3 canales RGB → mapa 2D (224, 224)
        # Se usa valor absoluto porque importa la magnitud, no la dirección del cambio
        mapa = tf.reduce_mean(tf.abs(gradientes[0]), axis=-1)

        # Normalizar el mapa entre 0 y 1 para visualización
        valor_maximo = tf.reduce_max(mapa)
        if valor_maximo > 0:
            mapa = mapa / valor_maximo

        return mapa.numpy()

    except Exception as e:
        print(f"[WARN] Error generando mapa de calor: {e}")
        return None


def superponer_mapa_calor(img_path: str, mapa: np.ndarray) -> bytes:
    """
    Combina la ecografía original con el mapa de calor generado
    para crear una imagen de visualización para el médico.

    Paleta de colores usada (COLORMAP_JET):
        Azul  → zona de baja activación (poca influencia)
        Verde → zona de activación media
        Rojo  → zona de alta activación (posibles folículos)

    La imagen original (60%) se mezcla con el mapa de calor (40%).

    Retorna:
        bytes JPEG de la imagen combinada, lista para subir a Storage
    """
    # Cargar imagen original en formato BGR (OpenCV usa BGR, no RGB)
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (224, 224))

    # Redimensionar el mapa de calor al mismo tamaño que la imagen
    mapa_resized = cv2.resize(mapa, (224, 224))

    # Convertir el mapa normalizado (0.0–1.0) a escala de 0–255 para OpenCV
    mapa_uint8 = np.uint8(255 * mapa_resized)

    # Aplicar mapa de colores tipo 'jet' (azul → verde → rojo)
    mapa_color = cv2.applyColorMap(mapa_uint8, cv2.COLORMAP_JET)

    # Mezclar imagen original (60%) con el mapa de calor coloreado (40%)
    imagen_final = cv2.addWeighted(img_bgr, 0.6, mapa_color, 0.4, 0)

    # Codificar la imagen resultante como bytes en formato JPEG
    _, buffer = cv2.imencode(".jpg", imagen_final)
    return buffer.tobytes()


# ══════════════════════════════════════════════════════════════════════════════
# CONTEO DE FOLÍCULOS CON OPENCV
# ══════════════════════════════════════════════════════════════════════════════

def contar_foliculos(img_path: str) -> int:
    """
    Estima el número de folículos visibles en una ecografía ovárica.

    Técnica usada: Transformada de Hough para círculos (HoughCircles)
    ─────────────────────────────────────────────────────────────────
    Los folículos son estructuras anecoicas (oscuras, casi negras) de
    forma circular en la ecografía. OpenCV puede detectarlos buscando
    patrones circulares en la imagen en escala de grises.

    Parámetros de búsqueda calibrados para ecografías ováricas:
      - Radio mínimo: 5 px  (~2-3 mm reales en imagen 224×224)
      - Radio máximo: 40 px (~12 mm reales)
      - Distancia mínima entre centros: 15 px (evita contar el mismo dos veces)

    IMPORTANTE: Este conteo es una estimación automática. La calidad
    del resultado depende de la calidad y escala de la ecografía.
    El médico puede corregir el valor manualmente en la interfaz.

    Retorna:
        int: Número estimado de folículos (0 si no se detectan o hay error)
    """
    # Leer la imagen en escala de grises (suficiente para detectar círculos)
    img_gris = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gris is None:
        print("[WARN] No se pudo leer la imagen para conteo de foliculos")
        return 0

    # Redimensionar a 224×224 para consistencia con el tamaño del modelo
    img_gris = cv2.resize(img_gris, (224, 224))

    # Aplicar desenfoque gaussiano (kernel 9×9) para reducir el ruido de speckle
    # El ruido de speckle es el patrón granular típico de imágenes de ultrasonido
    # Sin este paso, el algoritmo detectaría muchos círculos falsos
    img_suavizada = cv2.GaussianBlur(img_gris, (9, 9), 2)

    # Detectar círculos con la Transformada de Hough
    # HOUGH_GRADIENT: método basado en gradiente de imagen
    # dp=1.2        : resolución del acumulador (1.0 = misma resolución que la imagen)
    # minDist=15    : distancia mínima entre centros de dos círculos detectados
    # param1=50     : umbral de Canny para detección de bordes internamente
    # param2=30     : umbral del acumulador (menor valor = más detecciones, más falsos positivos)
    # minRadius=5   : radio mínimo en píxeles de un folículo
    # maxRadius=40  : radio máximo en píxeles de un folículo
    circulos = cv2.HoughCircles(
        img_suavizada,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=40
    )

    # HoughCircles devuelve None si no encuentra ningún círculo
    if circulos is not None:
        return len(circulos[0])  # circulos[0] contiene la lista de círculos detectados

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE STORAGE — SUBIDA DE IMÁGENES
# ══════════════════════════════════════════════════════════════════════════════

def subir_imagen_storage(archivo_bytes: bytes, nombre_original: str) -> tuple[str, str]:
    """
    Sube una ecografía al bucket de Supabase Storage.

    - La imagen se guarda en la carpeta 'ecografias/' dentro del bucket
    - Se genera un nombre único con UUID para evitar colisiones entre archivos
    - Se determina el tipo MIME según la extensión del archivo

    Retorna:
        (url_publica, nombre_guardado)
        Ejemplo: ("https://xdd...supabase.co/storage/.../uuid.jpg", "uuid.jpg")
    """
    # Extraer la extensión del nombre original (jpg, jpeg, png)
    extension = nombre_original.split(".")[-1].lower()

    # Generar un nombre único usando UUID para evitar sobreescribir archivos
    nombre_guardado = f"{uuid.uuid4()}.{extension}"
    ruta_storage = f"ecografias/{nombre_guardado}"

    # Mapear la extensión al tipo MIME que necesita Supabase Storage
    mime_types = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
    }
    content_type = mime_types.get(extension, "image/jpeg")

    # Subir los bytes del archivo al bucket configurado
    supabase.storage.from_(BUCKET).upload(
        ruta_storage,
        archivo_bytes,
        {"content-type": content_type}
    )

    # Obtener la URL pública del archivo recién subido
    url_publica = supabase.storage.from_(BUCKET).get_public_url(ruta_storage)
    return url_publica, nombre_guardado


def subir_mapa_calor_storage(mapa_bytes: bytes, nombre_ecografia: str) -> str:
    """
    Sube el mapa de calor generado al bucket de Supabase Storage.

    Se guarda en la subcarpeta 'mapas_calor/' con el sufijo '_mapa_calor'
    para diferenciarlo de la ecografía original del mismo estudio.

    Args:
        mapa_bytes       : Bytes JPEG del mapa de calor ya generado
        nombre_ecografia : Nombre del archivo de la ecografía original (con o sin extensión)

    Retorna:
        str: URL pública del mapa de calor en Storage
    """
    # Construir el nombre del mapa de calor basado en la ecografía original
    nombre_base = nombre_ecografia.rsplit(".", 1)[0]  # Quitar extensión si la tiene
    nombre_mapa = f"{nombre_base}_mapa_calor.jpg"
    ruta_storage = f"mapas_calor/{nombre_mapa}"

    # Subir al mismo bucket pero en la subcarpeta 'mapas_calor'
    supabase.storage.from_(BUCKET).upload(
        ruta_storage,
        mapa_bytes,
        {"content-type": "image/jpeg"}
    )

    url_publica = supabase.storage.from_(BUCKET).get_public_url(ruta_storage)
    return url_publica


# ══════════════════════════════════════════════════════════════════════════════
# BASE DE DATOS — TABLA estudio_ecografico
# ══════════════════════════════════════════════════════════════════════════════

def guardar_estudio(
    consulta_id:   Optional[str],   # Opcional para pruebas sin BD completa
    imagen_url:    str,
    imagen_nombre: str,
    prob_sop:      float,
    prob_normal:   float,
    resultado:     str,
    num_foliculos: Optional[int] = None,
) -> dict:
    """
    Inserta un nuevo registro en la tabla estudio_ecografico de Supabase.

    Guarda el resultado completo del análisis: probabilidades del modelo,
    clasificación (cumple/no cumple criterio), conteo de folículos estimado
    y la versión del modelo usada.

    El campo 'validado' se inicia en False porque el médico debe
    revisar y confirmar (o corregir) el resultado antes de que sirva
    para el reentrenamiento del modelo.

    Retorna:
        dict: El registro recién creado con todos sus campos, incluido el id generado por Supabase
    """
    datos = {
        "imagen_url":     imagen_url,
        "imagen_nombre":  imagen_nombre,
        "prob_sop":       round(prob_sop, 5),    # Guardar con 5 decimales de precisión
        "prob_normal":    round(prob_normal, 5),
        "resultado":      resultado,
        "validado":       False,                  # Pendiente de validación médica
        "version_modelo": VERSION_MODELO,
    }

    # Solo incluir consulta_id si se proporciona (permite pruebas sin FK completa)
    if consulta_id and consulta_id != "None" and consulta_id.strip() != "":
        datos["consulta_id"] = consulta_id

    # Agregar conteo de folículos si se pudo estimar (campo opcional)
    if num_foliculos is not None:
        datos["num_foliculos_izq"] = num_foliculos

    response = supabase.table("estudio_ecografico").insert(datos).execute()
    return response.data[0]


def validar_estudio(
    estudio_id:   str,
    etiqueta_real: str,
    observacion:  str | None,
) -> dict:
    """
    Registra la validación médica de un estudio ecográfico existente.

    Cuando el médico confirma o corrige el resultado del modelo, se actualiza:
      - validado = True
      - etiqueta_real = "SOP" o "Normal" (según el criterio del médico)
      - observacion_medico = notas adicionales del médico (opcional)
      - updated_at = timestamp actual

    Estos datos validados se acumulan para el futuro reentrenamiento del modelo,
    lo que permite que el modelo mejore con el tiempo.

    Retorna:
        dict: El registro actualizado con todos sus campos
    """
    datos = {
        "validado":           True,
        "etiqueta_real":      etiqueta_real,
        "observacion_medico": observacion,
        "updated_at":         datetime.now(timezone.utc).isoformat(),
    }
    response = (
        supabase.table("estudio_ecografico")
        .update(datos)
        .eq("id", estudio_id)
        .execute()
    )
    return response.data[0]


def obtener_estadisticas() -> dict:
    """
    Calcula y devuelve estadísticas globales del Criterio 3 para el dashboard.

    Métricas calculadas:
      - Total de estudios realizados en el sistema
      - Cuántos fueron validados por el médico vs cuántos están pendientes
      - Precisión real del modelo (basada en validaciones humanas)
      - Si ya hay suficientes validaciones para lanzar un reentrenamiento

    La 'precision_entrenamiento' (0.9531) es el valor fijo obtenido
    durante el entrenamiento original del modelo con el dataset base.
    La 'precision_real' se calcula dinámicamente con los estudios validados.

    Retorna:
        dict con todas las métricas para mostrar en el dashboard del médico
    """
    # Obtener todos los registros de la tabla de una sola consulta
    todos = supabase.table("estudio_ecografico").select("*").execute().data

    # Separar los registros según si fueron validados o no por el médico
    validados  = [r for r in todos if r["validado"]]
    pendientes = [r for r in todos if not r["validado"]]

    # Calcular precisión real solo si hay al menos un estudio validado
    precision_real = None
    if validados:
        # Un estudio es "correcto" si la etiqueta del médico coincide con el resultado del modelo
        correctos = sum(
            1 for r in validados
            if (r["etiqueta_real"] == "SOP"    and r["resultado"] == "Cumple criterio")
            or (r["etiqueta_real"] == "Normal" and r["resultado"] == "No cumple criterio")
        )
        precision_real = round(correctos / len(validados), 4)

    # Verificar si se superó el umbral mínimo para reentrenar
    listo_para_reentrenar = len(validados) >= UMBRAL_REENTRENAMIENTO

    return {
        "total_estudios":            len(todos),
        "total_validados":           len(validados),
        "total_pendientes":          len(pendientes),
        "precision_real":            precision_real,
        "precision_entrenamiento":   0.9531,           # Precisión del modelo original
        "imagenes_validadas_nuevas": len(validados),
        "umbral_reentrenamiento":    UMBRAL_REENTRENAMIENTO,
        "listo_para_reentrenar":     listo_para_reentrenar,
    }