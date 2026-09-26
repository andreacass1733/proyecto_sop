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
from services import security_service



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
    Prepara cada nueva imagen subida por el médico para EfficientNet-B0:

    1. Carga la imagen RGB redimensionada a 224×224 px (idéntico a cómo fue entrenado el modelo).
    2. Convierte a array numpy (224, 224, 3).
    3. Agrega dimensión de batch (1, 224, 224, 3).
    4. Aplica preprocess_input de EfficientNet.

    IMPORTANTE: No aplicar CLAHE ni filtros bilaterales aquí, ya que alteran la distribución
    de intensidades de los píxeles y distorsionan la clasificación de la red neuronal.
    """
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


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
    Estima el número de folículos antrales visibles usando segmentación anecoica adaptativa.

    Los folículos en ultrasonido son bolsas de líquido anecoicas (oscuras/negras) dentro del ovario.
    Este algoritmo:
      1. Define un ROI para ignorar bordes con texto/parámetros y zonas fuera del abanico ecográfico.
      2. Suaviza la imagen con Filtro Bilateral para eliminar ruido speckle.
      3. Aplica CLAHE y umbralización adaptativa dentro de la región ovárica.
      4. Filtra por área (20-450 px²), circularidad (≥ 0.35) y oscuridad anecoica estricta.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return 0

    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gris = cv2.resize(img_gris, (224, 224))
    h, w = img_gris.shape

    # 1. Crear máscara ROI para ignorar bordes (textos de cabecera, datos del ecógrafo y márgenes externos)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.12):int(h * 0.90), int(w * 0.08):int(w * 0.92)] = 255

    # Mediana de intensidad del tejido dentro del ROI (excluyendo fondo negro absoluto < 15 y brillo de rejilla > 235)
    valid_pixels = img_gris[(roi_mask > 0) & (img_gris > 15) & (img_gris < 235)]
    if len(valid_pixels) == 0:
        return 0

    median_roi = np.median(valid_pixels)

    # 2. Reducción de ruido Speckle con Filtro Bilateral
    denoised = cv2.bilateralFilter(img_gris, d=7, sigmaColor=75, sigmaSpace=75)

    # 3. Realce de contraste CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Umbralización adaptativa
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=6
    )

    # Aplicar ROI
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    foliculos_validos = 0

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        # Área típica de folículo antral en imagen 224x224: entre 20 px² y 450 px²
        if 20 <= area <= 450:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue

            circularidad = (4.0 * np.pi * area) / (perimetro ** 2)
            # Folículos tienen bordes suaves/redondeados (circularidad >= 0.35)
            if circularidad >= 0.35:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = float(w_box) / h_box if h_box > 0 else 0

                if 0.45 <= aspect_ratio <= 2.2:
                    # Comprobar intensidad interna anecoica (líquido folicular oscuro)
                    mask_cnt = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                    mean_val = cv2.mean(img_gris, mask=mask_cnt)[0]

                    if mean_val < min(65.0, median_roi * 0.70):
                        foliculos_validos += 1

    return foliculos_validos


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

    # Calcular hash de integridad SHA-256 según normativa HIPAA
    sha256_hash = security_service.calcular_hash_sha256(archivo_bytes)
    security_service.auditar_acceso_medico("MEDICO_ACTIVO", "SUBIDA_ECOGRAFIA_MEDICA", f"SHA256:{sha256_hash}")

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
# BASE DE DATOS — CONSULTAS AUTOMÁTICAS Y TABLA estudio_ecografico
# ══════════════════════════════════════════════════════════════════════════════

def obtener_o_crear_consulta_automatica(
    paciente_id: Optional[str] = None,
    consulta_id: Optional[str] = None
) -> str:
    """
    Garantiza que exista una consulta médica real en Supabase para vincular al estudio ecográfico.

    1. Si se proporciona consulta_id y existe en la BD, la devuelve.
    2. Si consulta_id no existe o no se envió, busca la paciente indicada (o paciente activa por defecto).
    3. Cuenta las consultas existentes de la paciente para generar un título numerado:
       'Consulta Ecográfica #N — Evaluación Criterio 3 de Rotterdam'
    4. Crea automáticamente el registro en la tabla 'consulta' y devuelve el UUID real.
    """
    if consulta_id and consulta_id != "None" and consulta_id.strip() != "":
        try:
            res = supabase.table("consulta").select("id").eq("id", consulta_id).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]["id"]
        except Exception as e_check:
            print("[WARN] Error verificando consulta_id previa:", e_check)

    # Si no se pasó paciente_id, buscar la primera paciente activa registrada en el sistema
    if not paciente_id or paciente_id == "None" or paciente_id.strip() == "":
        try:
            res_pac = supabase.table("paciente").select("id").eq("activo", True).limit(1).execute()
            if res_pac.data and len(res_pac.data) > 0:
                paciente_id = res_pac.data[0]["id"]
        except Exception as e_pac:
            print("[WARN] Error buscando paciente por defecto:", e_pac)

    if not paciente_id:
        raise Exception("Se requiere una paciente registrada en el sistema para asociar la consulta ecográfica.")

    # Contar consultas previas registradas para esa paciente
    consultas_previas = supabase.table("consulta").select("id").eq("paciente_id", paciente_id).execute().data or []
    num_consulta = len(consultas_previas) + 1

    motivo = f"Consulta Ecográfica #{num_consulta} — Evaluación Criterio 3 de Rotterdam"
    observaciones = f"Consulta N° {num_consulta} generada automáticamente al procesar imagen ecográfica de ovario."

    from schemas.consulta_schema import ConsultaCreate
    from services.consulta_service import crear_consulta

    nueva_consulta = crear_consulta(ConsultaCreate(
        paciente_id=paciente_id,
        motivo=motivo,
        observaciones=observaciones,
        estado="en_proceso"
    ))

    return nueva_consulta["id"]


def guardar_estudio(
    consulta_id:       Optional[str],   # Opcional para pruebas sin BD completa
    imagen_url:        str,
    imagen_nombre:     str,
    prob_sop:          float,
    prob_normal:       float,
    resultado:         str,
    num_foliculos:     Optional[int] = None,
    num_foliculos_izq: Optional[int] = None,
    num_foliculos_der: Optional[int] = None,
    lado_ovario:       Optional[str] = "izquierdo",
    mapa_calor_url:    Optional[str] = None,
) -> dict:
    """
    Inserta un nuevo registro en la tabla estudio_ecografico de Supabase.

    Guarda el resultado completo del análisis con todos sus 16 campos:
    probabilidades del modelo, clasificación (cumple/no cumple criterio),
    conteo de folículos estimado, mapa de calor URL y versión del modelo.
    """
    datos = {
        "imagen_url":     imagen_url,
        "imagen_nombre":  imagen_nombre,
        "prob_sop":       round(prob_sop, 5),    # Guardar con 5 decimales de precisión
        "prob_normal":    round(prob_normal, 5),
        "resultado":      resultado,
        "validado":       False,                  # Pendiente de validación médica
        "version_modelo": VERSION_MODELO,
        "num_foliculos":  num_foliculos if num_foliculos is not None else 0,
        "mapa_calor_url": mapa_calor_url,
    }

    # Solo incluir consulta_id si se proporciona (permite pruebas sin FK completa)
    if consulta_id and consulta_id != "None" and consulta_id.strip() != "":
        datos["consulta_id"] = consulta_id

    # Asignación diferenciada entre Ovario Izquierdo y Derecho si los campos existen
    if num_foliculos_izq is not None:
        datos["num_foliculos_izq"] = num_foliculos_izq
    elif num_foliculos is not None and lado_ovario == "izquierdo":
        datos["num_foliculos_izq"] = num_foliculos

    if num_foliculos_der is not None:
        datos["num_foliculos_der"] = num_foliculos_der
    elif num_foliculos is not None and lado_ovario == "derecho":
        datos["num_foliculos_der"] = num_foliculos

    try:
        response = supabase.table("estudio_ecografico").insert(datos).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
    except Exception as e:
        print("[WARN] Error insertando en tabla estudio_ecografico, intentando sin campos opcionales num_foliculos_izq/der:", e)
        # Fallback si num_foliculos_izq/der no están en la tabla
        datos.pop("num_foliculos_izq", None)
        datos.pop("num_foliculos_der", None)
        response = supabase.table("estudio_ecografico").insert(datos).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]

    return datos


def validar_estudio(
    estudio_id:    str,
    etiqueta_real: str,
    observacion:   Optional[str] = None,
    paciente_id:   Optional[str] = None,
    consulta_id:   Optional[str] = None,
    imagen_url:    Optional[str] = None,
    imagen_nombre: Optional[str] = None,
    prob_sop:      Optional[float] = None,
    prob_normal:   Optional[float] = None,
    resultado:     Optional[str] = None,
    num_foliculos: Optional[int] = None,
    mapa_calor_url: Optional[str] = None,
    lado_ovario:   Optional[str] = "izquierdo",
) -> dict:
    """
    Registra la evaluación médica definitiva en la base de datos de Supabase.

    Solamente en este punto (cuando el médico evalúa y confirma el estudio):
      1. Se crea la consulta médica real en la tabla 'consulta'.
      2. Se inserta el registro oficial en la tabla 'estudio_ecografico'.
    """
    # 1. Garantizar consulta médica real al momento de la evaluación
    real_consulta_id = None
    try:
        real_consulta_id = obtener_o_crear_consulta_automatica(
            paciente_id=paciente_id,
            consulta_id=consulta_id
        )
    except Exception as e_cons:
        print("[WARN] Error al obtener o crear consulta médica:", e_cons)

    # 2. Intentar actualizar si ya existía en BD por id
    datos_update = {
        "validado":           True,
        "etiqueta_real":      etiqueta_real,
        "observacion_medico": observacion,
        "updated_at":         datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = (
            supabase.table("estudio_ecografico")
            .update(datos_update)
            .eq("id", estudio_id)
            .execute()
        )
        if response.data and len(response.data) > 0:
            return response.data[0]
    except Exception:
        pass

    # 3. Si no existía en BD (porque fue vista previa preliminar), INSERTAR oficialmente en Supabase:
    datos_insert = {
        "imagen_url":     imagen_url or "",
        "imagen_nombre":  imagen_nombre or "",
        "prob_sop":       round(prob_sop, 5) if prob_sop is not None else 0.0,
        "prob_normal":    round(prob_normal, 5) if prob_normal is not None else 0.0,
        "resultado":      resultado or ("Cumple criterio" if etiqueta_real == "SOP" else "No cumple criterio"),
        "validado":       True,
        "etiqueta_real":  etiqueta_real,
        "observacion_medico": observacion,
        "version_modelo": VERSION_MODELO,
        "num_foliculos":  num_foliculos if num_foliculos is not None else 0,
        "mapa_calor_url": mapa_calor_url,
    }
    if real_consulta_id:
        datos_insert["consulta_id"] = real_consulta_id

    try:
        response = supabase.table("estudio_ecografico").insert(datos_insert).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
    except Exception as e_ins:
        print("[WARN] Error al insertar estudio ecográfico evaluado:", e_ins)

    return datos_insert


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