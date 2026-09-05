"""
Script de prueba del modelo EfficientNet-B0 — SIN base de datos.

Uso:
    python test_modelo.py ruta/de/imagen.jpg
    python test_modelo.py                        ← usa la primera imagen de test_images/

Qué hace:
    1. Carga el modelo EfficientNet-B0
    2. Predice si la imagen cumple el criterio ecografico de Rotterdam
    3. Estima el numero de foliculos con OpenCV
    4. Genera un mapa de calor y lo guarda en outputs/
    5. Muestra todos los resultados en consola

No necesita Supabase, ni base de datos, ni internet.
"""

import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ══════════════════════════════════════════════════════════════════════════════

# Rutas donde se buscara el modelo entrenado
RUTAS_MODELO = [
    "models_ia/best_fase1.keras",
    "models/best_fase1.keras",
]

# Carpeta de imagenes de prueba por defecto
CARPETA_TEST = "test_images"

# Carpeta donde se guardan los resultados (mapa de calor)
CARPETA_SALIDA = "outputs/test_resultados"

# Umbral de decision: si prob_sop >= 0.5 -> "Cumple criterio"
UMBRAL = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════════════════════

def cargar_modelo() -> tf.keras.Model:
    """Busca y carga el modelo .keras desde las rutas configuradas."""
    for ruta in RUTAS_MODELO:
        if os.path.exists(ruta):
            print(f"[OK] Cargando modelo desde: {ruta}")
            modelo = tf.keras.models.load_model(ruta)
            print("[OK] Modelo cargado correctamente")
            return modelo

    print("[ERROR] No se encontro el modelo en ninguna de estas rutas:")
    for r in RUTAS_MODELO:
        print(f"       - {r}")
    sys.exit(1)


def preprocesar_imagen(img_path: str) -> np.ndarray:
    """
    Carga la imagen, la redimensiona a 224x224 y aplica
    la normalizacion que espera EfficientNet-B0.
    """
    img = keras_image.load_img(img_path, target_size=(224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predecir(modelo: tf.keras.Model, img_path: str) -> tuple:
    """
    Pasa la imagen por el modelo y obtiene las probabilidades.

    Retorna:
        (prob_sop, prob_normal, resultado)
    """
    arr = preprocesar_imagen(img_path)
    pred = modelo.predict(arr, verbose=0)[0]

    prob_normal = float(pred[0])
    prob_sop    = float(pred[1])
    resultado   = "Cumple criterio" if prob_sop >= UMBRAL else "No cumple criterio"

    return prob_sop, prob_normal, resultado


def contar_foliculos(img_path: str) -> int:
    """
    Estima el numero de foliculos en la imagen usando
    la transformada de Hough para deteccion de circulos (OpenCV).

    Los foliculos son estructuras circulares oscuras en las ecografias.
    """
    img_gris = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gris is None:
        print("[WARN] No se pudo leer la imagen con OpenCV")
        return 0

    # Redimensionar a 224x224 para consistencia
    img_gris = cv2.resize(img_gris, (224, 224))

    # Desenfoque gaussiano para reducir ruido de speckle (tipico en ultrasonido)
    img_blur = cv2.GaussianBlur(img_gris, (9, 9), 2)

    # Detectar circulos con HoughCircles
    circulos = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,        # Distancia minima entre foliculos
        param1=50,         # Umbral de Canny
        param2=30,         # Umbral del acumulador
        minRadius=5,       # Radio minimo de un foliculo (px)
        maxRadius=40       # Radio maximo de un foliculo (px)
    )

    if circulos is not None:
        return len(circulos[0])

    return 0


def generar_mapa_calor(modelo: tf.keras.Model, img_path: str, clase_idx: int = 1) -> np.ndarray | None:
    """
    Genera un mapa de calor que muestra que zonas de la imagen
    influyeron mas en la prediccion (saliency map por gradientes).
    """
    try:
        arr = preprocesar_imagen(img_path)
        img_var = tf.Variable(arr, dtype=tf.float32)

        # Calcular gradientes de la puntuacion de la clase objetivo
        # respecto a cada pixel de la imagen de entrada
        with tf.GradientTape() as tape:
            predicciones = modelo(img_var, training=False)
            puntuacion   = predicciones[:, clase_idx]

        gradientes = tape.gradient(puntuacion, img_var)

        if gradientes is None:
            return None

        # Promediar canales RGB -> mapa 2D, usar valor absoluto
        mapa = tf.reduce_mean(tf.abs(gradientes[0]), axis=-1)

        # Normalizar entre 0 y 1
        max_val = tf.reduce_max(mapa)
        if max_val > 0:
            mapa = mapa / max_val

        return mapa.numpy()

    except Exception as e:
        print(f"[WARN] No se pudo generar el mapa de calor: {e}")
        return None


def guardar_mapa_calor(img_path: str, mapa: np.ndarray, nombre_salida: str):
    """
    Superpone el mapa de calor sobre la imagen original y guarda el resultado.
    Colores: azul = poca activacion, rojo = mucha activacion (posibles foliculos).
    """
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Cargar imagen original y redimensionar
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (224, 224))

    # Convertir mapa normalizado a escala de color
    mapa_resized = cv2.resize(mapa, (224, 224))
    mapa_uint8   = np.uint8(255 * mapa_resized)
    mapa_color   = cv2.applyColorMap(mapa_uint8, cv2.COLORMAP_JET)

    # Mezclar: 60% imagen original + 40% mapa de calor
    resultado = cv2.addWeighted(img_bgr, 0.6, mapa_color, 0.4, 0)

    ruta_guardado = os.path.join(CARPETA_SALIDA, nombre_salida)
    cv2.imwrite(ruta_guardado, resultado)
    print(f"[OK] Mapa de calor guardado en: {ruta_guardado}")


def imprimir_resultado(img_path, prob_sop, prob_normal, resultado, num_foliculos):
    """Imprime los resultados de forma clara en la consola."""
    print()
    print("=" * 55)
    print("  RESULTADO DEL ANALISIS - CRITERIO 3 (Rotterdam)")
    print("=" * 55)
    print(f"  Imagen analizada   : {os.path.basename(img_path)}")
    print(f"  Resultado          : {resultado}")
    print(f"  Probabilidad SOP   : {round(prob_sop * 100, 2)}%")
    print(f"  Probabilidad Normal: {round(prob_normal * 100, 2)}%")
    print(f"  Foliculos estimados: {num_foliculos}")
    print("=" * 55)

    if resultado == "Cumple criterio":
        print("  [!] La imagen CUMPLE el criterio ecografico de SOP")
    else:
        print("  [OK] La imagen NO cumple el criterio ecografico de SOP")

    print("=" * 55)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCION PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Determinar que imagen usar
    if len(sys.argv) > 1:
        # El usuario paso una ruta por argumento
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"[ERROR] No se encontro la imagen: {img_path}")
            sys.exit(1)
    else:
        # Buscar la primera imagen en la carpeta test_images/
        if not os.path.exists(CARPETA_TEST):
            print(f"[ERROR] No existe la carpeta '{CARPETA_TEST}/'")
            print("        Coloca una imagen ahi o pasa la ruta como argumento:")
            print("        python test_modelo.py mi_imagen.jpg")
            sys.exit(1)

        imagenes = [
            f for f in os.listdir(CARPETA_TEST)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not imagenes:
            print(f"[ERROR] No hay imagenes JPG/PNG en '{CARPETA_TEST}/'")
            sys.exit(1)

        img_path = os.path.join(CARPETA_TEST, imagenes[0])
        print(f"[INFO] Usando imagen: {img_path}")

    # ── Paso 1: Cargar el modelo ───────────────────────────────────────────────
    modelo = cargar_modelo()

    # ── Paso 2: Predecir ──────────────────────────────────────────────────────
    print("[INFO] Analizando imagen...")
    prob_sop, prob_normal, resultado = predecir(modelo, img_path)

    # ── Paso 3: Contar foliculos ──────────────────────────────────────────────
    print("[INFO] Contando foliculos...")
    num_foliculos = contar_foliculos(img_path)

    # ── Paso 4: Generar mapa de calor ─────────────────────────────────────────
    print("[INFO] Generando mapa de calor...")
    mapa = generar_mapa_calor(modelo, img_path)

    if mapa is not None:
        nombre_base   = os.path.splitext(os.path.basename(img_path))[0]
        nombre_salida = f"{nombre_base}_mapa_calor.jpg"
        guardar_mapa_calor(img_path, mapa, nombre_salida)
    else:
        print("[WARN] No se pudo generar el mapa de calor")

    # ── Paso 5: Mostrar resultados ────────────────────────────────────────────
    imprimir_resultado(img_path, prob_sop, prob_normal, resultado, num_foliculos)


if __name__ == "__main__":
    main()
