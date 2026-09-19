import os
import cv2
import numpy as np

def aplicar_preprocesamiento_imagen(ruta_origen, ruta_destino, target_size=(224, 224)):
    """
    Aplica técnicas de preprocesamiento avanzado a una ecografía ovárica:
    1. Redimensionamiento a 224x224 px.
    2. Conversión a escala de grises.
    3. Reducción de ruido Speckle mediante Filtro Bilateral (preserva bordes foliculares).
    4. Realce de contraste adaptativo (CLAHE - Contrast Limited Adaptive Histogram Equalization).
    5. Normalización de intensidad (0 a 255 uint8).
    """
    img = cv2.imread(ruta_origen)
    if img is None:
        return False

    # Convertir a escala de grises si tiene 3 canales
    if len(img.shape) == 3 and img.shape[2] == 3:
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gris = img

    # 1. Redimensionamiento
    gris_resized = cv2.resize(gris, target_size, interpolation=cv2.INTER_AREA)

    # 2. Reducción de ruido speckle (Filtro Bilateral: d=5, sigmaColor=75, sigmaSpace=75)
    # A diferencia del desenfoque gaussiano, preserva los bordes nítidos de los folículos
    denoised = cv2.bilateralFilter(gris_resized, d=5, sigmaColor=75, sigmaSpace=75)

    # 3. Realce de contraste adaptativo (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Reconvertir a 3 canales RGB (exigido por EfficientNet-B0)
    img_final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Guardar en destino
    os.makedirs(os.path.dirname(ruta_destino), exist_ok=True)
    cv2.imwrite(ruta_destino, img_final)
    return True

def preprocesar_dataset_imagenes_completo():
    """
    Procesa el conjunto de datos de entrenamiento/validación/prueba de EfficientNet al 70%.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eff_dir = os.path.dirname(script_dir)
    
    # Origen del dataset original o dataset dividido
    origen_dir = os.path.join(eff_dir, "dataset")
    if not os.path.exists(origen_dir):
        # Si no está en pipelines, buscar en backend raíz
        origen_dir = r"d:\Andrea\Proyecto\proyecto_sop\backend\dataset"

    destino_dir = os.path.join(eff_dir, "dataset_procesado")
    
    print(f"[INFO] Iniciando preprocesamiento avanzado de imágenes (70% Tarea 2)...")
    print(f" Origen:  {origen_dir}")
    print(f" Destino: {destino_dir}")

    total_procesadas = 0
    splits = ["train", "val", "test"]
    clases = ["sop", "normal"]

    for split in splits:
        for clase in clases:
            dir_clase = os.path.join(origen_dir, split, clase)
            if not os.path.exists(dir_clase):
                continue
            
            archivos = [f for f in os.listdir(dir_clase) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Muestra de hasta 100 por categoría para demostración rápida de 70%
            archivos_muestra = archivos[:100]

            for archivo in archivos_muestra:
                src_path = os.path.join(dir_clase, archivo)
                dst_path = os.path.join(destino_dir, split, clase, archivo)
                if aplicar_preprocesamiento_imagen(src_path, dst_path):
                    total_procesadas += 1

    print(f"[EXITO] Preprocesamiento de imágenes (70% Tarea 2) finalizado. Total procesadas: {total_procesadas}")

if __name__ == '__main__':
    preprocesar_dataset_imagenes_completo()
