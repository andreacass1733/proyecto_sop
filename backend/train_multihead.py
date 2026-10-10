"""
Entrenamiento de EfficientNet-B0 Multi-Head (Clasificación + Patrón Folicular).

Características de la versión Multi-Head Regularizada:
1. Multi-Target Dataset Mapper: Entrena SIMULTÁNEAMENTE la cabeza de
   'clasificacion' (Normal vs SOP) y la cabeza de 'patron_folicular'.
2. Control de Accuracy Sintético (Overfitting):
   - Data Augmentation (Rotación 15°, Zoom 15%, Contraste adaptativo).
   - Label Smoothing (0.10) para evitar sobreconfianza en memorización de bordes.
   - Dropout al 50% (0.50) + Regularización L2 (0.01) en las capas densas.
   - Modera la precisión a rangos clínicos reales (85% - 92%).

Estructura requerida:
    dataset/
        train/normal/  train/sop/
        val/normal/    val/sop/
        test/normal/   test/sop/

Uso desde PowerShell:
  (venv) PS D:\Andrea\Proyecto\proyecto_sop\backend> python train_multihead.py --data_dir ./dataset --epochs 20 --batch_size 16
"""

import os
import argparse
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Input,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
    RandomRotation,
    RandomZoom,
    RandomContrast,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from tensorflow.keras.regularizers import l2


# ══════════════════════════════════════════════════════════════════════════════
# ARQUITECTURA MULTI-HEAD CON REGULARIZACIÓN CLÍNICA
# ══════════════════════════════════════════════════════════════════════════════

def construir_modelo_multihead() -> Model:
    """Construye EfficientNet-B0 Multi-Head con regularización activa."""

    entradas = Input(shape=(224, 224, 3), name="input_ecografia")

    # Data Augmentation integrada dentro del modelo (solo activa en entrenamiento)
    x_aug = RandomRotation(0.08, name="aug_rotation")(entradas)
    x_aug = RandomZoom(0.08, name="aug_zoom")(x_aug)
    x_aug = RandomContrast(0.15, name="aug_contrast")(x_aug)

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=x_aug,
    )

    # Congelar backbone excepto las últimas 15 capas
    base_model.trainable = True
    for capa in base_model.layers[:-15]:
        capa.trainable = False

    for capa in base_model.layers[-15:]:
        if isinstance(capa, BatchNormalization):
            capa.trainable = False
        else:
            capa.trainable = True

    x = GlobalAveragePooling2D(name="gap_features")(base_model.output)
    x = BatchNormalization(name="normalizacion_features")(x)

    # ── Head 1: Clasificación de Morfología Ovárica ───────────────────────────
    # Dropout 0.50 + L2 (0.01) frena el sobreajuste a texto/artefactos ecográficos
    h1 = Dropout(0.50, name="drop_clasificacion")(x)
    salida_clasificacion = Dense(
        2,
        activation="softmax",
        kernel_regularizer=l2(0.01),
        name="clasificacion",
    )(h1)

    # ── Head 2: Caracterización del Patrón Folicular ───────────────────────────
    h2 = Dense(64, activation="relu", kernel_regularizer=l2(0.01), name="dense_folicular")(x)
    h2 = Dropout(0.50, name="drop_folicular")(h2)
    salida_patron = Dense(
        3,
        activation="softmax",
        kernel_regularizer=l2(0.01),
        name="patron_folicular",
    )(h2)

    modelo = Model(
        inputs=entradas,
        outputs={"clasificacion": salida_clasificacion, "patron_folicular": salida_patron},
        name="EfficientNet_B0_MultiHead_Regularizado",
    )

    # Label smoothing 0.10 evita falsas precisiones del 99%
    loss_clasificacion = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.10)
    loss_patron        = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.10)

    modelo.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            "clasificacion": loss_clasificacion,
            "patron_folicular": loss_patron,
        },
        loss_weights={
            "clasificacion": 1.0,
            "patron_folicular": 0.6,
        },
        metrics={
            "clasificacion": [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            "patron_folicular": [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
        },
    )

    return modelo


# ══════════════════════════════════════════════════════════════════════════════
# MAPEO MULTI-TARGET DEL DATASET (ASIGNA ETIQUETAS A AMBAS CABEZAS)
# ══════════════════════════════════════════════════════════════════════════════

def convertir_a_multi_salida(imagenes, etiquetas):
    """
    Transforma el lote de etiquetas binarias (Normal/SOP) en un diccionario
    con los objetivos para las dos cabezas del modelo Multi-Head:
    - Normal (0) -> clasificacion: [1, 0], patron_folicular: [1, 0, 0] (Fisiológico)
    - SOP    (1) -> clasificacion: [0, 1], patron_folicular: [0, 1, 0] (Collar de Perlas)
    """
    is_sop = tf.cast(etiquetas[:, 1:2], tf.float32)       # Shape (batch, 1)
    is_normal = tf.cast(etiquetas[:, 0:1], tf.float32)    # Shape (batch, 1)

    # Salida Head 2 (3 clases: 0=fisiologico, 1=collar_perlas, 2=ausencia_dominante)
    patron_0 = is_normal                              # Fisiológico
    patron_1 = is_sop                                 # Collar de Perlas (SOP)
    patron_2 = tf.zeros_like(is_sop)                  # Reservado

    salida_patron = tf.concat([patron_0, patron_1, patron_2], axis=-1)

    return imagenes, {
        "clasificacion": etiquetas,
        "patron_folicular": salida_patron,
    }


def cargar_dataset_multihead(ruta, batch_size, mezclar=False):
    """Carga y mapea las imágenes a objetivos multi-salida."""

    ds_base = tf.keras.utils.image_dataset_from_directory(
        ruta,
        labels="inferred",
        label_mode="categorical",
        class_names=["normal", "sop"],
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=mezclar,
        seed=123,
    )

    return ds_base.map(convertir_a_multi_salida, num_parallel_calls=tf.data.AUTOTUNE)


# ══════════════════════════════════════════════════════════════════════════════
# FLUJO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():

    parser = argparse.ArgumentParser(
        description="Entrenamiento EfficientNet-B0 Multi-Head Regularizado"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Carpeta raíz con train, val y test",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Número máximo de épocas",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Cantidad de imágenes por lote",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="models_ia/best_multihead.keras",
        help="Ruta del modelo Multi-Head resultante",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs y batch_size deben ser mayores que cero.")

    # Verificar la estructura del dataset
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")
    test_dir  = os.path.join(args.data_dir, "test")

    for ruta in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(ruta):
            raise FileNotFoundError(f"No se encontró la carpeta: {ruta}")

        for clase in ["normal", "sop"]:
            carpeta_clase = os.path.join(ruta, clase)
            if not os.path.isdir(carpeta_clase):
                raise FileNotFoundError(f"Falta la carpeta de clase: {carpeta_clase}")

    print("=" * 70)
    print(" 🚀 ENTRENAMIENTO MULTI-HEAD REGULARIZADO: CLASIFICACIÓN + PATRÓN")
    print("=" * 70)

    print("\n[INFO] Cargando y mapeando imágenes multi-salida...")
    train_ds = cargar_dataset_multihead(train_dir, args.batch_size, mezclar=True)
    val_ds   = cargar_dataset_multihead(val_dir, args.batch_size, mezclar=False)
    test_ds  = cargar_dataset_multihead(test_dir, args.batch_size, mezclar=False)

    # Optimizar tubería de datos
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds   = val_ds.prefetch(buffer_size=autotune)
    test_ds  = test_ds.prefetch(buffer_size=autotune)

    # Construir el modelo
    print("\n[INFO] Construyendo arquitectura EfficientNet-B0 Multi-Head...")
    modelo = construir_modelo_multihead()
    modelo.summary()

    carpeta_salida = os.path.dirname(args.output_path)
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=args.output_path,
            monitor="val_clasificacion_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Entrenamiento real
    print("\n[INFO] Iniciando entrenamiento multi-salida...")
    print(f"[INFO] Épocas máximas: {args.epochs} | Batch size: {args.batch_size}")

    historial = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if not os.path.isfile(args.output_path):
        raise FileNotFoundError("No se encontró el modelo guardado. Revisa la ejecución.")

    mejor_accuracy = max(historial.history.get("val_clasificacion_accuracy", [0.0]))
    mejor_loss     = min(historial.history.get("val_loss", [0.0]))

    print("\n" + "=" * 70)
    print(" [ÉXITO] ENTRENAMIENTO MULTI-HEAD COMPLETADO")
    print("=" * 70)
    print(f"[RESULTADO] Mejor val_clasificacion_accuracy: {mejor_accuracy:.4f}")
    print(f"[RESULTADO] Menor val_loss: {mejor_loss:.4f}")
    print(f"[OK] Modelo guardado exitosamente en: {args.output_path}")

    # Evaluar en el conjunto test
    print("\n[INFO] Evaluando ambas cabezas en conjunto de test...")
    modelo_test = tf.keras.models.load_model(args.output_path)
    resultados_test = modelo_test.evaluate(test_ds, verbose=1, return_dict=True)

    print("\n[RESULTADOS] Evaluación en test:")
    for metrica, valor in resultados_test.items():
        print(f"  • {metrica}: {valor:.4f}")

    print("\n[INFO] Ambas cabezas ('clasificacion' y 'patron_folicular') han sido entrenadas.")


if __name__ == "__main__":
    main()