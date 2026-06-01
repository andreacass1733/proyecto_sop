import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# ── Configuración ──────────────────────────────────────────────────────
TRAIN_DIR = 'D:/Andrea/Proyecto/proyecto_sop/backend/dataset/train'
VAL_DIR   = 'D:/Andrea/Proyecto/proyecto_sop/backend/dataset/val'
TEST_DIR  = 'D:/Andrea/Proyecto/proyecto_sop/backend/dataset/test'
IMG_SIZE  = (224, 224)
BATCH     = 32
CLASES    = ['Normal', 'SOP']

# ── Generadores con preprocess_input de EfficientNet ──────────────────
# MUY IMPORTANTE: No usar rescale=1./255, sino preprocess_input
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', shuffle=False
)
test_gen = val_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', shuffle=False
)

# ── Class weights para desbalance ──────────────────────────────────────
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights_arr))
print("Class weights:", class_weights)

# ── Construcción del modelo ────────────────────────────────────────────
def build_model():
    base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False  # Congelado en fase 1

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    return models.Model(inputs, outputs), base

model, base = build_model()

# Label smoothing — ayuda cuando las clases son difíciles de separar
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ── FASE 1: Entrenar solo el head ──────────────────────────────────────
print("\n========== FASE 1: Entrenando head ==========")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['accuracy']
)

callbacks_f1 = [
    EarlyStopping(monitor='val_accuracy', patience=6,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/best_fase1.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1)
]

history1 = model.fit(
    train_gen,
    epochs=25,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks_f1
)

# ── FASE 2: Fine-tuning — descongelar últimas 50 capas ────────────────
print("\n========== FASE 2: Fine-tuning (50 capas) ==========")
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

callbacks_f2 = [
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/best_fase2.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                      patience=4, min_lr=1e-7, verbose=1)
]

history2 = model.fit(
    train_gen,
    epochs=40,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks_f2
)

# ── FASE 3: Fine-tuning profundo — descongelar últimas 100 capas ──────
print("\n========== FASE 3: Fine-tuning profundo (100 capas) ==========")
for layer in base.layers[-100:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss=loss_fn,
    metrics=['accuracy']
)

callbacks_f3 = [
    EarlyStopping(monitor='val_accuracy', patience=10,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/best_fase3.keras', monitor='val_accuracy',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                      patience=5, min_lr=1e-8, verbose=1)
]

history3 = model.fit(
    train_gen,
    epochs=40,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks_f3
)

# ── Guardar modelo final ───────────────────────────────────────────────
model.save('models/efficientnet_sop_model.h5')
print("\n✅ Modelo guardado como efficientnet_sop_model.h5")

# ── Evaluación en test ─────────────────────────────────────────────────
print("\n========== EVALUACIÓN EN TEST ==========")
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=CLASES))

# ── Gráficas de entrenamiento ──────────────────────────────────────────
def plot_history(histories, labels):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    offset = 0
    for h, label in zip(histories, labels):
        epochs = range(offset + 1, offset + len(h.history['accuracy']) + 1)
        plt.plot(epochs, h.history['accuracy'], label=f'{label} - train')
        plt.plot(epochs, h.history['val_accuracy'], '--', label=f'{label} - val')
        offset += len(h.history['accuracy'])
    plt.title('Accuracy por época')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    offset = 0
    for h, label in zip(histories, labels):
        epochs = range(offset + 1, offset + len(h.history['loss']) + 1)
        plt.plot(epochs, h.history['loss'], label=f'{label} - train')
        plt.plot(epochs, h.history['val_loss'], '--', label=f'{label} - val')
        offset += len(h.history['loss'])
    plt.title('Loss por época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('images/entrenamiento/training_history.png')
    plt.show()
    print("📊 Gráfica guardada como training_history.png")

plot_history(
    [history1, history2, history3],
    ['Fase 1', 'Fase 2', 'Fase 3']
)