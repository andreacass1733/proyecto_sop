from keras.applications import EfficientNetB0
from keras import layers, models

# Cargar EfficientNet-B0 preentrenado (sin la última capa)
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,   # quitamos la parte final
    input_shape=(224, 224, 3)
)

# Congelar el modelo base (no se entrena aún)
base_model.trainable = False

# Crear nuevas capas para tu problema (Cuenta con el tercer criterio vs. Normal)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(2, activation='softmax')(x)

# Modelo final
model = models.Model(inputs=base_model.input, outputs=output)

# Mostrar el modelo
model.summary()