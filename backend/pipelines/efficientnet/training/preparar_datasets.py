from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Data augmentation para entrenamiento ---
train_datagen = ImageDataGenerator(
    rescale=1./255,          # normaliza los píxeles entre 0 y 1
    rotation_range=20,       # gira un poco las imágenes
    width_shift_range=0.1,   # mueve horizontalmente
    height_shift_range=0.1,  # mueve verticalmente
    shear_range=0.1,         # efecto de corte
    zoom_range=0.1,          # zoom
    horizontal_flip=True,    # voltea horizontal
    fill_mode='nearest'
)

# Solo normalizamos para validación y test
test_val_datagen = ImageDataGenerator(rescale=1./255)

# --- Cargar imágenes desde carpetas ---
train_dataset = train_datagen.flow_from_directory(
    'D:/Andrea/Proyecto/proyecto_sop/backend/dataset/train',         
    target_size=(224, 224),   # tamaño que necesita EfficientNet-B0
    batch_size=32,
    class_mode='categorical'  # porque tenemos 2 clases: Normal y SOP
)

val_dataset = test_val_datagen.flow_from_directory(
    'D:/Andrea/Proyecto/proyecto_sop/backend/dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_dataset = test_val_datagen.flow_from_directory(
    'D:/Andrea/Proyecto/proyecto_sop/backend/dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False               # importante para métricas precisas
)