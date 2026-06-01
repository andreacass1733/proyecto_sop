### Importación de carpetas y rutas
import os
### Ayuda para la mezcla de las imagenes aleatoriamente
import random
### Copia de archivos -> En neustro caso las imagenes
import shutil

# Rutas
dataset_original = "dataset_original" ## Todas las imagenes que se encuentran en esta parte se dividiran en la nueva carpeta a crear
dataset_destino = "dataset" ##Las imagenes se crearan en esta carpeta

# Clases
clases = ["sop", "normal"] ## Nombres de las carpetas que contienen las imagenes

# Porcentajes
train_ratio = 0.7 ## Parte del 70% del entrenamiento
val_ratio = 0.15 ## 15% para el ajuste del proyecto
test_ratio = 0.15 ## 15% para la evaluación del resultado final

# Crear carpetas destino
for split in ["train", "val", "test"]:
    for clase in clases:
        ruta = os.path.join(dataset_destino, split, clase)
        os.makedirs(ruta, exist_ok=True)

# Procesar cada clase
for clase in clases:
    ruta_clase = os.path.join(dataset_original, clase)
    imagenes = os.listdir(ruta_clase)

    random.shuffle(imagenes)

    total = len(imagenes)
    train_fin = int(total * train_ratio)
    val_fin = int(total * (train_ratio + val_ratio))

    train_imgs = imagenes[:train_fin]
    val_imgs = imagenes[train_fin:val_fin]
    test_imgs = imagenes[val_fin:]

    for img in train_imgs:
        shutil.copy(os.path.join(ruta_clase, img),
                    os.path.join(dataset_destino, "train", clase, img))

    for img in val_imgs:
        shutil.copy(os.path.join(ruta_clase, img),
                    os.path.join(dataset_destino, "val", clase, img))

    for img in test_imgs:
        shutil.copy(os.path.join(ruta_clase, img),
                    os.path.join(dataset_destino, "test", clase, img))

print("✅ Dataset dividido correctamente")