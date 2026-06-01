import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuración ──────────────────────────────────────────────────────
TEST_DIR  = 'D:/Andrea/Proyecto/proyecto_sop/backend/dataset/test'
IMG_SIZE  = (224, 224)
BATCH     = 32
CLASES    = ['Normal', 'SOP']

# ── Cargar el mejor modelo guardado ───────────────────────────────────
model = load_model('../models/best_fase1.keras')  # ← cambia si usas fase2 o fase3
print("✅ Modelo cargado correctamente")

# ── Generador de test ──────────────────────────────────────────────────
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

# ── Predicciones ───────────────────────────────────────────────────────
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# ── Reporte ────────────────────────────────────────────────────────────
print("\nMatriz de Confusión:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=CLASES))

# ── Visualización de la matriz de confusión ────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASES, yticklabels=CLASES)
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('../images/entrenamiento/confusion_matrix.png')
plt.show()
print("📊 Matriz guardada como confusion_matrix.png")