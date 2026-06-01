import numpy as np
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model("../models/best_fase1.keras")

classes = ["No cumple criterio ecográfico", "Cumple criterio ecográfico"]

results = []

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ✅ correcto

    prediction = model.predict(img_array, verbose=0)[0]

    prob_criterio = prediction[1]  # clase positiva (SOP)
    clase = "Cumple criterio" if prob_criterio >= 0.5 else "No cumple criterio"

    print("\nImagen:", os.path.basename(img_path))
    print(f"Resultado: {clase}")
    print(f"Probabilidad criterio ecográfico: {round(float(prob_criterio), 4)}")
    print(f"Probabilidades → Normal: {round(float(prediction[0])*100, 1)}% | SOP: {round(float(prediction[1])*100, 1)}%")

    results.append({
        "imagen": img_path,
        "probabilidad_criterio": float(prob_criterio),
        "resultado": clase
    })


# ── PROBAR TODA LA CARPETA ─────────────────────────────────────────────
folder = "../test_images"

for img_name in os.listdir(folder):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder, img_name)
        predict_image(img_path)


# ── GUARDAR RESULTADOS ─────────────────────────────────────────────────
os.makedirs("../outputs", exist_ok=True)

with open("../outputs/resultados_prediccion.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Imagen", "Probabilidad_Criterio", "Resultado"])

    for r in results:
        writer.writerow([r["imagen"], r["probabilidad_criterio"], r["resultado"]])

total = len(results)
sop = sum(1 for r in results if r["probabilidad_criterio"] >= 0.5)
normal = total - sop
avg = sum(r["probabilidad_criterio"] for r in results) / total if total > 0 else 0

print("\n========================")
print("RESUMEN FINAL")
print("========================")
print("Total imágenes:", total)
print("Cumple criterio (SOP):", sop)
print("No cumple criterio:", normal)
print("Probabilidad media:", round(avg, 4))

# ── GRÁFICA ────────────────────────────────────────────────────────────
labels = ["No cumple", "Cumple"]
values = [normal, sop]
colores = ["#4CAF50", "#F44336"]

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=colores)
plt.title("Resultado del criterio ecográfico")
plt.ylabel("Cantidad de imágenes")
plt.tight_layout()
plt.savefig("../outputs/grafica_resultados.png")
plt.show()

print("\n✅ Resultados guardados en outputs/resultados_prediccion.csv")
print("📊 Gráfica guardada en outputs/grafica_resultados.png")