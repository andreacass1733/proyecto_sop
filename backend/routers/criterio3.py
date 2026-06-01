import os
import shutil
import numpy as np
import subprocess
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from database import get_db
from models import Criterio3
from schemas import Criterio3Response, ValidarRequest

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

router = APIRouter(prefix="/criterio3", tags=["Criterio 3 - Ecografía"])

# ── Cargar modelo ──────────────────────────────────────────────────────
MODEL_PATH = "models/best_fase1.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado")

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASES = ["Normal", "SOP"]


def run_prediction(img_path: str):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array, verbose=0)[0]
    prob_sop = float(pred[1])
    resultado = "Cumple criterio" if prob_sop >= 0.5 else "No cumple criterio"
    return prob_sop, resultado


# ── POST /criterio3/predecir ───────────────────────────────────────────
@router.post("/predecir")
async def predecir(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prob_sop, resultado = run_prediction(filepath)

    registro = Criterio3(
        imagen_path=filepath,
        prob_sop=prob_sop,
        resultado=resultado
    )
    db.add(registro)
    db.commit()
    db.refresh(registro)

    return {
        "id":              registro.id,
        "resultado":       resultado,
        "prob_sop":        round(prob_sop * 100, 2),
        "prob_normal":     round((1 - prob_sop) * 100, 2),
        "imagen_guardada": filepath
    }


# ── GET /criterio3/stats ───────────────────────────────────────────────
@router.get("/stats")   # ✅ sin /criterio3/ porque el prefix ya lo agrega
def get_stats(db: Session = Depends(get_db)):
    total     = db.query(Criterio3).count()
    cumple    = db.query(Criterio3).filter(Criterio3.resultado == "Cumple criterio").count()
    no_cumple = db.query(Criterio3).filter(Criterio3.resultado == "No cumple criterio").count()

    # Precisión real basada en imágenes validadas
    validadas = db.query(Criterio3).filter(Criterio3.validado == True).count()
    correctas = db.query(Criterio3).filter(
        Criterio3.validado == True,
        Criterio3.resultado == Criterio3.etiqueta_real
    ).count()
    precision = (correctas / validadas) if validadas > 0 else 0.9531

    return {
        "total":                       total,
        "cumple":                      cumple,
        "no_cumple":                   no_cumple,
        "precision_modelo":            precision,
        "total_dataset_entrenamiento": 3096,
    }


# ── GET /criterio3/imagenes ────────────────────────────────────────────
@router.get("/imagenes", response_model=list[Criterio3Response])
def listar(db: Session = Depends(get_db)):
    return db.query(Criterio3).order_by(Criterio3.fecha.desc()).all()


# ── PUT /criterio3/validar/{id} ────────────────────────────────────────
@router.put("/validar/{id}")
def validar(id: int, body: ValidarRequest, db: Session = Depends(get_db)):
    registro = db.query(Criterio3).filter(Criterio3.id == id).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Registro no encontrado")

    registro.validado     = True
    registro.etiqueta_real = body.etiqueta_real
    db.commit()
    return {"mensaje": f"Imagen {id} validada como {body.etiqueta_real} ✅"}


# ── POST /criterio3/reentrenar ─────────────────────────────────────────
@router.post("/reentrenar")
def reentrenar(db: Session = Depends(get_db)):
    validadas = db.query(Criterio3).filter(Criterio3.validado == True).count()

    if validadas < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Solo hay {validadas} imágenes validadas. Se necesitan al menos 50."
        )

    subprocess.Popen(["python", "training/entrenar_modelo.py"])
    return {"mensaje": f"Reentrenamiento iniciado con {validadas} imágenes validadas 🚀"}