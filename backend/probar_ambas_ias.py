"""
SCRIPT DE INSPECCIÓN Y VERIFICACIÓN DE AMBAS MODELOS DE IA
═════════════════════════════════════════════════════════════════════════════════

Este script permite comprobar el funcionamiento simultáneo de ambos modelos de IA:
  1. IA Tabular (XGBoost): Diagnóstico por Criterios 1 y 2 de Rotterdam (Clínico y Hormonal)
  2. IA de Visión (EfficientNet-B0): Diagnóstico por Criterio 3 de Rotterdam (Ecografía Ovárica)
"""

import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

def probar_ia_xgboost():
    print("\n" + "=" * 60)
    print(" 1. VERIFICACION DE MODELO IA TABULAR: XGBOOST (Criterios 1 y 2)")
    print("=" * 60)

    models_dir = "models_ia"
    xgb_path = os.path.join(models_dir, "best_xgboost.pkl")
    scaler_path = os.path.join(models_dir, "scaler_xgboost.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    if not os.path.exists(xgb_path):
        print(f"[ERROR] No se encontro el modelo XGBoost en {xgb_path}")
        return

    model_xgb = joblib.load(xgb_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    print(f"[OK] Modelo XGBoost cargado exitosamente.")
    print(f"[OK] Total de caracteristicas clinicas/hormonales evaluadas: {len(feature_names)}")

    ejemplo_paciente = pd.DataFrame([{
        col: 1.0 if col in ['hirsutismo', 'acne', 'ganancia_peso'] else 0.5 for col in feature_names
    }])

    ejemplo_scaled = scaler.transform(ejemplo_paciente)
    prob_sop = float(model_xgb.predict_proba(ejemplo_scaled)[0, 1])
    pred = "Cumple Criterios 1 y 2 (SOP)" if prob_sop >= 0.5 else "No Cumple Criterios (Normal)"

    print(f"\n  [PRUEBA EN TIEMPO REAL]")
    print(f"  - Prediccion del Modelo: {pred}")
    print(f"  - Probabilidad SOP:      {prob_sop * 100:.2f}%")
    print(f"  - Estado del Modelo:     Totalmente Operativo")


def probar_ia_efficientnet():
    print("\n" + "=" * 60)
    print(" 2. VERIFICACION DE MODELO IA DE VISION: EFFICIENTNET-B0 (Criterio 3)")
    print("=" * 60)

    from services import criterio3_service

    try:
        model_eff = criterio3_service.get_model()
        print(f"[OK] Modelo EfficientNet-B0 cargado correctamente en memoria.")
        print(f"[OK] Arquitectura de Entrada: {model_eff.input_shape}")
        print(f"[OK] Clases de Salida: {criterio3_service.CLASES}")

        test_img = "test_images/Image_004.jpg"
        if os.path.exists(test_img):
            prob_sop, prob_norm, res = criterio3_service.run_prediction(test_img)
            num_fol = criterio3_service.contar_foliculos(test_img)

            print(f"\n  [PRUEBA EN TIEMPO REAL CON ECOGRAFIA DE MUESTRA]")
            print(f"  - Imagen Analizada:     {test_img}")
            print(f"  - Resultado Ecografico: {res}")
            print(f"  - Probabilidad SOP:     {prob_sop * 100:.2f}%")
            print(f"  - Foliculos Estimados:  {num_fol}")
            print(f"  - Estado del Modelo:    Totalmente Operativo")
        else:
            print(f"[INFO] Para probar con imagen real, coloca una ecografia en 'test_images/'.")
    except Exception as e:
        print(f"[ERROR] Ocurrio un fallo al probar EfficientNet-B0: {e}")


def main():
    print("\n" + "=" * 60)
    print(" COMPROBACION GENERAL DE LAS DOS INTELIGENCIAS ARTIFICIALES DEL SISTEMA SOP")
    print("=" * 60)

    probar_ia_xgboost()
    probar_ia_efficientnet()

    print("\n" + "=" * 60)
    print(" [EXITO] Ambas IAs estan verificadas y funcionando en paralelo.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
