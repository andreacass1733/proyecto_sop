import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

def predict_xgboost_sample(datos_paciente_dict: dict) -> tuple[float, str]:
    """
    Recibe un diccionario con las características clínicas y hormonales de la paciente.
    Aplica la imputación y estandarización de pipelines/xgboost/models/ y predice con el modelo XGBoost.
    
    Retorna: (probabilidad_sop, resultado_clasificacion)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_base_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(xgb_base_dir, "models")

    # Cargar artefactos
    scaler = joblib.load(os.path.join(models_dir, "scaler_xgboost.pkl"))
    imputer = joblib.load(os.path.join(models_dir, "imputer_xgboost.pkl"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(models_dir, "best_xgboost.json"))

    # Crear dataframe de 1 fila alineado a las características esperadas
    row_data = {}
    for feat in feature_names:
        row_data[feat] = datos_paciente_dict.get(feat, np.nan)
        
    df_single = pd.DataFrame([row_data])

    # Transformar
    df_imputed = imputer.transform(df_single)
    df_scaled = scaler.transform(df_imputed)

    # Predecir
    prob_sop = float(model.predict_proba(df_scaled)[0, 1])
    resultado = "Cumple criterio clínico/hormonal (SOP)" if prob_sop >= 0.5 else "No cumple criterio clínico/hormonal"

    return prob_sop, resultado

if __name__ == '__main__':
    # Ejemplo de prueba rápida
    ejemplo = {
        'presion_sistolica': 120,
        'presion_diastolica': 80,
        'cadera_cm': 95,
        'cintura_cm': 85,
        'relacion_cintura_cadera': 0.89,
        'hirsutismo': 1,
        'perdida_cabello': 1,
        'acne': 1,
        'oscurecimiento_piel': 1,
        'ganancia_peso': 1,
        'embarazada': 0,
        'fsh': 5.5,
        'lh': 12.8,
        'tsh': 2.1,
        'amh': 6.2,
        'prolactina': 18.5,
        'progesterona': 0.8,
        'glucosa_basal': 95,
        'testosterona_total': 75.0,
        'androstenediona': 3.5,
        'insulina_basal': 15.0,
        'dhea_s': 350.0,
        'colesterol_total': 190.0,
        'ldl': 110.0,
        'hdl': 45.0,
        'trigliceridos': 150.0
    }
    prob, res = predict_xgboost_sample(ejemplo)
    print(f"[OK] Probabilidad SOP XGBoost: {prob * 100:.2f}% -> {res}")
