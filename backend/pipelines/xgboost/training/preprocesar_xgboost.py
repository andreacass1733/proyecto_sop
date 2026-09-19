"""
MÓDULO DE PREPROCESAMIENTO TABULAR EXCLUSIVO PARA XGBOOST (70% TAREA 2)
═════════════════════════════════════════════════════════════════════════════

Este script realiza el tratamiento científico completo de los datos clínicos
y hormonales necesarios para entrenar el modelo XGBoost en el diagnóstico de SOP:

PASOS DE EJECUCIÓN:
  1. Carga de los archivos Excel originales desde `raw_data/`
  2. Limpieza de nombres de variables y mapeo categórico binario (Sí/No -> 1/0)
  3. Fusión de características clínicas (11) y hormonales (15) por registro de paciente
  4. Imputación de valores vacíos mediante la mediana estadística del atributo (SimpleImputer)
  5. Acotamiento de valores atípicos (Outlier Clipping por Rango Intercuartílico - IQR)
  6. Estandarización de variables continuas Z-Score (StandardScaler: media=0, std=1)
  7. División estratificada de conjuntos: 70% Entrenamiento, 15% Validación, 15% Prueba
  8. Exportación de artefactos serializados (pkl) y datasets procesados (csv)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocesar_datos_xgboost():
    """
    Función principal de preprocesamiento tabular para XGBoost.
    """
    # ── PASO 1: Configurar rutas de archivos ──────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_base_dir = os.path.dirname(script_dir)
    
    raw_dir = os.path.join(xgb_base_dir, "raw_data")
    processed_dir = os.path.join(xgb_base_dir, "processed_data")
    models_dir = os.path.join(xgb_base_dir, "models")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    ruta_clinico = os.path.join(raw_dir, "dataset_clinico_limpio.xlsx")
    ruta_hormonal = os.path.join(raw_dir, "dataset_hormonal_limpio.xlsx")

    print("[INFO] Cargando datasets clinicos y hormonales para XGBoost...")
    df_clinico = pd.read_excel(ruta_clinico)
    df_hormonal = pd.read_excel(ruta_hormonal)

    # ── PASO 2: Normalización de nombres de columnas y codificación ─────────
    df_clinico.columns = df_clinico.columns.str.strip().str.lower()
    df_hormonal.columns = df_hormonal.columns.str.strip().str.lower()

    # Función auxiliar para convertir valores cualitativos a 1 o 0 de forma segura
    def binary_map(val):
        if pd.isna(val):
            return val
        s_val = str(val).strip().lower()
        if s_val in ['sí', 'si', '1', 'true', 'yes', 's']:
            return 1
        elif s_val in ['no', '0', 'false', 'n']:
            return 0
        return val

    # Transformar columnas sintomáticas a binarias (1/0)
    cat_cols_clinico = ['tiene_sop', 'hirsutismo', 'perdida_cabello', 'acne', 'oscurecimiento_piel', 'ganancia_peso', 'embarazada']
    for col in cat_cols_clinico:
        if col in df_clinico.columns:
            df_clinico[col] = df_clinico[col].apply(binary_map)

    if 'tiene_sop' in df_hormonal.columns:
        df_hormonal['tiene_sop'] = df_hormonal['tiene_sop'].apply(binary_map)
        df_hormonal_features = df_hormonal.drop(columns=['tiene_sop'])
    else:
        df_hormonal_features = df_hormonal

    # ── PASO 3: Fusión de variables clínicas y hormonales por paciente ────────
    df_completo = pd.concat([df_clinico, df_hormonal_features], axis=1)
    print(f"[OK] Datasets fusionados exitosamente. Forma matriz: {df_completo.shape}")

    # Separar la variable objetivo (y: tiene_sop) de las variables predictoras (X)
    y = df_completo['tiene_sop'].astype(int)
    X = df_completo.drop(columns=['tiene_sop'])

    feature_names = X.columns.tolist()

    # ── PASO 4: Tratamiento de Outliers (IQR Clipping para robustez clínica) ─
    # Evita que picos hormonales aislados alteren la distribución del modelo
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            X[col] = np.clip(X[col], lower_bound, upper_bound)

    # ── PASO 5: Imputación de nulos y Estandarización Z-Score ────────────────
    imputer = SimpleImputer(strategy='median')  # Rellena valores omisos con la mediana
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()                   # Estandariza a media=0 y desviacion=1
    X_scaled = scaler.fit_transform(X_imputed)

    # ── PASO 6: Guardar transformadores para inferencia futura en el backend ─
    joblib.dump(scaler, os.path.join(models_dir, "scaler_xgboost.pkl"))
    joblib.dump(imputer, os.path.join(models_dir, "imputer_xgboost.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))
    print("[OK] Artefactos de transformacion guardados en pipelines/xgboost/models/")

    # ── PASO 7: División estratificada (70% Train, 15% Val, 15% Test) ────────
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    val_ratio_adjusted = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42, stratify=y_train_val
    )

    # ── PASO 8: Exportar datasets procesados en formato CSV ──────────────────
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['tiene_sop'] = y_train.values
    df_train.to_csv(os.path.join(processed_dir, "train_xgboost.csv"), index=False)

    df_val = pd.DataFrame(X_val, columns=feature_names)
    df_val['tiene_sop'] = y_val.values
    df_val.to_csv(os.path.join(processed_dir, "val_xgboost.csv"), index=False)

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['tiene_sop'] = y_test.values
    df_test.to_csv(os.path.join(processed_dir, "test_xgboost.csv"), index=False)

    df_processed_all = pd.DataFrame(X_scaled, columns=feature_names)
    df_processed_all['tiene_sop'] = y.values
    df_processed_all.to_csv(os.path.join(processed_dir, "dataset_xgboost_completo.csv"), index=False)

    print("\n[EXITO] Preprocesamiento XGBoost (70% Tarea 2) completado:")
    print(f" - Train 70%: {X_train.shape[0]} registros")
    print(f" - Val 15%:   {X_val.shape[0]} registros")
    print(f" - Test 15%:  {X_test.shape[0]} registros")

if __name__ == '__main__':
    preprocesar_datos_xgboost()
