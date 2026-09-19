import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocesar_datos_xgboost():
    """
    Script de preprocesamiento exclusivo para el modelo XGBoost.
    Carga datasets de raw_data, limpia, estandariza y guarda artefactos en processed_data y models.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_base_dir = os.path.dirname(script_dir)
    
    raw_dir = os.path.join(xgb_base_dir, "raw_data")
    processed_dir = os.path.join(xgb_base_dir, "processed_data")
    models_dir = os.path.join(xgb_base_dir, "models")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    ruta_clinico = os.path.join(raw_dir, "dataset_clinico_limpio.xlsx")
    ruta_hormonal = os.path.join(raw_dir, "dataset_hormonal_limpio.xlsx")

    print("[INFO] Cargando datasets clinicos y hormonales exclusivamente para XGBoost...")
    df_clinico = pd.read_excel(ruta_clinico)
    df_hormonal = pd.read_excel(ruta_hormonal)

    # Limpieza de nombres de columnas
    df_clinico.columns = df_clinico.columns.str.strip().str.lower()
    df_hormonal.columns = df_hormonal.columns.str.strip().str.lower()

    # Mapeo binario para variables Si/No
    def binary_map(val):
        if pd.isna(val):
            return val
        s_val = str(val).strip().lower()
        if s_val in ['sí', 'si', '1', 'true', 'yes', 's']:
            return 1
        elif s_val in ['no', '0', 'false', 'n']:
            return 0
        return val

    cat_cols_clinico = ['tiene_sop', 'hirsutismo', 'perdida_cabello', 'acne', 'oscurecimiento_piel', 'ganancia_peso', 'embarazada']
    for col in cat_cols_clinico:
        if col in df_clinico.columns:
            df_clinico[col] = df_clinico[col].apply(binary_map)

    if 'tiene_sop' in df_hormonal.columns:
        df_hormonal['tiene_sop'] = df_hormonal['tiene_sop'].apply(binary_map)
        df_hormonal_features = df_hormonal.drop(columns=['tiene_sop'])
    else:
        df_hormonal_features = df_hormonal

    # Fusion tabular de caracteristicas
    df_completo = pd.concat([df_clinico, df_hormonal_features], axis=1)
    print(f"[OK] Datasets fusionados. Forma total: {df_completo.shape}")

    # Separacion de target (y) y variables predictoras (X)
    y = df_completo['tiene_sop'].astype(int)
    X = df_completo.drop(columns=['tiene_sop'])

    feature_names = X.columns.tolist()

    # Imputacion y Estandarizacion
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Guardado de artefactos del modelo XGBoost
    joblib.dump(scaler, os.path.join(models_dir, "scaler_xgboost.pkl"))
    joblib.dump(imputer, os.path.join(models_dir, "imputer_xgboost.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))
    print("[OK] Artefactos guardados en pipelines/xgboost/models/")

    # Division estratificada 70% Train, 15% Val, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    val_ratio_adjusted = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42, stratify=y_train_val
    )

    # Exportacion de conjuntos de datos procesados
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

    print("\n[EXITO] Preprocesamiento XGBoost finalizado:")
    print(f" - Train 70%: {X_train.shape[0]} registros -> {os.path.join(processed_dir, 'train_xgboost.csv')}")
    print(f" - Val 15%:   {X_val.shape[0]} registros -> {os.path.join(processed_dir, 'val_xgboost.csv')}")
    print(f" - Test 15%:  {X_test.shape[0]} registros -> {os.path.join(processed_dir, 'test_xgboost.csv')}")

if __name__ == '__main__':
    preprocesar_datos_xgboost()
