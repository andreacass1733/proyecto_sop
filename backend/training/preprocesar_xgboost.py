import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocesar_datos_xgboost():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xgb_dir = os.path.join(base_dir, "XGBoost")
    output_dir = os.path.join(xgb_dir, "preprocesado")
    os.makedirs(output_dir, exist_ok=True)

    ruta_clinico = os.path.join(xgb_dir, "dataset_clinico_limpio.xlsx")
    ruta_hormonal = os.path.join(xgb_dir, "dataset_hormonal_limpio.xlsx")

    print("[INFO] Cargando datasets clinicos y hormonales para XGBoost...")
    df_clinico = pd.read_excel(ruta_clinico)
    df_hormonal = pd.read_excel(ruta_hormonal)

    # Limpiar nombres de columnas
    df_clinico.columns = df_clinico.columns.str.strip().str.lower()
    df_hormonal.columns = df_hormonal.columns.str.strip().str.lower()

    # Mapeo binario para Si / No (manejo seguro de tildes o caracteres)
    def binary_map(val):
        if pd.isna(val):
            return val
        s_val = str(val).strip().lower()
        if s_val in ['sí', 'si', '1', 'true', 'yes', 's']:
            return 1
        elif s_val in ['no', '0', 'false', 'n']:
            return 0
        return val

    # Aplicar mapeo categorico en dataset clinico
    cat_cols_clinico = ['tiene_sop', 'hirsutismo', 'perdida_cabello', 'acne', 'oscurecimiento_piel', 'ganancia_peso', 'embarazada']
    for col in cat_cols_clinico:
        if col in df_clinico.columns:
            df_clinico[col] = df_clinico[col].apply(binary_map)

    if 'tiene_sop' in df_hormonal.columns:
        df_hormonal['tiene_sop'] = df_hormonal['tiene_sop'].apply(binary_map)
        df_hormonal_features = df_hormonal.drop(columns=['tiene_sop'])
    else:
        df_hormonal_features = df_hormonal

    # Fusion de caracteristicas clinicas y hormonales por registro
    df_completo = pd.concat([df_clinico, df_hormonal_features], axis=1)

    print(f"[OK] Datasets fusionados. Forma total: {df_completo.shape} (Registros, Caracteristicas)")

    # Separar Target (y) y Features (X)
    y = df_completo['tiene_sop'].astype(int)
    X = df_completo.drop(columns=['tiene_sop'])

    feature_names = X.columns.tolist()

    # Imputacion de valores faltantes (estrategia mediana por robustez clinica)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Estandarizacion de variables continuas y binarias (Z-score: mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Guardar transformadores para inferencia futura en el backend
    joblib.dump(scaler, os.path.join(output_dir, "scaler_xgboost.pkl"))
    joblib.dump(imputer, os.path.join(output_dir, "imputer_xgboost.pkl"))
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))
    print("[OK] Transformadores guardados: scaler_xgboost.pkl, imputer_xgboost.pkl, feature_names.pkl")

    # Division estratificada en 70% Train, 15% Val, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )

    val_ratio_adjusted = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42, stratify=y_train_val
    )

    # Exportacion de archivos CSV preprocesados
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['tiene_sop'] = y_train.values
    df_train.to_csv(os.path.join(output_dir, "train_xgboost.csv"), index=False)

    df_val = pd.DataFrame(X_val, columns=feature_names)
    df_val['tiene_sop'] = y_val.values
    df_val.to_csv(os.path.join(output_dir, "val_xgboost.csv"), index=False)

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['tiene_sop'] = y_test.values
    df_test.to_csv(os.path.join(output_dir, "test_xgboost.csv"), index=False)

    df_processed_all = pd.DataFrame(X_scaled, columns=feature_names)
    df_processed_all['tiene_sop'] = y.values
    df_processed_all.to_csv(os.path.join(output_dir, "dataset_xgboost_completo.csv"), index=False)

    print("\n[EXITO] PREPROCESAMIENTO INICIAL (35% TAREA 2) COMPLETADO:")
    print(f" - Muestras Totales: {len(df_completo)}")
    print(f" - N° de Caracteristicas ({len(feature_names)}): {feature_names}")
    print(f" - Conjunto Entrenamiento (Train 70%): {X_train.shape[0]} muestras")
    print(f" - Conjunto Validacion (Val 15%): {X_val.shape[0]} muestras")
    print(f" - Conjunto Prueba (Test 15%): {X_test.shape[0]} muestras")
    print(f" - Archivos guardados en: {output_dir}")

if __name__ == '__main__':
    preprocesar_datos_xgboost()
