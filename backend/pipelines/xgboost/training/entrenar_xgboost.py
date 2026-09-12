import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import joblib

def entrenar_xgboost():
    """
    Script de entrenamiento exclusivo para XGBoost.
    Lee los CSVs procesados de processed_data, entrena XGBClassifier y guarda el modelo en models.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_base_dir = os.path.dirname(script_dir)

    processed_dir = os.path.join(xgb_base_dir, "processed_data")
    models_dir = os.path.join(xgb_base_dir, "models")

    ruta_train = os.path.join(processed_dir, "train_xgboost.csv")
    ruta_val = os.path.join(processed_dir, "val_xgboost.csv")
    ruta_test = os.path.join(processed_dir, "test_xgboost.csv")

    if not os.path.exists(ruta_train):
        print("[ERROR] No se encontraron los datos preprocesados. Ejecuta primero preprocesar_xgboost.py")
        return

    print("[INFO] Cargando conjuntos de datos para XGBoost...")
    df_train = pd.read_csv(ruta_train)
    df_val = pd.read_csv(ruta_val)
    df_test = pd.read_csv(ruta_test)

    X_train, y_train = df_train.drop(columns=['tiene_sop']), df_train['tiene_sop']
    X_val, y_val = df_val.drop(columns=['tiene_sop']), df_val['tiene_sop']
    X_test, y_test = df_test.drop(columns=['tiene_sop']), df_test['tiene_sop']

    print(f"[INFO] Entrenando XGBClassifier (Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)})...")

    # Configuración del modelo XGBoost para clasificación binaria
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # Evaluación en Test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n[RESULTADOS XGBOOST EN TEST SET]")
    print(f" Accuracy:  {acc * 100:.2f}%")
    print(f" Precision: {prec * 100:.2f}%")
    print(f" Recall:    {rec * 100:.2f}%")
    print(f" F1-Score:  {f1 * 100:.2f}%")
    print(f" ROC-AUC:   {auc:.4f}")
    print("\nReporte detallado:\n", classification_report(y_test, y_pred, target_names=['Normal', 'SOP']))

    # Guardar modelo
    ruta_modelo_json = os.path.join(models_dir, "best_xgboost.json")
    model.save_model(ruta_modelo_json)
    
    ruta_modelo_joblib = os.path.join(models_dir, "best_xgboost.pkl")
    joblib.dump(model, ruta_modelo_joblib)

    print(f"[OK] Modelo XGBoost guardado exitosamente en: {ruta_modelo_json} y {ruta_modelo_joblib}")

if __name__ == '__main__':
    entrenar_xgboost()
