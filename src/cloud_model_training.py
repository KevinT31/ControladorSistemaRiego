# cloud_model_training.py

import functions_framework
import os
import logging
import pickle
import json
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

# Scikit-Learn y XGBoost
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Google Cloud
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Umbral mínimo de muestras para entrenar
MINIMUM_REQUIRED_SAMPLES = 100


def _train_single_model(X_train, y_train, xgb_model, param_dist, model_name="model"):
    """
    Aplica RandomizedSearchCV para un único modelo XGBRegressor. Retorna:
      - El mejor estimador entrenado.
      - Los hiperparámetros óptimos.
    
    :param X_train: Matriz de entrenamiento (features).
    :param y_train: Vector (o serie) de la variable objetivo.
    :param xgb_model: Instancia base de XGBRegressor.
    :param param_dist: Diccionario de hiperparámetros para búsqueda.
    :param model_name: Nombre descriptivo para logs.
    :return: (best_estimator, best_params)
    """
    logging.info(f"Buscando hiperparámetros para {model_name}...")
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_estimator = random_search.best_estimator_
    logging.info(f"Mejores hiperparámetros ({model_name}): {random_search.best_params_}")
    return best_estimator, random_search.best_params_


def _is_new_model_better(mse_fert, mse_agua, r2_fert, r2_agua,
                         old_mse_fert, old_mse_agua, old_r2_fert, old_r2_agua):
    """
    Decide si el nuevo modelo es mejor que el anterior con un enfoque más flexible.
    Se considera mejor si:
    - Mejora al menos 3 de las 4 métricas, o
    - Cumple con un puntaje combinado ajustable (si se desea incluir).

    Criterios de mejora:
     - MSE_fert y MSE_agua deben disminuir.
     - R2_fert y R2_agua deben aumentar.
    """
    if (old_mse_fert is None or
        old_mse_agua is None or
        old_r2_fert is None or
        old_r2_agua is None):
        # Si no hay modelo anterior o métricas de referencia, se considera mejor.
        return True

    # Inicializamos un contador de métricas mejoradas
    improved_metrics = 0

    # Verificamos las métricas individuales
    if mse_fert < old_mse_fert:
        improved_metrics += 1
    if mse_agua < old_mse_agua:
        improved_metrics += 1
    if r2_fert > old_r2_fert:
        improved_metrics += 1
    if r2_agua > old_r2_agua:
        improved_metrics += 1

    # Se considera mejor si mejora al menos 3 de las 4 métricas
    if improved_metrics >= 3:
        return True

    # Si se desea incluir una lógica de puntaje combinado (opcional)
    # Ejemplo: combinar mejoras en MSE y R2 con diferentes ponderaciones
    combined_score_old = (1 / old_mse_fert + 1 / old_mse_agua) + (old_r2_fert + old_r2_agua)
    combined_score_new = (1 / mse_fert + 1 / mse_agua) + (r2_fert + r2_agua)
    if combined_score_new > combined_score_old:
        return True

    return False

@functions_framework.http
def train_model(request):
    """
    Función HTTP de Google Cloud para entrenar dos modelos de ML que predicen:
      - porcentaje_fertilizante
      - cantidad_agua
    
    Flujo general:
      1) Lee datos históricos (sensor_data y decision_data) desde Cloud Storage.
      2) Fusiona y limpia (outliers, NaNs, duplicados, etc.).
      3) Aplica dummies a 'season', escalado (MinMaxScaler), y RandomizedSearchCV (XGB).
      4) Entrena dos modelos: uno para fertilizante y otro para agua.
      5) Evalúa métricas (MSE, R²).
      6) Compara con el modelo anterior usando 'model_metrics.txt'.
      7) Si el nuevo modelo mejora, se guarda y se actualizan 'model_metrics.txt' y 'model_metadata.json'.
      8) Si no mejora, se descarta.
    """
    bucket_name = "sistema-riego-datos-admin"

    # Intento de conectar a GCP
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
    except DefaultCredentialsError as e:
        logging.error(f"Error de autenticación: {e}")
        return ("Error de autenticación con GCP.", 500)
    except Exception as e:
        logging.error(f"Error al inicializar cliente de GCP: {e}")
        return (f"Error al inicializar GCP: {e}", 500)

    try:
        # (1) Descarga y unión de datos
        logging.info("Descargando CSVs de sensor_data/ y decision_data/ desde Cloud Storage...")
        all_dfs = []
        for prefix_dir in ["sensor_data/", "decision_data/"]:
            blobs = bucket.list_blobs(prefix=prefix_dir)
            for blob in blobs:
                if blob.name.endswith(".csv"):
                    content = blob.download_as_string().decode("utf-8")
                    df_temp = pd.read_csv(StringIO(content))
                    all_dfs.append(df_temp)

        if not all_dfs:
            msg = "No se encontraron archivos CSV de sensor_data/ ni decision_data/ en la nube."
            logging.warning(msg)
            return (msg, 200)

        data = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        logging.info(f"Datos combinados: {data.shape}")

        # (2) Limpieza de columnas relevantes
        required_cols = [
            "humidity", "temperature", "ph", "ce",
            "water_level", "flow_rate", "season",
            "porcentaje_fertilizante", "cantidad_agua"
        ]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            msg = f"Faltan columnas en los datos: {missing_cols}"
            logging.error(msg)
            return (msg, 500)

        data.dropna(subset=required_cols, inplace=True)
        if data.empty:
            msg = "Los datos se quedaron vacíos tras eliminar NaNs."
            logging.error(msg)
            return (msg, 200)

        if len(data) < MINIMUM_REQUIRED_SAMPLES:
            msg = f"No hay suficientes muestras para entrenar (mínimo {MINIMUM_REQUIRED_SAMPLES})."
            logging.warning(msg)
            return (msg, 200)

        # (3) Manejo de outliers con IsolationForest
        iso_cols = ["humidity", "temperature", "ph", "ce", "water_level", "flow_rate"]
        iforest = IsolationForest(contamination=0.05, random_state=42)
        outlier_preds = iforest.fit_predict(data[iso_cols])
        mask = outlier_preds != -1
        data = data[mask]
        logging.info(f"Datos tras filtrar outliers: {data.shape}")

        if len(data) < MINIMUM_REQUIRED_SAMPLES:
            msg = "Quedaron muy pocos datos tras filtrar outliers."
            logging.warning(msg)
            return (msg, 200)

        # (4) Preparación final de features
        X = data[["humidity", "temperature", "ph", "ce", "water_level", "flow_rate", "season"]]
        y = data[["porcentaje_fertilizante", "cantidad_agua"]]

        X = pd.get_dummies(X, columns=["season"], drop_first=False)
        feature_cols = X.columns.tolist()

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # (5) Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        y_train_fert = y_train["porcentaje_fertilizante"]
        y_train_agua = y_train["cantidad_agua"]
        y_test_fert = y_test["porcentaje_fertilizante"].values
        y_test_agua = y_test["cantidad_agua"].values

        # (6) Entrenar XGB (dos modelos) con factorización
        param_dist = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [1, 1.5, 2]
        }

        xgb_base = XGBRegressor(random_state=42, objective="reg:squarederror")

        # Modelo para fertilizante
        model_fert, fert_params = _train_single_model(
            X_train, y_train_fert, xgb_base, param_dist, "porcentaje_fertilizante"
        )

        # Modelo para agua
        model_agua, agua_params = _train_single_model(
            X_train, y_train_agua, xgb_base, param_dist, "cantidad_agua"
        )

        # (7) Evaluar los modelos
        fert_pred_test = model_fert.predict(X_test)
        agua_pred_test = model_agua.predict(X_test)

        mse_fert = mean_squared_error(y_test_fert, fert_pred_test)
        mse_agua = mean_squared_error(y_test_agua, agua_pred_test)
        r2_fert = r2_score(y_test_fert, fert_pred_test)
        r2_agua = r2_score(y_test_agua, agua_pred_test)

        logging.info(f"[Fertilizante] MSE: {mse_fert:.4f}, R2: {r2_fert:.4f}")
        logging.info(f"[Agua]         MSE: {mse_agua:.4f}, R2: {r2_agua:.4f}")

        # (8) Comparar con modelo anterior
        mejor_que_anterior = True
        old_mse_fert = None
        old_mse_agua = None
        old_r2_fert = None
        old_r2_agua = None

        metrics_blob = bucket.blob("models/model_metrics.txt")
        if metrics_blob.exists():
            old_metrics_str = metrics_blob.download_as_string().decode("utf-8")
            lines = old_metrics_str.strip().split("\n")
            if lines:
                last_line = lines[-1]
                # Ejemplo de formato:
                # "2025-01-25 10:00:00 => MSE_fert=0.50, R2_fert=0.76, MSE_agua=0.70, R2_agua=0.70"
                try:
                    parts = last_line.split("=>")
                    if len(parts) == 2:
                        metric_part = parts[1].strip()
                        kvs = metric_part.split(",")
                        kvs = [kv.strip() for kv in kvs]
                        for kv in kvs:
                            k, v = kv.split("=")
                            k = k.strip()
                            v = float(v.strip())
                            if k == "MSE_fert":
                                old_mse_fert = v
                            elif k == "R2_fert":
                                old_r2_fert = v
                            elif k == "MSE_agua":
                                old_mse_agua = v
                            elif k == "R2_agua":
                                old_r2_agua = v

                except Exception:
                    logging.warning(
                        "No se pudo parsear correctamente la última línea de model_metrics.txt. "
                        "Se asume que el nuevo modelo puede ser mejor."
                    )

        # Llamamos a la función de comparación
        mejor_que_anterior = _is_new_model_better(
            mse_fert, mse_agua, r2_fert, r2_agua,
            old_mse_fert, old_mse_agua, old_r2_fert, old_r2_agua
        )

        if not mejor_que_anterior:
            msg = "El nuevo modelo NO mejoró las métricas previas. Se descarta."
            logging.warning(msg)
            return (msg, 200)

        # (9) Guardar el nuevo modelo
        model_data = {
            "model": {
                "model_fert": model_fert,
                "model_agua": model_agua
            },
            "scaler": scaler,
            "features": feature_cols
        }

        tmp_model_path = "/tmp/modelo_actualizado.pkl"
        with open(tmp_model_path, "wb") as f:
            pickle.dump(model_data, f)

        model_blob = bucket.blob("models/modelo_actualizado.pkl")
        model_blob.upload_from_filename(tmp_model_path)
        logging.info("Nuevo modelo guardado en 'models/modelo_actualizado.pkl'.")

        # (10) Registrar métricas en model_metrics.txt
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_line = (
            f"{now_str} => "
            f"MSE_fert={mse_fert:.4f}, R2_fert={r2_fert:.4f}, "
            f"MSE_agua={mse_agua:.4f}, R2_agua={r2_agua:.4f}\n"
        )

        local_metrics_file = "/tmp/model_metrics.txt"
        mode = "a"
        if metrics_blob.exists():
            metrics_blob.download_to_filename(local_metrics_file)
        else:
            mode = "w"

        with open(local_metrics_file, mode) as f:
            f.write(new_line)

        metrics_blob.upload_from_filename(local_metrics_file)
        logging.info("Métricas actualizadas en models/model_metrics.txt.")

        # (11) Manejo de versionado en 'model_metadata.json'
        model_metadata_blob = bucket.blob("models/model_metadata.json")
        metadata = {"versions": []}

        if model_metadata_blob.exists():
            try:
                old_json = model_metadata_blob.download_as_string().decode("utf-8")
                metadata = json.loads(old_json)
                if "versions" not in metadata:
                    metadata["versions"] = []
            except Exception as e:
                logging.warning(f"No se pudo parsear model_metadata.json. Se inicia de cero: {e}")

        if metadata["versions"]:
            last_version = metadata["versions"][-1].get("version", 0)
            new_version = last_version + 1
        else:
            new_version = 1

        new_entry = {
            "version": new_version,
            "timestamp": now_str,
            "mse_fert": mse_fert,
            "r2_fert": r2_fert,
            "mse_agua": mse_agua,
            "r2_agua": r2_agua
        }
        metadata["versions"].append(new_entry)

        tmp_metadata_path = "/tmp/model_metadata.json"
        with open(tmp_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        model_metadata_blob.upload_from_filename(tmp_metadata_path)
        logging.info(f"model_metadata.json actualizado con version={new_version}.")

        return ("Entrenamiento completado y modelo actualizado.", 200)

    except Exception as e:
        logging.exception("Error durante el entrenamiento en la nube:")
        return (f"Error durante el entrenamiento: {e}", 500)
