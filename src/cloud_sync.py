# cloud_sync.py

import os
import csv
import pickle
import json
import logging
import requests
import time
from datetime import datetime, timedelta
import shutil  # para mover archivos con try/except si se desea

# Intentamos importar las librerías de Google Cloud Platform (GCP)
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    storage = None
    service_account = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class CloudSync:
    """
    Clase que maneja la sincronización de datos y la actualización de modelos
    entre la Raspberry Pi y Google Cloud.

    Funcionalidades:
      - Sube archivos CSV de sensor_data y decision_data al bucket.
      - Sube también cualquier archivo en offline_cache/.
      - Llama a la Cloud Function (train_model) para entrenar remotamente (opcional).
      - Descarga y compara el nuevo modelo con el local.
      - Actualiza el modelo local (con backup del anterior) solo si mejora las métricas.
      - Maneja un versionado básico en model_metadata.json.
      - Permite revertir el modelo local a un backup anterior.
    """

    def __init__(self,
                 credentials_path='config/credentials.json',
                 bucket_name='sistema-riego-datos-admin',
                 offline_cache_dir='data/offline_cache'):
        """
        Inicializa la clase CloudSync con parámetros de conexión a GCP y rutas locales.
        
        :param credentials_path: Ruta al archivo de credenciales (service account JSON).
        :param bucket_name: Nombre del bucket en GCP.
        :param offline_cache_dir: Carpeta local donde se guardan archivos en caso offline.
        """
        self.credentials_path = credentials_path
        self.bucket_name = bucket_name
        self.offline_cache_dir = offline_cache_dir

        self.client = None
        self.bucket = None

        # Intentar inicializar el cliente GCP
        if storage and service_account:
            self._initialize_gcp_client()
        else:
            logging.warning(
                "Librerías de GCP (google.cloud) no disponibles. "
                "Funcionalidad de nube limitada o inoperante."
            )

        # URL de la función Cloud (adaptar si se cambia de región/nombre en GCF)
        self.function_url = "https://us-central1-sistemariegointeligente.cloudfunctions.net/train_model"

    def _initialize_gcp_client(self):
        """
        Inicializa el cliente de Google Cloud Storage con las credenciales dadas.
        Lanza warnings o errores si no se encuentra el archivo de credenciales.
        """
        try:
            if os.path.exists(self.credentials_path):
                creds = service_account.Credentials.from_service_account_file(self.credentials_path)
                self.client = storage.Client(credentials=creds)
                self.bucket = self.client.get_bucket(self.bucket_name)
                logging.info(f"Conectado a bucket GCP: {self.bucket_name}")
            else:
                logging.warning(
                    f"No se encontró archivo de credenciales en: {self.credentials_path}. "
                    "No será posible subir/descargar archivos."
                )
        except DefaultCredentialsError as e:
            logging.error(f"Error de credenciales con GCP: {e}")
        except Exception as e:
            logging.error(f"No se pudo inicializar GCP Client: {e}")

    def verificar_conexion(self):
        """
        Verifica si hay conexión a GCP intentando listar un blob (prueba sencilla).
        Devuelve True si hay conexión y se puede acceder al bucket, False si no.
        """
        if not self.bucket:
            return False
        try:
            # Simplemente listamos 1 blob como prueba de conexión
            _ = list(self.bucket.list_blobs(max_results=1))
            return True
        except Exception as e:
            logging.warning(f"No se pudo verificar conexión GCP: {e}")
            return False

    def sincronizar_con_nube(self, forzar_entrenamiento=True):
        """
        Proceso principal de sincronización:
          1) Subir datos locales (sensor_data, decision_data, offline_cache).
          2) (Opcional) Llamar a la Cloud Function (train_model) para entrenar en la nube.
          3) Descargar y comparar el modelo actualizado.
          4) Si mejora métricas, actualizar modelo local.

        :param forzar_entrenamiento: Si True, se dispara el entrenamiento en la nube.
        :return: True si todo fue bien, False si falló algo o no hay conexión.
        """
        logging.info("Iniciando sincronización con la nube...")

        if not self.verificar_conexion():
            logging.warning("No hay conexión con GCP. Sincronización cancelada.")
            return False

        # 1) Subir datos locales
        self.subir_datos_locales()

        # 2) Entrenamiento en la nube (opcional)
        if forzar_entrenamiento:
            exito_entrenamiento = self.entrenar_modelo_en_nube()
            if not exito_entrenamiento:
                logging.warning("No se pudo completar el entrenamiento en la nube.")
                return False

        # 3) Descargar y comparar modelo
        self.descargar_y_comparar_modelo()
        return True

    def subir_datos_locales(self):
        """
        Sube:
          - sensor_data.csv -> sensor_data/sensor_data.csv
          - decision_data.csv -> decision_data/decision_data.csv
          - Archivos en offline_cache/.
        
        Si el bucket no está inicializado (falla credenciales), no hace nada.
        """
        if not self.bucket:
            logging.warning("Bucket no inicializado; no se suben datos.")
            return

        # Subir sensor_data.csv y decision_data.csv si existen
        for local_file, remote_path in [
            ("data/sensor_data.csv", "sensor_data/sensor_data.csv"),
            ("data/decision_data.csv", "decision_data/decision_data.csv")
        ]:
            self._subir_archivo_con_reintento(local_file, remote_path)

        # Subir archivos en offline_cache
        if os.path.isdir(self.offline_cache_dir):
            for filename in os.listdir(self.offline_cache_dir):
                local_path = os.path.join(self.offline_cache_dir, filename)
                if os.path.isfile(local_path):
                    remote_path = f"offline_cache/{filename}"
                    exito = self._subir_archivo_con_reintento(local_path, remote_path)
                    if exito:
                        # Si se subió con éxito, podemos borrar del cache
                        os.remove(local_path)
        else:
            os.makedirs(self.offline_cache_dir, exist_ok=True)

    def _subir_archivo_con_reintento(self, local_path, remote_path, reintentos=3):
        """
        Sube un archivo al bucket con varios reintentos en caso de error.
        
        :param local_path: Ruta local del archivo.
        :param remote_path: Ruta remota en el bucket.
        :param reintentos: Número de intentos máximos.
        :return: True si logra subir el archivo, False en caso contrario.
        """
        if not os.path.exists(local_path):
            return False

        for intento in range(1, reintentos + 1):
            try:
                blob = self.bucket.blob(remote_path)
                blob.upload_from_filename(local_path)
                logging.info(f"Archivo '{local_path}' subido como '{remote_path}' en GCP.")
                return True
            except Exception as e:
                logging.warning(
                    f"Fallo al subir '{local_path}' a '{remote_path}' (intento {intento}/{reintentos}): {e}"
                )
                time.sleep(3)
        return False

    def entrenar_modelo_en_nube(self):
        """
        Llama a la Cloud Function (cloud_model_training.py) vía HTTP POST
        para iniciar el entrenamiento remoto, con un timeout de 5 minutos (300s).
        
        :return: True si el entrenamiento inició correctamente, False si hubo error.
        """
        if not self.verificar_conexion():
            logging.warning("No hay conexión GCP para entrenar en la nube.")
            return False

        try:
            resp = requests.post(self.function_url, timeout=300)  # 5 min de timeout
            if resp.status_code == 200:
                logging.info(f"Entrenamiento en la nube iniciado con éxito: {resp.text}")
                return True
            else:
                logging.error(f"Error al entrenar en la nube: {resp.status_code} => {resp.text}")
                return False
        except Exception as e:
            logging.error(f"Excepción al llamar a la función de entrenamiento en la nube: {e}")
            return False

    def descargar_y_comparar_modelo(self):
        """
        Descarga:
          - models/modelo_actualizado.pkl
          - models/model_metrics.txt (última línea => MSE_fert, R2_fert, MSE_agua, R2_agua)
          - models/model_metadata.json (opcional)
        
        Compara con el modelo local (basado en métricas locales).
        Si el de la nube es mejor, actualiza localmente y hace backup.
        También actualiza la metadata local.
        """
        if not self.bucket:
            logging.warning("Bucket no inicializado; no se descarga nada.")
            return

        # 1) Descargar y leer metadata de versión (opcional)
        version_nube = None
        metadata_blob = self.bucket.blob("models/model_metadata.json")
        tmp_metadata_path = "data/model_metadata_nube.json"

        if metadata_blob.exists():
            metadata_blob.download_to_filename(tmp_metadata_path)
            try:
                with open(tmp_metadata_path, "r") as f:
                    metadata_nube = json.load(f)
                if "versions" in metadata_nube and len(metadata_nube["versions"]) > 0:
                    version_nube = metadata_nube["versions"][-1].get("version")
                    logging.info(f"Versión más reciente en la nube: {version_nube}")
            except Exception as e:
                logging.warning(f"No se pudo parsear model_metadata.json: {e}")

        version_local = self._cargar_version_local()
        if version_local is not None:
            logging.info(f"Versión local actual: {version_local}")
        else:
            logging.info("No hay versión local guardada (model_metadata_local.json).")

        # 2) Descargar model_metrics.txt para extraer métricas
        metrics_blob = self.bucket.blob("models/model_metrics.txt")
        if not metrics_blob.exists():
            logging.info("No existe model_metrics.txt en la nube; no se puede comparar métricas.")
            return

        remote_metrics = metrics_blob.download_as_string().decode("utf-8").strip().split("\n")
        if not remote_metrics:
            logging.info("El archivo model_metrics.txt está vacío. No se puede comparar métricas.")
            return

        last_line = remote_metrics[-1]
        mse_fert_nube = None
        mse_agua_nube = None
        r2_fert_nube = None
        r2_agua_nube = None

        # Ejemplo de línea: "2025-01-25 10:00:00 => MSE_fert=0.50, R2_fert=0.76, MSE_agua=0.70, R2_agua=0.70"
        try:
            if "=>" in last_line:
                _, metric_str = last_line.split("=>")
                kvs = metric_str.strip().split(",")
                kvs = [kv.strip() for kv in kvs]
                for kv in kvs:
                    k, v = kv.split("=")
                    k = k.strip()
                    v = float(v.strip())
                    if k == "MSE_fert":
                        mse_fert_nube = v
                    elif k == "R2_fert":
                        r2_fert_nube = v
                    elif k == "MSE_agua":
                        mse_agua_nube = v
                    elif k == "R2_agua":
                        r2_agua_nube = v
        except Exception as e:
            logging.warning(f"No se pudo parsear la última línea de model_metrics.txt: {e}")
            # Continúa, pero sin métricas no se puede comparar.

        # 3) Verificar si existe 'modelo_actualizado.pkl' en la nube
        nuevo_modelo_blob = self.bucket.blob("models/modelo_actualizado.pkl")
        if not nuevo_modelo_blob.exists():
            logging.info("No existe 'modelo_actualizado.pkl' en la nube.")
            return

        # 4) Cargar métricas locales
        (mse_fert_local, r2_fert_local,
         mse_agua_local, r2_agua_local) = self._cargar_metricas_locales()

        # Decidir si el modelo nube es mejor
        if (mse_fert_local is None or r2_fert_local is None or
            mse_agua_local is None or r2_agua_local is None):
            # No hay métricas locales => asumimos que el de la nube es mejor
            mejor_nube = True
        else:
            # Chequeo estricto (puede personalizarse)
            mejor_nube = (
                mse_fert_nube is not None and
                mse_agua_nube is not None and
                r2_fert_nube is not None and
                r2_agua_nube is not None and
                mse_fert_nube < mse_fert_local and
                mse_agua_nube < mse_agua_local and
                r2_fert_nube > r2_fert_local and
                r2_agua_nube > r2_agua_local
            )

        if mejor_nube:
            logging.info("El modelo en la nube parece MEJOR. Actualizando localmente...")
            self._actualizar_modelo_local(
                nuevo_modelo_blob, mse_fert_nube, r2_fert_nube, mse_agua_nube, r2_agua_nube
            )

            # Si tenemos metadata con versión en la nube, guardarla
            if version_nube is not None:
                try:
                    with open(tmp_metadata_path, "r") as f:
                        metadata_nube = json.load(f)
                    self._guardar_version_local(version_nube, metadata_nube)
                except Exception as e:
                    logging.error(f"No se pudo guardar metadata local: {e}")
        else:
            logging.info("El modelo en la nube NO mejora respecto al local. Se mantiene el actual.")

    def _actualizar_modelo_local(self, blob_modelo,
                                 mse_fert, r2_fert,
                                 mse_agua, r2_agua):
        """
        Descarga y valida el nuevo modelo. Hace backup del anterior si existe,
        y reemplaza el archivo local. Finalmente actualiza las métricas locales.
        """
        temp_path = "data/modelo_nube_tmp.pkl"
        blob_modelo.download_to_filename(temp_path)

        # Validar con pickle
        try:
            with open(temp_path, "rb") as f:
                data_model = pickle.load(f)
            # Chequeo de estructura mínima
            if ("model" not in data_model or
                "model_fert" not in data_model["model"] or
                "model_agua" not in data_model["model"] or
                "scaler" not in data_model or
                "features" not in data_model):
                raise ValueError("Estructura del modelo descargado no es la esperada.")
        except Exception as e:
            logging.error(f"El modelo descargado está corrupto/incorrecto. Error: {e}")
            return

        modelo_local_path = "data/modelo_actualizado.pkl"
        backup_name = f"data/modelo_anterior_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        # Backup si existe modelo local
        if os.path.exists(modelo_local_path):
            try:
                os.rename(modelo_local_path, backup_name)
                logging.info(f"Modelo anterior respaldado en: {backup_name}")
            except Exception as e:
                logging.error(f"No se pudo respaldar el modelo anterior: {e}")
                # Aun así, seguimos intentando actualizar

        # Mover tmp al path oficial (podemos usar os.rename o shutil.move)
        try:
            os.rename(temp_path, modelo_local_path)
            logging.info("Modelo local actualizado con la versión de la nube.")
        except Exception as e:
            logging.error(f"No se pudo renombrar {temp_path} a {modelo_local_path}: {e}")
            return

        # Guardar nuevas métricas locales
        self._guardar_metricas_locales(mse_fert, r2_fert, mse_agua, r2_agua)

    def _cargar_metricas_locales(self):
        """
        Lee 'data/model_metrics_local.txt' para recuperar las métricas del último modelo local.
        Retorna (mse_fert, r2_fert, mse_agua, r2_agua).
        """
        metrics_file = "data/model_metrics_local.txt"
        if not os.path.exists(metrics_file):
            return (None, None, None, None)

        try:
            with open(metrics_file, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if not lines:
                    return (None, None, None, None)
                # Tomamos la última línea
                last_line = lines[-1]
                # Formato: "YYYY-MM-DD HH:MM:SS => MSE_fert=0.50, R2_fert=0.76, MSE_agua=0.70, R2_agua=0.70"
                if "=>" not in last_line:
                    return (None, None, None, None)

                parts = last_line.split("=>")
                kv_str = parts[1].strip() if len(parts) > 1 else ""
                kvs = [x.strip() for x in kv_str.split(",")]
                parsed = {}
                for kv in kvs:
                    if "=" in kv:
                        k, v = kv.split("=")
                        parsed[k.strip()] = float(v.strip())

                return (
                    parsed.get("MSE_fert"),
                    parsed.get("R2_fert"),
                    parsed.get("MSE_agua"),
                    parsed.get("R2_agua"),
                )
        except Exception as e:
            logging.warning(f"No se pudo cargar métricas locales: {e}")
            return (None, None, None, None)

    def _guardar_metricas_locales(self, mse_fert, r2_fert, mse_agua, r2_agua):
        """
        Agrega (append) una línea a 'data/model_metrics_local.txt' con las métricas.
        """
        os.makedirs("data", exist_ok=True)
        metrics_file = "data/model_metrics_local.txt"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (f"{now_str} => MSE_fert={mse_fert:.4f}, R2_fert={r2_fert:.4f}, "
                f"MSE_agua={mse_agua:.4f}, R2_agua={r2_agua:.4f}\n")
        try:
            with open(metrics_file, "a") as f:
                f.write(line)
            logging.info("Métricas locales del modelo actualizadas.")
        except Exception as e:
            logging.error(f"No se pudo guardar métricas locales: {e}")

    # -------------------------------------------------------------------------
    # Manejo de versionado local en un archivo JSON
    # -------------------------------------------------------------------------
    def _cargar_version_local(self):
        """
        Retorna la última versión (int) guardada en 'data/model_metadata_local.json'
        o None si no existe o si no se puede parsear.
        """
        local_metadata_file = "data/model_metadata_local.json"
        if not os.path.exists(local_metadata_file):
            return None

        try:
            with open(local_metadata_file, "r") as f:
                local_data = json.load(f)
            if "versions" in local_data and len(local_data["versions"]) > 0:
                return local_data["versions"][-1].get("version")
        except Exception as e:
            logging.warning(f"No se pudo parsear model_metadata_local.json: {e}")
        return None

    def _guardar_version_local(self, version_nube, metadata=None):
        """
        Almacena la versión de la nube en 'data/model_metadata_local.json'.
        
        :param version_nube: Número de versión (int) a guardar.
        :param metadata: Diccionario completo, si deseas sobrescribir todo.
        """
        local_metadata_file = "data/model_metadata_local.json"
        os.makedirs("data", exist_ok=True)

        try:
            if metadata is not None:
                # Guardar el JSON entero proveniente de la nube
                with open(local_metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
            else:
                # Simplemente añadir/actualizar la versión
                if os.path.exists(local_metadata_file):
                    with open(local_metadata_file, "r") as f:
                        local_data = json.load(f)
                else:
                    local_data = {"versions": []}

                new_entry = {
                    "version": version_nube,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                local_data["versions"].append(new_entry)

                with open(local_metadata_file, "w") as f:
                    json.dump(local_data, f, indent=2)

            logging.info(f"Guardada versión local: {version_nube}")
        except Exception as e:
            logging.error(f"No se pudo guardar versión local: {e}")

    # -------------------------------------------------------------------------
    # Revertir modelo local
    # -------------------------------------------------------------------------
    def revertir_modelo_local(self, backup_filename=None):
        """
        Revierte el modelo local al backup especificado. Si no se pasa backup_filename,
        se elige el más reciente (alfabéticamente descendente).
        
        Ejemplo de backup: modelo_anterior_YYYYmmDD_HHMMSS.pkl
        """
        os.makedirs("data", exist_ok=True)
        backups = [
            f for f in os.listdir("data")
            if f.startswith("modelo_anterior_") and f.endswith(".pkl")
        ]
        if not backups:
            logging.warning("No hay backups de modelo para revertir.")
            return

        if backup_filename is None:
            backups_sorted = sorted(backups, reverse=True)
            backup_filename = backups_sorted[0]

        backup_path = os.path.join("data", backup_filename)
        if not os.path.exists(backup_path):
            logging.warning(f"No se encuentra el backup especificado: {backup_path}")
            return

        # Respaldar modelo actual
        modelo_local_path = "data/modelo_actualizado.pkl"
        if os.path.exists(modelo_local_path):
            revert_path = f"data/modelo_revertido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.rename(modelo_local_path, revert_path)
            logging.info(f"Respaldo del modelo actual en: {revert_path}")

        # Reemplazar con el backup
        os.rename(backup_path, modelo_local_path)
        logging.info(f"Modelo revertido a partir de: {backup_filename}.")
