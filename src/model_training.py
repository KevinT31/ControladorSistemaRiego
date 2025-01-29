# model_training.py

import os
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor

logger = logging.getLogger('model_training')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class ModelTraining:
    """
    Clase para entrenar un modelo de Machine Learning que predice dos salidas:
      - porcentaje_fertilizante
      - cantidad_agua

    Este modelo se basa en datos de los sensores y del estado del sistema
    (humedad, temperatura, pH, CE, water_level, flow_rate, etc.),
    así como la estación 'season' y otras variables que se deseen agregar.

    Mejoras aplicadas:
      - Manejo opcional de EDA (para evitar problemas en entornos sin interfaz gráfica).
      - Se pueden guardar las gráficas en /tmp en lugar de mostrarlas directamente con plt.show().
      - Manejo opcional de columnas adicionales como 'veces_fuera_umbrales' (si existiesen).
      - Se mantiene el enfoque de entrenamiento offline/local, que se puede ampliar para un 'forecast' a 2 semanas,
        integrando nuevas columnas o lógicas de modelado.
    """

    def __init__(self,
                 model_path="data/modelo_actualizado.pkl",
                 data_path="data/decision_data.csv",
                 perform_eda=True,
                 save_plots=True):
        """
        :param model_path: Ruta donde se guarda el modelo entrenado (archivo .pkl).
        :param data_path: Ruta al CSV con datos históricos de entrenamiento.
        :param perform_eda: Si se debe realizar EDA o no.
        :param save_plots: Si se deben guardar los gráficos en /tmp en vez de mostrarlos por pantalla.
                           En entornos sin display, la opción de mostrar (plt.show()) podría dar errores.
        """
        self.model_path = model_path
        self.data_path = data_path

        # Aquí guardaremos los dos modelos (fertilizante y agua) dentro de un dict
        self.model = None
        # Scaler para todas las features
        self.scaler = None
        # Lista de features (columnas finales tras dummies y transformaciones)
        self.features = None

        # Opciones de EDA
        self.perform_eda = perform_eda
        self.save_plots = save_plots

        # Si no hay display disponible y se quisieran mostrar plots, evitamos error usando backend 'Agg':
        if not os.environ.get('DISPLAY'):
            matplotlib.use('Agg')

    def cargar_datos(self):
        """
        Carga los datos desde data_path, los limpia de outliers,
        hace encoding de la columna 'season', escala las características
        y retorna (X_scaled, y).

        Además, si existiera una columna opcional como 'veces_fuera_umbrales',
        la incorpora también a X, para permitir su uso en el entrenamiento.

        :return: (X_scaled, y) o (None, None) si ocurre un error.
        """
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Datos cargados exitosamente desde {self.data_path}. Tamaño: {data.shape}")

            # Columnas mínimas requeridas
            required_columns = [
                'humidity', 'temperature', 'ph', 'ce', 'water_level', 'flow_rate',
                'porcentaje_fertilizante', 'cantidad_agua', 'season'
            ]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Faltan columnas requeridas en {self.data_path}: {missing_columns}")
                return None, None

            # Eliminar filas con valores faltantes en estas columnas
            data.dropna(subset=required_columns, inplace=True)
            logger.info(f"Datos tras dropna: {data.shape}")

            # ADVERTENCIA: IsolationForest con contamination=0.05 filtra un 5% de outliers
            # Si el dataset es muy pequeño, podría eliminar demasiados datos.
            iso_columns = ['humidity', 'temperature', 'ph', 'ce', 'water_level', 'flow_rate']
            iso = IsolationForest(contamination=0.05, random_state=42)
            iso_preds = iso.fit_predict(data[iso_columns])
            mask = iso_preds != -1  # Mantener solo instancias no marcadas como outliers
            data = data[mask]
            logger.info(f"Datos tras IsolationForest (outliers removidos): {data.shape}")

            # Separar X e y
            X = data[['humidity', 'temperature', 'ph', 'ce', 'water_level', 'flow_rate', 'season']]
            y = data[['porcentaje_fertilizante', 'cantidad_agua']]

            # Si existe una columna opcional 'veces_fuera_umbrales', la añadimos como feature
            if 'veces_fuera_umbrales' in data.columns:
                X['veces_fuera_umbrales'] = data['veces_fuera_umbrales']
                logger.info("Incluyendo 'veces_fuera_umbrales' como feature adicional.")

            # Convertir 'season' a variables dummy
            X = pd.get_dummies(X, columns=['season'], drop_first=False)

            # Guardar features finales
            self.features = X.columns.tolist()

            # Escalar
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Hacer un EDA básico si se solicita
            if self.perform_eda:
                self.realizar_eda(data, X, y)

            return X_scaled, y

        except Exception as e:
            logger.error(f"Error al cargar y procesar datos: {e}")
            return None, None

    def _save_or_show_plot(self, filename):
        """
        Función auxiliar para guardar o mostrar la figura actual, dependiendo
        de la configuración (self.save_plots) y la disponibilidad de DISPLAY.
        """
        if self.save_plots:
            # Crear carpeta /tmp si no existe para evitar errores en entornos minimalistas
            try:
                os.makedirs("/tmp", exist_ok=True)
            except Exception as e:
                logger.warning(f"No se pudo crear /tmp: {e}")

            # Guardar el gráfico en /tmp
            output_path = os.path.join("/tmp", filename)
            plt.savefig(output_path)
            logger.info(f"Gráfico guardado en {output_path}")
        else:
            # Intentar mostrar en pantalla
            if os.environ.get('DISPLAY'):
                plt.show()
            else:
                logger.warning("No hay display disponible, se omitirá plt.show()")
        plt.close()

    def realizar_eda(self, data, X, y):
        """
        Realiza un análisis exploratorio de datos básico:
         - Matriz de correlación
         - Scatter plots contra las variables objetivo

        Si no se puede mostrar en pantalla (entorno sin interfaz),
        se guardarán las gráficas en /tmp/ si self.save_plots es True.
        """
        try:
            # Unir X e y para ver correlaciones
            data_eda = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

            plt.figure(figsize=(12, 10))
            corr_matrix = data_eda.corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Matriz de Correlación (Características y Variables Objetivo)")
            plt.tight_layout()
            self._save_or_show_plot("correlacion.png")

            # Dispersión de sensores vs targets
            sensor_columns = ['humidity', 'temperature', 'ph', 'ce', 'water_level', 'flow_rate']
            target_columns = ['porcentaje_fertilizante', 'cantidad_agua']

            for target in target_columns:
                plt.figure(figsize=(15, 8))
                for i, sensor in enumerate(sensor_columns):
                    plt.subplot(2, 3, i + 1)
                    if sensor in data_eda.columns:
                        sns.scatterplot(data=data_eda, x=sensor, y=target, alpha=0.7)
                    plt.title(f"{sensor} vs {target}")
                plt.tight_layout()
                self._save_or_show_plot(f"dispersion_{target}.png")

        except Exception as e:
            logger.warning(f"Error durante EDA: {e}")

    def entrenar_modelo_local(self):
        """
        Entrena un modelo XGBRegressor (dos salidas) con los datos locales
        y guarda el mejor modelo en self.model_path.
        """
        X, y = self.cargar_datos()
        if X is None or y is None:
            logger.error("No se puede entrenar el modelo por falta de datos.")
            return

        try:
            # Dividir en training y test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Búsqueda aleatoria de hiperparámetros
            param_dist = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2]
            }

            # Para problemas multi-output, entrenamos un XGB por cada salida.
            y_fert = y_train['porcentaje_fertilizante']
            y_agua = y_train['cantidad_agua']

            xgb_base = XGBRegressor(
                random_state=42,
                objective='reg:squarederror'
            )

            # RandomizedSearchCV para el primer modelo (porcentaje_fertilizante)
            random_search_fert = RandomizedSearchCV(
                xgb_base,
                param_distributions=param_dist,
                n_iter=30,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            logger.info("Buscando hiperparámetros para porcentaje_fertilizante...")
            random_search_fert.fit(X_train, y_fert)
            best_model_fert = random_search_fert.best_estimator_
            logger.info(f"Mejores hiperparámetros (fertilizante): {random_search_fert.best_params_}")

            # RandomizedSearchCV para el segundo modelo (cantidad_agua)
            random_search_agua = RandomizedSearchCV(
                xgb_base,
                param_distributions=param_dist,
                n_iter=30,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            logger.info("Buscando hiperparámetros para cantidad_agua...")
            random_search_agua.fit(X_train, y_agua)
            best_model_agua = random_search_agua.best_estimator_
            logger.info(f"Mejores hiperparámetros (agua): {random_search_agua.best_params_}")

            # Guardar ambos modelos en un diccionario
            self.model = {
                'model_fert': best_model_fert,
                'model_agua': best_model_agua
            }

            # Validación cruzada (simple)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores_fert = cross_val_score(best_model_fert, X_train, y_fert, cv=kf, scoring='neg_mean_squared_error')
            scores_agua = cross_val_score(best_model_agua, X_train, y_agua, cv=kf, scoring='neg_mean_squared_error')
            logger.info(f"[Fertilizante] MSE medio CV: {-np.mean(scores_fert):.4f}")
            logger.info(f"[Agua]         MSE medio CV: {-np.mean(scores_agua):.4f}")

            # Evaluar en X_test
            y_fert_test = y_test['porcentaje_fertilizante'].values
            y_agua_test = y_test['cantidad_agua'].values

            fert_pred = best_model_fert.predict(X_test)
            agua_pred = best_model_agua.predict(X_test)

            mse_fert = mean_squared_error(y_fert_test, fert_pred)
            mse_agua = mean_squared_error(y_agua_test, agua_pred)
            r2_fert = r2_score(y_fert_test, fert_pred)
            r2_agua = r2_score(y_agua_test, agua_pred)
            logger.info(f"[Fertilizante] MSE: {mse_fert:.4f}, R2: {r2_fert:.4f}")
            logger.info(f"[Agua]         MSE: {mse_agua:.4f}, R2: {r2_agua:.4f}")

            # Visualizar resultados
            self.visualizar_resultados(y_fert_test, fert_pred, "Porcentaje Fertilizante")
            self.visualizar_resultados(y_agua_test, agua_pred, "Cantidad Agua")

            # Guardar en disco el dict con los modelos, scaler y features
            self.guardar_modelo()

        except Exception as e:
            logger.error(f"Error durante el entrenamiento del modelo: {e}")

    def visualizar_resultados(self, y_true, y_pred, titulo):
        """
        Visualiza (o guarda) valor real vs predicción para una variable,
        dependiendo de la configuración (save_plots).
        """
        try:
            plt.figure(figsize=(6, 5))
            plt.scatter(y_true, y_pred, alpha=0.7, c='blue')
            plt.xlabel("Valor Real")
            plt.ylabel("Predicción")
            plt.title(f"{titulo}: Real vs Predicción")
            mini = min(min(y_true), min(y_pred))
            maxi = max(max(y_true), max(y_pred))
            plt.plot([mini, maxi], [mini, maxi], 'r--')
            plt.tight_layout()

            # Guardamos o mostramos
            fname = f"resultado_{titulo.replace(' ', '_')}.png"
            self._save_or_show_plot(fname)

        except Exception as e:
            logger.warning(f"No se pudo visualizar/guardar resultados para {titulo}: {e}")

    def guardar_modelo(self):
        """
        Guarda en self.model_path un dict con:
         {
           'model': dict con {'model_fert':..., 'model_agua':...},
           'scaler': self.scaler,
           'features': self.features
         }
        """
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'features': self.features
                }, f)
            logger.info(f"Modelo(s) entrenado(s) guardado(s) en {self.model_path}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")

    def cargar_modelo(self):
        """
        Carga el modelo (dos XGBRegressors) + scaler + features.
        """
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.features = model_data['features']
            logger.info("Modelo cargado exitosamente desde archivo local.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            self.model = None
            self.scaler = None
            self.features = None

    def predecir(self, sensor_values):
        """
        Realiza predicciones con los dos modelos:
          - Porcentaje fertilizante
          - Cantidad de agua

        :param sensor_values: dict con valores {
           'humidity':..., 'temperature':..., 'ph':..., 'ce':..., 
           'water_level':..., 'flow_rate':..., 'season':...
        }
        :return: dict con {'porcentaje_fertilizante': val, 'cantidad_agua': val} o None si error.
        """
        if (self.model is None or
            not isinstance(self.model, dict) or
            'model_fert' not in self.model or
            'model_agua' not in self.model or
            self.scaler is None or
            self.features is None):
            logger.error("No hay modelos disponibles para predecir (model_fert / model_agua).")
            return None

        try:
            # Crear DataFrame de 1 fila
            df_input = pd.DataFrame([sensor_values])

            # Manejo de 'season'
            if 'season' in df_input.columns:
                df_input = pd.get_dummies(df_input, columns=['season'], drop_first=False)
            else:
                logger.warning("No se encuentra la columna 'season' en sensor_values. Se asignarán 0 a dummies.")

            # Agregar columnas faltantes con 0
            for col in self.features:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[self.features]

            # Escalar
            X_scaled = self.scaler.transform(df_input)

            # Predecir en ambos modelos
            fert_pred = self.model['model_fert'].predict(X_scaled)[0]
            agua_pred = self.model['model_agua'].predict(X_scaled)[0]

            # Se aplica max(0, ...) para evitar resultados negativos por pequeñas desviaciones
            return {
                'porcentaje_fertilizante': max(0, fert_pred),
                'cantidad_agua': max(0, agua_pred)
            }

        except Exception as e:
            logger.error(f"Error al predecir con los modelos: {e}")
            return None


if __name__ == "__main__":
    # Ejemplo de uso:
    trainer = ModelTraining(
        model_path="data/modelo_actualizado.pkl",
        data_path="data/decision_data.csv",
        perform_eda=True,
        save_plots=True
    )
    trainer.entrenar_modelo_local()