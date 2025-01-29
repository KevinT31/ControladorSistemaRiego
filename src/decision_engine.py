# decision_engine.py
# -------------------------------------------------------------------------------
# Motor de toma de decisiones del sistema de riego.
# Integra:
#  - Modelo de ML (random forest, etc.) con 2 salidas: porcentaje_fertilizante, cantidad_agua
#  - Reglas preestablecidas cuando se sale de los umbrales (pH, CE, N, P, K, etc.)
#  - Ajuste según estación (summer, autumn, winter, spring)
#  - Modo manual sugerido: generar_sugerencias()
#  - Día de fertilizante: correcciones específicas en pH, CE, N, P, K
# -------------------------------------------------------------------------------

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class DecisionEngine:
    """
    Clase principal del motor de decisiones. Utiliza dos modelos de ML (regresores):
    - model_fert: predice porcentaje de fertilizante
    - model_agua: predice cantidad de agua
    También contempla reglas de umbrales para corregir pH, CE, N, P, K, etc.
    """

    def __init__(self, model_path: str = "data/modelo_actualizado.pkl", gui_interface=None):
        """
        :param model_path: Ruta del archivo con los modelos entrenados (pickle).
        :param gui_interface: (opcional) Referencia a NodeRedInterface para mostrar alertas.
        """
        self.model_path = model_path

        # Dos atributos para los modelos de fertilizante y agua
        self.model_fert = None
        self.model_agua = None

        # Scalers y listas de features (pueden ser distintos para cada modelo)
        self.scaler_fert = None
        self.scaler_agua = None
        self.features_fert = None
        self.features_agua = None

        # Umbrales por estación (ver _definir_umbrales_por_estacion)
        self.umbrales: Dict[str, Dict[str, float]] = {}

        # Contador de veces fuera de umbrales (para estadísticas o ajustes futuros)
        self.veces_fuera_umbrales = 0

        # Interfaz a Node-RED (si la quieres para mandar alertas)
        self.gui_interface = gui_interface

        # Cargar modelos si existen
        self.cargar_modelo()

        logging.info("DecisionEngine inicializado.")

    # --------------------------------------------------------------------------
    # Función auxiliar para enviar mensajes a Node-RED (o log) 
    # --------------------------------------------------------------------------
    def _nodered_mensaje(self, texto: str) -> None:
        """
        Envía un mensaje de alerta a Node-RED, o un logging si Node-RED no está disponible.
        """
        if self.gui_interface and getattr(self.gui_interface, 'node_red_available', False):
            self.gui_interface.mostrar_sugerencias({"alerta": texto})
        else:
            logging.info(f"[ALERTA] {texto}")

    # --------------------------------------------------------------------------
    # 1) Carga y actualización del modelo
    # --------------------------------------------------------------------------
    def cargar_modelo(self) -> None:
        """
        Carga los modelos (y sus scalers, features) desde self.model_path.
        Estructura del pickle asumida:
            {
                'model': {
                    'model_fert': ...,
                    'model_agua': ...
                },
                'scaler': ...,
                'features': ...
            }
        Se sobreentiende que la misma 'scaler' y 'features' aplican
        a ambos modelos (fert y agua).
        """
        if not os.path.isfile(self.model_path):
            logging.warning(f"Archivo de modelo no encontrado: {self.model_path}. Usando solo umbrales.")
            return

        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                # Ajustado para la nueva estructura
                self.model_fert = data['model']['model_fert']
                self.model_agua = data['model']['model_agua']
                self.scaler_fert = data['scaler']
                self.scaler_agua = data['scaler']
                self.features_fert = data['features']
                self.features_agua = data['features']

            logging.info(f"Modelos ML (fert y agua) cargados desde {self.model_path}.")
        except Exception as e:
            logging.exception(f"Error al cargar modelos: {e}. Se usarán solo reglas de umbrales.")
            self.model_fert = None
            self.model_agua = None
            self.scaler_fert = None
            self.scaler_agua = None
            self.features_fert = None
            self.features_agua = None

    def actualizar_modelo(self) -> None:
        """
        Vuelve a cargar los modelos, p. ej. tras entrenamiento.
        """
        logging.info("Recargando modelos de DecisionEngine...")
        self.cargar_modelo()

    # --------------------------------------------------------------------------
    # 2) Método principal: evaluar(sensor_values)
    # --------------------------------------------------------------------------
    def evaluar(self, sensor_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determina la acción de riego/fertilización:
          - Verificación de umbrales (pH, CE, humidity, temp, N, P, K).
          - Corrección escalonada si temp/hum está fuera.
          - Si pH, CE, N, P, K muy fuera => acción emergente.
          - Si todo OK => usar ML si está disponible; fallback si no.
          - Se incrementa self.veces_fuera_umbrales si se sale de umbrales.

        :param sensor_values: dict con keys como 'temperature','humidity','ph','ce',
                              'N','P','K','water_level','flow_rate','season', etc.
        :return: dict con la acción recomendada (activar bomba, abrir válvulas, etc.)
        """
        logging.info("DecisionEngine.evaluar => Recibiendo sensor_values...")

        # Ajuste de umbrales por estación
        season = sensor_values.get('season', 'unknown')
        self.umbrales = self._definir_umbrales_por_estacion(season)

        dia_fertil = sensor_values.get('dia_fertilizante', False)

        # Verifica umbrales y obtiene posible corrección escalonada
        ok, partial_temp_hum = self._estan_dentro_umbrales_cascada(sensor_values, dia_fertil)

        if not ok:
            # Se sale de umbrales => acción emergente
            logging.warning("Valores de pH/CE/N/P/K fuera de umbral => acción emergente.")
            self.veces_fuera_umbrales += 1
            return self._acciones_emergencia_simple()

        if partial_temp_hum is not None:
            # Ajuste escalonado de temp/hum, prioriza la corrección
            logging.warning("Aplicando corrección escalonada temp/hum (sin ML).")
            return partial_temp_hum

        # Si llegamos aquí, pH/CE/ etc. están en rango,
        # y temp/hum no requiere corrección escalonada.
        # => Intentar usar ML
        if any(x is None for x in [
            self.model_fert, self.model_agua,
            self.scaler_fert, self.scaler_agua,
            self.features_fert, self.features_agua
        ]):
            logging.warning("Modelos ML no disponibles => fallback a decisiones_por_defecto.")
            return self._decisiones_por_defecto(sensor_values)
        else:
            try:
                return self._usar_modelo_ml(sensor_values)
            except Exception as e:
                logging.exception(f"Error con modelos ML: {e}")
                return self._decisiones_por_defecto(sensor_values)

    # --------------------------------------------------------------------------
    # 3) Cascada de umbrales (temperature/humidity, pH, CE, N/P/K)
    # --------------------------------------------------------------------------
    def _estan_dentro_umbrales_cascada(
        self,
        sensor_values: Dict[str, Any],
        dia_fertilizante: bool
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verifica en cascada:
         - Temp/hum => corrección escalonada
         - pH => emergente si fuera
         - CE => emergente si fuera
         - N, P, K => emergente si fuera, solo en día fertil

        :return:
         (False, None) => emergente
         (True, None) => sin corrección escalonada
         (True, dict) => corrección escalonada de temp/hum
        """
        temp = sensor_values.get('temperature', sensor_values.get('temp_avg', 25))
        hum = sensor_values.get('humidity', sensor_values.get('humedad_avg', 50))
        ph = sensor_values.get('ph', 6.5)
        ce = sensor_values.get('ce', 1.0)
        n_ = sensor_values.get('N', 30)
        p_ = sensor_values.get('P', 15)
        k_ = sensor_values.get('K', 20)

        # Corrige temp/hum si fuera de rango
        partial_temp_hum = self._acciones_escalonadas_temp_hum(temp, hum)

        # pH
        if ph > self.umbrales['ph']['max']:
            if dia_fertilizante:
                self._nodered_mensaje("pH elevado => corrector ácido (día fertil).")
            else:
                self._nodered_mensaje("pH elevado => corrector ácido (sin ML).")
            return (False, None)
        elif ph < self.umbrales['ph']['min']:
            if dia_fertilizante:
                self._nodered_mensaje("pH bajo => corrector alcalino (día fertil).")
            else:
                self._nodered_mensaje("pH bajo => corrector alcalino (sin ML).")
            return (False, None)

        # CE
        if ce > self.umbrales['ce']['max']:
            if dia_fertilizante:
                self._nodered_mensaje("CE muy alta => no inyectar fertilizante.")
            else:
                self._nodered_mensaje("CE elevada => diluir riego.")
            return (False, None)
        elif ce < self.umbrales['ce']['min']:
            if dia_fertilizante:
                self._nodered_mensaje("CE baja => incrementar fertilizante.")
            else:
                self._nodered_mensaje("CE baja => se sugiere fertilizar.")
            return (False, None)

        # N, P, K => solo emergente en día fertil
        if dia_fertilizante:
            if n_ > self.umbrales['N']['max']:
                self._nodered_mensaje("N alto => reducir fertilizante nitrogenado.")
                return (False, None)
            elif n_ < self.umbrales['N']['min']:
                self._nodered_mensaje("N bajo => aumentar fertilizante nitrogenado.")
                return (False, None)

            if p_ > self.umbrales['P']['max']:
                self._nodered_mensaje("P alto => evitar fertilizantes fosfatados.")
                return (False, None)
            elif p_ < self.umbrales['P']['min']:
                self._nodered_mensaje("P bajo => añadir fertilizante fosfatado.")
                return (False, None)

            if k_ > self.umbrales['K']['max']:
                self._nodered_mensaje("K alto => reducir fertilizante potásico.")
                return (False, None)
            elif k_ < self.umbrales['K']['min']:
                self._nodered_mensaje("K bajo => aumentar fertilizante potásico.")
                return (False, None)

        # Si no hubo emergente => OK
        return (True, partial_temp_hum)

    def _acciones_escalonadas_temp_hum(self, temp: float, hum: float) -> Optional[Dict[str, Any]]:
        """
        Correcciones simples si la temp/hum está fuera de ciertos rangos.
        """
        if temp > 35 and hum < 40:
            return {
                'activar_bomba': True,
                'abrir_valvula_riego': True,
                'inyectar_fertilizante': False,
                'abrir_valvula_suministro': False,
                'porcentaje_fertilizante': 0,
                'cantidad_agua': 5
            }
        elif temp > 35 and hum > 70:
            return {
                'activar_bomba': False,
                'abrir_valvula_riego': False,
                'inyectar_fertilizante': False,
                'abrir_valvula_suministro': False,
                'porcentaje_fertilizante': 0,
                'cantidad_agua': 0
            }
        elif temp < 10 and hum < 30:
            return {
                'activar_bomba': True,
                'abrir_valvula_riego': True,
                'inyectar_fertilizante': False,
                'abrir_valvula_suministro': False,
                'porcentaje_fertilizante': 0,
                'cantidad_agua': 3
            }
        return None

    def _acciones_emergencia_simple(self) -> Dict[str, Any]:
        """
        Acción emergente nula: no riego ni fertilizante.
        """
        return {
            'activar_bomba': False,
            'abrir_valvula_riego': False,
            'inyectar_fertilizante': False,
            'abrir_valvula_suministro': False,
            'porcentaje_fertilizante': 0,
            'cantidad_agua': 0
        }

    # --------------------------------------------------------------------------
    # 4) Lógica ML (usar_modelo_ml)
    # --------------------------------------------------------------------------
    def _usar_modelo_ml(self, sensor_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica los dos modelos ML (fertilizante y agua) para predecir, y 
        luego genera la decisión final. 
        Ejemplo: si dia_fertilizante y CE algo alta => reducir 20% fertilizante.
        """
        df_in = pd.DataFrame([sensor_values])

        # Remover columnas no deseadas
        if 'timestamp' in df_in.columns:
            df_in.drop(columns=['timestamp'], inplace=True)

        # Crear dummies para 'season'
        if 'season' in df_in.columns:
            df_in = pd.get_dummies(df_in, columns=['season'], drop_first=False)

        # Asegurar columnas para fertilizante
        for col in (self.features_fert or []):
            if col not in df_in.columns:
                df_in[col] = 0
        X_fert = self.scaler_fert.transform(df_in[self.features_fert])
        fert_array = self.model_fert.predict(X_fert)
        fert = fert_array[0] if len(fert_array) else 0.0

        # Asegurar columnas para agua
        for col in (self.features_agua or []):
            if col not in df_in.columns:
                df_in[col] = 0
        X_agua = self.scaler_agua.transform(df_in[self.features_agua])
        agua_array = self.model_agua.predict(X_agua)
        agua = agua_array[0] if len(agua_array) else 0.0

        fert = max(0, fert)
        agua = max(0, agua)

        # --------------- Lógica extra: Ejemplo de ajuste ---------------
        # Si es día fertilizante y CE algo alta (2.0 < CE < 2.5),
        # reducimos la dosis un 20% adicional.
        dia_fertilizante = sensor_values.get('dia_fertilizante', False)
        ce_val = sensor_values.get('ce', 1.0)

        if dia_fertilizante and (2.0 < ce_val < 2.5):
            logging.info("Día fertilizante y CE algo alta => reduciendo 20% la dosis de fertilizante.")
            fert *= 0.8
            fert = max(fert, 0)

        # --------------------------------------------------------------

        decision = self._generar_decisiones(fert, agua, sensor_values)
        logging.info(f"DecisionEngine => Decisión final (con ML): {decision}")
        return decision

    # --------------------------------------------------------------------------
    # 5) Decisiones por defecto
    # --------------------------------------------------------------------------
    def _decisiones_por_defecto(self, sensor_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback cuando no hay modelos o hubo error. Reglas sencillas.
        """
        logging.info("DecisionEngine => _decisiones_por_defecto.")
        decision = {
            'activar_bomba':         False,
            'abrir_valvula_riego':   False,
            'inyectar_fertilizante': False,
            'abrir_valvula_suministro': False,
            'porcentaje_fertilizante': 0,
            'cantidad_agua': 0
        }

        hum = sensor_values.get('humidity') or sensor_values.get('humedad_avg', 50)
        if hum < 30:
            decision['activar_bomba'] = True
            decision['abrir_valvula_riego'] = True
            decision['cantidad_agua'] = 10

        ce = sensor_values.get('ce', 1.5)
        if ce < 1.0:
            decision['inyectar_fertilizante'] = True
            decision['porcentaje_fertilizante'] = 5

        wlev = sensor_values.get('water_level', 50)
        if wlev < 20:
            decision['abrir_valvula_suministro'] = True

        return decision

    # --------------------------------------------------------------------------
    # 6) Definir umbrales por estación (pH, CE, N, P, K, etc.)
    # --------------------------------------------------------------------------
    def _definir_umbrales_por_estacion(self, season: str) -> Dict[str, Dict[str, float]]:
        """
        Ajusta min/max para temperature, humidity, ph, ce, N, P, K según la estación.
        """
        if season == 'summer':
            return {
                'temperature': {'min': 15,  'max': 40},
                'humidity':    {'min': 20,  'max': 80},
                'ph':          {'min': 5.5, 'max': 7.5},
                'ce':          {'min': 1.0, 'max': 2.5},
                'N':           {'min': 10,  'max': 80},
                'P':           {'min': 5,   'max': 60},
                'K':           {'min': 5,   'max': 70}
            }
        elif season == 'autumn':
            return {
                'temperature': {'min': 10,  'max': 30},
                'humidity':    {'min': 25,  'max': 75},
                'ph':          {'min': 5.5, 'max': 7.5},
                'ce':          {'min': 1.0, 'max': 2.5},
                'N':           {'min': 10,  'max': 70},
                'P':           {'min': 5,   'max': 50},
                'K':           {'min': 5,   'max': 60}
            }
        elif season == 'winter':
            return {
                'temperature': {'min': 5,   'max': 25},
                'humidity':    {'min': 30,  'max': 70},
                'ph':          {'min': 5.5, 'max': 7.5},
                'ce':          {'min': 1.0, 'max': 2.5},
                'N':           {'min': 5,   'max': 60},
                'P':           {'min': 5,   'max': 40},
                'K':           {'min': 5,   'max': 50}
            }
        elif season == 'spring':
            return {
                'temperature': {'min': 10,  'max': 30},
                'humidity':    {'min': 25,  'max': 75},
                'ph':          {'min': 5.5, 'max': 7.5},
                'ce':          {'min': 1.0, 'max': 2.5},
                'N':           {'min': 10,  'max': 80},
                'P':           {'min': 5,   'max': 60},
                'K':           {'min': 5,   'max': 70}
            }
        else:
            # Por defecto
            return {
                'temperature': {'min': 10,  'max': 35},
                'humidity':    {'min': 20,  'max': 80},
                'ph':          {'min': 5.0, 'max': 8.0},
                'ce':          {'min': 0.5, 'max': 3.0},
                'N':           {'min': 5,   'max': 70},
                'P':           {'min': 5,   'max': 50},
                'K':           {'min': 5,   'max': 60}
            }

    # --------------------------------------------------------------------------
    # 7) Generar decisiones tras la predicción
    # --------------------------------------------------------------------------
    def _generar_decisiones(self, fert: float, agua: float,
                            sensor_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construye el dict de decisión final (activar bomba, abrir riego, etc.)
        según los valores de fertilizante y agua recibidos.
        """
        decision = {
            'activar_bomba':         False,
            'abrir_valvula_riego':   False,
            'inyectar_fertilizante': False,
            'abrir_valvula_suministro': False,
            'porcentaje_fertilizante': 0.0,
            'cantidad_agua': 0.0
        }

        if agua > 0:
            decision['activar_bomba'] = True
            decision['abrir_valvula_riego'] = True
            decision['cantidad_agua'] = round(agua, 2)

        if fert > 0:
            decision['inyectar_fertilizante'] = True
            decision['porcentaje_fertilizante'] = round(fert, 2)

        wlev = sensor_values.get('water_level', 50)
        if wlev < 20:
            logging.info("Decisión: water_level <20 => abrir suministro de agua.")
            decision['abrir_valvula_suministro'] = True

        return decision

    # --------------------------------------------------------------------------
    # 8) Generar sugerencias (modo manual)
    # --------------------------------------------------------------------------
    def generar_sugerencias(self, sensor_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un diccionario con sugerencias en modo manual,
        basado en la evaluación normal.
        """
        logging.info("Generando sugerencias (modo manual)...")
        decision = self.evaluar(sensor_values)
        sugerencias = {
            'accion_recomendada':     "Abrir riego" if decision.get('abrir_valvula_riego', False) else "No regar",
            'agua_sugerida(L)':      decision.get('cantidad_agua', 0),
            'fertilizar':            "Sí" if decision.get('inyectar_fertilizante', False) else "No",
            'dosis_fertilizante(%)': decision.get('porcentaje_fertilizante', 0),
            'abrir_suministro':      "Sí" if decision.get('abrir_valvula_suministro', False) else "No"
        }
        return sugerencias

    # --------------------------------------------------------------------------
    # 9) Entrenamiento placeholders (local vs. nube)
    # --------------------------------------------------------------------------
    def entrenar_localmente(self, data_path: str = 'data/sensor_data.csv') -> None:
        logging.info(f"Entrenamiento local con data en {data_path} (pendiente).")

    def entrenar_en_nube(self) -> None:
        logging.info("Entrenamiento en la nube (placeholder).")

    # --------------------------------------------------------------------------
    # FIN
    # --------------------------------------------------------------------------
