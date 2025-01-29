# controller.py
# -------------------------------------------------------------------------------
# Controlador principal del sistema de riego inteligente.
# Integra:
#   - 7 válvulas (desague_1, desague_2, mora, lechuga, aguaymanto, mango, suministro_pucp)
#   - 1 bomba DC
#   - 4 sensores de suelo 7en1 (mora, mango, aguaymanto, lechuga)
#   - 1 sensor de flujo
#   - 1 sensor de nivel (ultrasónico)
#   - Modo automático vs manual sugerido
#   - Cronograma de riego (L, M, X) y domingos especiales
#   - Limpieza de datos (3 meses automático, 6 meses manual), CSVs
#   - Chequeo de coherencia
# -------------------------------------------------------------------------------

import time
import logging
import threading
import os
import csv
import sys
import requests
from datetime import datetime, timedelta
import schedule
import shutil
import gzip
import pandas as pd

# Módulos del proyecto
from sensors import SoilSensor7en1, LevelSensor, FlowSensor
from actuators import PumpControl, ValveControl
from signal_conditioning import SignalConditioning
from decision_engine import DecisionEngine
from cloud_sync import CloudSync
from gui import NodeRedInterface

from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class ControladorSistemaRiego:
    """
    Clase principal que coordina la lógica de riego:
     - 7 electroválvulas (desague_1, desague_2, mora, lechuga, aguaymanto, mango, suministro_pucp)
     - Bomba DC
     - 4 sensores 7en1 (mora, mango, aguaymanto, lechuga)
     - Sensor de flujo y sensor de nivel
     - Cronograma de riego programado (lunes, martes, miércoles)
     - Días especiales (fertilizante o entrenamiento) los domingos
     - Modo automático vs. manual
     - Limpieza de datos local (CSV)
     - Chequeo de coherencia en las decisiones
    """

    def __init__(self) -> None:
        # ----------------------------------------------------------------------
        # 1) Inicializar Sensores
        # ----------------------------------------------------------------------
        self.soil_sensor_mora       = SoilSensor7en1(address=1)
        self.soil_sensor_mango      = SoilSensor7en1(address=2)
        self.soil_sensor_aguaymanto = SoilSensor7en1(address=3)
        self.soil_sensor_lechuga    = SoilSensor7en1(address=4)

        self.level_sensor = LevelSensor(
            trig_pin=18,
            echo_pin=24,
            calibration_params={'tank_height': 200}
        )
        self.flow_sensor = FlowSensor(
            gpio_pin=17,
            calibration_params={
                'factor': 5.5,
                'tolerance': 0.2,
                'minimal_flow_threshold': 0.5
            }
        )

        # ----------------------------------------------------------------------
        # 2) Acondicionamiento de señal (filtros, etc.)
        # ----------------------------------------------------------------------
        self.signal_conditioning = SignalConditioning()

        # ----------------------------------------------------------------------
        # 3) Motor de decisiones (Machine Learning + Reglas de umbrales)
        # ----------------------------------------------------------------------
        self.decision_engine = DecisionEngine(model_path='data/modelo_actualizado.pkl')

        # ----------------------------------------------------------------------
        # 4) Sincronización con la nube (cloud_sync)
        # ----------------------------------------------------------------------
        self.cloud_sync = CloudSync(credentials_path='config/credentials.json')

        # ----------------------------------------------------------------------
        # 5) Interfaz Node-RED o GUI
        # ----------------------------------------------------------------------
        self.gui = NodeRedInterface()

        # ----------------------------------------------------------------------
        # 6) Parámetros y estado global
        # ----------------------------------------------------------------------
        self.data_collection_frequency = 60  # seg (leer cada 1 min)
        self.sensor_data: List[Dict[str, Any]] = []
        self.online_mode = False
        self.control_mode = 'automatic'   # 'manual' es la otra opción
        self.is_busy = False              # Evita acciones simultáneas en _ejecutar_decision()
        self.last_manual_interaction = datetime.now()  # Forzar 'auto' si 2 días sin respuesta

        # ----------------------------------------------------------------------
        # 7) Definir pines GPIO para bomba + 7 válvulas (ajustar a tu hardware)
        # ----------------------------------------------------------------------
        GPIO_PINS = {
            'bomba_dc':        25,
            'desague_1':       5,
            'desague_2':       6,
            'mora':            13,
            'lechuga':         19,
            'aguaymanto':      26,
            'mango':           16,
            'suministro_pucp': 20
        }

        # ----------------------------------------------------------------------
        # 8) Inicializar actuadores
        # ----------------------------------------------------------------------
        self.pump_control = PumpControl(GPIO_PINS['bomba_dc'])
        self.valve_control = ValveControl({
            'desague_1':       GPIO_PINS['desague_1'],
            'desague_2':       GPIO_PINS['desague_2'],
            'mora':            GPIO_PINS['mora'],
            'lechuga':         GPIO_PINS['lechuga'],
            'aguaymanto':      GPIO_PINS['aguaymanto'],
            'mango':           GPIO_PINS['mango'],
            'suministro_pucp': GPIO_PINS['suministro_pucp']
        })

        # ----------------------------------------------------------------------
        # 9) Variables de riego/fertilizante y cronograma
        # ----------------------------------------------------------------------
        self.bomba_activada_hoy = False
        self.ultimo_dia_bomba: Optional[datetime.date] = None

        self.total_daily_water = 0
        self.remaining_daily_water = 0
        self.daily_fertilizer_percentage = 0

        # Programar riego lunes, martes, miércoles:
        self._programar_riego_programado()
        # Programar domingo especial:
        self._programar_domingo_especial()

        # ----------------------------------------------------------------------
        # 10) Lock de hilos
        # ----------------------------------------------------------------------
        self.lock = threading.Lock()

        # Si se corre en Windows, modo simulado
        self.simulation_mode = (sys.platform == "win32")

        logging.info("ControladorSistemaRiego inicializado con éxito.")

    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    def iniciar(self) -> None:
        """
        Bucle principal con schedule + lecturas periódicas.
        """
        logging.info("Iniciando loop principal del controlador de riego...")

        while True:
            try:
                # 1) Verificar si se cambia de día para resetear la bomba
                self._verificar_reset_bomba_diaria()

                # 2) Verificar si el modo manual lleva mucho tiempo inactivo
                self._verificar_time_out_manual()

                # 3) Ejecutar las tareas programadas
                schedule.run_pending()

                # 4) Ciclo de lectura y procesamiento
                self._ciclo_lectura_y_procesamiento()

                time.sleep(self.data_collection_frequency)

            except KeyboardInterrupt:
                logging.info("Interrupción manual (Ctrl + C). Saliendo...")
                break
            except Exception:
                logging.exception("Error en el loop principal:")
                time.sleep(5)

    def _verificar_reset_bomba_diaria(self) -> None:
        """
        Resetea el flag de bomba_activada_hoy si es un nuevo día.
        """
        hoy = datetime.now().date()
        if self.ultimo_dia_bomba is None or hoy != self.ultimo_dia_bomba:
            self.bomba_activada_hoy = False
            self.ultimo_dia_bomba = hoy

    def _ciclo_lectura_y_procesamiento(self) -> None:
        """
        Tareas recurrentes en cada ciclo:
          - Leer sensores
          - Manejar nivel de tanque
          - Riego no programado
          - Actualizar GUI
          - Monitorear almacenamiento
          - En modo manual => recibir comandos
        """
        sensor_values = self._leer_sensores_global()
        self.online_mode = self._verificar_conexion_internet()
        self._control_nivel_tanque(sensor_values)

        # Riego no programado (jueves=3, viernes=4, sábado=5)
        self._manejar_riego_no_programado(sensor_values)

        # Manejo modo manual
        if self.control_mode == 'manual':
            comandos = self.gui.recibir_comandos()
            # Procesar comandos
            if comandos.get('activar_bomba'):
                self.pump_control.activar()
            if comandos.get('abrir_valvula_mora'):
                self.valve_control.abrir_valvula('mora')
            if comandos.get('cerrar_valvula_mora'):
                self.valve_control.cerrar_valvula('mora')
            # Otros comandos análogos...

            # Registrar acción manual en CSV
            accion_manual = {'accion_manual': True}
            if comandos.get('activar_bomba'):
                accion_manual['activar_bomba'] = True
            if comandos.get('abrir_valvula_mora'):
                accion_manual['abrir_valvula_mora'] = True
            if comandos.get('cerrar_valvula_mora'):
                accion_manual['cerrar_valvula_mora'] = True

            if len(accion_manual) > 1:  # al menos una acción
                self._guardar_decision_csv(accion_manual)
            self.last_manual_interaction = datetime.now()

        # Detectar fallos de flujo
        pump_on = self.pump_control.estado_actual()
        any_valve_open = any(self.valve_control.estado_actual(v) for v in self.valve_control.valvulas)
        expected_flow = 5 if (pump_on and any_valve_open) else 0
        fallo = self.flow_sensor.detectar_fallo(expected_flow=expected_flow)

        if fallo:
            if self.control_mode == 'automatic':
                self.pump_control.desactivar()
                logging.error("Fuga/obstrucción detectada en modo automático. Bomba desactivada.")
            else:
                logging.warning("Posible obstrucción/fuga. Avisar al usuario en modo manual.")

        # Actualizar interfaz
        self.gui.actualizar_interfaz(sensor_values, self.control_mode, {})
        # Limpieza de datos antiguos si hace falta
        self._monitorear_almacenamiento()

    def _verificar_time_out_manual(self) -> None:
        """
        Si en modo manual no hay interacción en 2 días => pasar a automático.
        """
        if self.control_mode == 'manual':
            dias_sin = (datetime.now() - self.last_manual_interaction).days
            if dias_sin >= 2:
                with self.lock:
                    if not self.is_busy:
                        self.control_mode = 'automatic'
                        logging.warning("Forzando modo automático por 2 días sin respuesta en manual.")

    # =========================================================================
    # LECTURA DE SENSORES
    # =========================================================================
    def _leer_sensores_global(self) -> Dict[str, Any]:
        """
        Lee 4 sensores 7en1 + nivel + flujo.
        Retorna un dict con:
          'humedad_avg', 'temp_avg', 'ce', 'ph', 'N', 'P', 'K',
          'water_level', 'flow_rate', 'season', 'modo_control', ...
        """
        try:
            mora_data  = self.soil_sensor_mora.leer() or {}
            mango_data = self.soil_sensor_mango.leer() or {}
            agm_data   = self.soil_sensor_aguaymanto.leer() or {}
            lech_data  = self.soil_sensor_lechuga.leer()  or {}

            # Promediar humedad, temp, ce, ph, N, P, K
            humidity_avg = (
                mora_data.get('humidity', 50) +
                mango_data.get('humidity', 50) +
                agm_data.get('humidity', 50)  +
                lech_data.get('humidity', 50)
            ) / 4.0
            temp_avg = (
                mora_data.get('temperature', 20) +
                mango_data.get('temperature', 20) +
                agm_data.get('temperature', 20)  +
                lech_data.get('temperature', 20)
            ) / 4.0
            ce_avg = (
                mora_data.get('ce', 2.0) +
                mango_data.get('ce', 2.0) +
                agm_data.get('ce', 2.0) +
                lech_data.get('ce', 2.0)
            ) / 4.0
            ph_avg = (
                mora_data.get('ph', 6.5) +
                mango_data.get('ph', 6.5) +
                agm_data.get('ph', 6.5) +
                lech_data.get('ph', 6.5)
            ) / 4.0
            n_avg = (
                mora_data.get('N', 15.0) +
                mango_data.get('N', 15.0) +
                agm_data.get('N', 15.0) +
                lech_data.get('N', 15.0)
            ) / 4.0
            p_avg = (
                mora_data.get('P', 15.0) +
                mango_data.get('P', 15.0) +
                agm_data.get('P', 15.0) +
                lech_data.get('P', 15.0)
            ) / 4.0
            k_avg = (
                mora_data.get('K', 15.0) +
                mango_data.get('K', 15.0) +
                agm_data.get('K', 15.0) +
                lech_data.get('K', 15.0)
            ) / 4.0

            # Acondicionar humedad y temperatura
            humidity = self.signal_conditioning.acondicionar_humedad(humidity_avg)
            temperature = self.signal_conditioning.acondicionar_temperatura(temp_avg)

            # Nivel
            water_level = self.level_sensor.leer()
            water_level = self.signal_conditioning.acondicionar_nivel(water_level)

            # Flujo
            flow = self.flow_sensor.leer(
                bomba_activa=self.pump_control.estado_actual(),
                valvula_riego_abierta=True
            )

            # Armar dict final
            now_str = datetime.now().isoformat()
            sensor_values = {
                'timestamp':   now_str,
                'humedad_avg': round(humidity, 2),
                'temp_avg':    round(temperature, 2),
                'ce':          round(ce_avg, 2),
                'ph':          round(ph_avg, 2),
                'N':           round(n_avg, 2),
                'P':           round(p_avg, 2),
                'K':           round(k_avg, 2),
                'water_level': round(water_level, 2) if water_level else None,
                'flow_rate':   round(flow, 2) if flow else 0.0,
                'season':      self._get_current_season(),
                'modo_control': self.control_mode
            }

            self._guardar_sensores_csv(sensor_values)
            return sensor_values

        except Exception:
            logging.exception("Error en _leer_sensores_global:")
            return {}

    def _guardar_sensores_csv(self, data: Dict[str, Any]) -> None:
        csv_path = 'data/sensor_data.csv'
        file_exists = os.path.isfile(csv_path)
        campos = [
            'timestamp','humedad_avg','temp_avg','ce','ph','N','P','K',
            'water_level','flow_rate','season','modo_control'
        ]
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=campos)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
        except Exception:
            logging.exception("Error al guardar sensor_data.csv:")

    def _get_current_season(self) -> str:
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        elif month in [9, 10, 11]:
            return 'spring'
        return 'unknown'

    # =========================================================================
    # CONTROL DE NIVEL DE TANQUE
    # =========================================================================
    def _control_nivel_tanque(self, sensor_values: Dict[str, Any]) -> None:
        lvl = sensor_values.get('water_level')
        if lvl is None:
            logging.error("Sensor de nivel devolvió None. Revisar sensor.")
            self.gui.enviar_mensaje("El sensor de nivel no está disponible. Revise hardware o conexión.")
            return

        lo = 30.0
        hi = 85.0
        if self.control_mode == 'manual':
            self._control_nivel_tanque_manual(lvl, lo, hi)
        else:
            self._control_nivel_tanque_automatico(lvl, lo, hi)

    def _control_nivel_tanque_automatico(self, level: float, lo: float, hi: float) -> None:
        with self.lock:
            if level < lo:
                self.valve_control.cerrar_valvula('desague_1')
                self.valve_control.cerrar_valvula('desague_2')
                self.valve_control.abrir_valvula('suministro_pucp')
            elif level > hi:
                for valv in ['mora','lechuga','aguaymanto','mango']:
                    self.valve_control.cerrar_valvula(valv)
                self.valve_control.abrir_valvula('desague_1')
                self.valve_control.abrir_valvula('desague_2')
                self.valve_control.cerrar_valvula('suministro_pucp')
            else:
                self.valve_control.cerrar_valvula('desague_1')
                self.valve_control.cerrar_valvula('desague_2')
                self.valve_control.cerrar_valvula('suministro_pucp')

    def _control_nivel_tanque_manual(self, level: float, lo: float, hi: float) -> None:
        if level < lo:
            logging.info("Modo Manual: Sugerir abrir 'suministro_pucp' (nivel bajo).")
        elif level > hi:
            logging.info("Modo Manual: Sugerir abrir desagües 1 y 2 (nivel alto).")
        else:
            logging.debug("Nivel dentro de umbral, no sugerir nada en modo manual.")

    # =========================================================================
    # RIEGO NO PROGRAMADO (jueves=3, viernes=4, sábado=5)
    # =========================================================================
    def _manejar_riego_no_programado(self, sensor_values: Dict[str, Any]) -> None:
        weekday = datetime.now().weekday()
        if weekday in [3, 4, 5]:  # jueves, viernes, sábado
            hum = sensor_values.get('humedad_avg', 50)
            if hum < 20 or hum > 90:
                logging.info("Fuera de umbrales (día no programado) => corrección simple.")
                self._corrigir_fuera_umbrales(sensor_values)

    def _corrigir_fuera_umbrales(self, sensor_values: Dict[str, Any]) -> None:
        """
        Corrección simple de humedad fuera de umbrales.
        CUIDADO: se libera el lock entre dos bloques consecutivos.
        """
        with self.lock:
            if self.is_busy:
                return

        hum = sensor_values.get('humedad_avg', 50)
        if hum < 10:
            # Emergencia, ignorar restricción de bomba diario
            logging.warning("Humedad < 10%. Emergencia => ignorar restricción de bomba diaria.")
            decision = {
                'activar_bomba': True,
                'abrir_valvula_riego': True,
                'emergencia': True,
                'inyectar_fertilizante': False,
                'abrir_valvula_suministro': False,
                'porcentaje_fertilizante': 0,
                'cantidad_agua': 5
            }
            self._ejecutar_decision(decision)
            return

        if hum < 20:
            # Riego corto de 5 seg
            with self.lock:
                self.pump_control.activar()
                self.valve_control.abrir_valvula('mora')
            time.sleep(5)
            with self.lock:
                self.valve_control.cerrar_valvula('mora')
                self.pump_control.desactivar()
        elif hum > 90:
            logging.warning("Humedad > 90%. Se sugiere no regar en este momento.")

    # =========================================================================
    # PROGRAMAR RIEGO (lunes=0, martes=1, miércoles=2)
    # =========================================================================
    def _programar_riego_programado(self) -> None:
        schedule.clear('riego_programado')
        schedule.every().day.at("07:00").do(self._check_riego_programado).tag('riego_programado')
        logging.info("Riego programado (lunes, martes, miércoles) a las 07:00h.")

    def _check_riego_programado(self):
        w = datetime.now().weekday()
        if w in [0, 1, 2]:
            self._evento_riego_programado()

    def _evento_riego_programado(self) -> None:
        logging.info("Evento de riego programado (IA).")
        try:
            sensor_values = self._leer_sensores_global()
            decision = self.decision_engine.evaluar(sensor_values)

            # 1) Coherencia
            if not self._verificar_coherencia(sensor_values, decision):
                logging.warning("Coherencia fallida => fallback o corrección.")
                decision = self._decisiones_coherencia_fallback(sensor_values)

            # 2) Umbrales
            if not self._decision_dentro_umbrales(sensor_values, decision):
                logging.warning("Decisión IA fuera de umbrales => corrección simple.")
                self._corrigir_fuera_umbrales(sensor_values)
            else:
                self._ejecutar_decision(decision)

        except Exception:
            logging.exception("Error en _evento_riego_programado:")

    def _verificar_coherencia(self, sensor_values: Dict[str, Any], decision: Dict[str, Any]) -> bool:
        ce_real = sensor_values.get('ce', 1.5)
        hum = sensor_values.get('humedad_avg', 50)
        fert_pct = decision.get('porcentaje_fertilizante', 0)
        agua = decision.get('cantidad_agua', 0)

        if ce_real > 3.0 and fert_pct > 5:
            logging.warning("Coherencia: CE alta con fertilizante elevado => Fail.")
            return False
        if hum >= 90 and agua > 10:
            logging.warning("Coherencia: Humedad >=90 y agua>10 => Fail.")
            return False
        return True

    def _decisiones_coherencia_fallback(self, sensor_values: Dict[str, Any]) -> Dict[str, Any]:
        decision_fb: Dict[str, Any] = {
            'activar_bomba': False,
            'abrir_valvula_riego': False,
            'inyectar_fertilizante': False,
            'abrir_valvula_suministro': False,
            'porcentaje_fertilizante': 0,
            'cantidad_agua': 0
        }
        ce = sensor_values.get('ce', 1.5)
        if ce > 3.0:
            decision_fb['activar_bomba'] = True
            decision_fb['abrir_valvula_riego'] = True
            decision_fb['porcentaje_fertilizante'] = 1.0
            decision_fb['cantidad_agua'] = 5
        else:
            decision_fb['activar_bomba'] = True
            decision_fb['abrir_valvula_riego'] = True
            decision_fb['cantidad_agua'] = 8
        return decision_fb

    def _decision_dentro_umbrales(self, sensor_values: Dict[str, Any], decision: Dict[str, Any]) -> bool:
        ce_mora = sensor_values.get('ce', 2.0)
        fert = decision.get('porcentaje_fertilizante', 0)
        if ce_mora > 3.5 and fert > 10:
            return False
        return True

    def _ejecutar_decision(self, decision: Dict[str, Any]) -> None:
        """
        ATENCIÓN sobre la sección con time.sleep(10):
          - Actualmente se libera el lock durante esos 10 seg, lo cual permite que
            otros hilos puedan modificar válvulas o la bomba en paralelo. Esto
            podría provocar estados intermedios imprevistos.
          - Si se desea bloquear completamente, se mantendría el lock durante toda
            la operación (incluyendo el sleep). Aquí se deja como está para permitir
            cierta concurrencia con cautela.
        """
        with self.lock:
            if self.is_busy:
                logging.warning("is_busy=True, se omite la acción para evitar conflicto.")
                return
            self.is_busy = True

            # 1) Activar bomba, respetando la restricción diaria (salvo 'emergencia')
            if decision.get('activar_bomba', False):
                if not decision.get('emergencia', False) and self.bomba_activada_hoy:
                    logging.warning("Bomba ya fue activada hoy, se omite nueva activación.")
                else:
                    # Encender bomba y apagar en 20 min (en hilo aparte)
                    self._encender_bomba_contemporizada()
                    self.bomba_activada_hoy = True
                    self.ultimo_dia_bomba = datetime.now().date()
            else:
                self.pump_control.desactivar()

            # 2) Válvula de riego
            if decision.get('abrir_valvula_riego', False):
                self.valve_control.abrir_valvula('mora')
            else:
                self.valve_control.cerrar_valvula('mora')

            # 3) Fertilizante
            if decision.get('inyectar_fertilizante', False):
                fert_pct = decision.get('porcentaje_fertilizante', 0)
                self._dosificar_fertilizante(fert_pct)

            # 4) Válvula de suministro
            if decision.get('abrir_valvula_suministro', False):
                self.valve_control.abrir_valvula('suministro_pucp')
            else:
                self.valve_control.cerrar_valvula('suministro_pucp')

            # Guardar en CSV
            self._guardar_decision_csv(decision)

        # Aquí liberamos el lock para permitir que otros hilos sigan
        # corriendo, aunque hay un lapso de 10 seg de 'espera' local:
        time.sleep(10)

        # Re-adquirimos lock para cerrar la acción
        with self.lock:
            # Apagar bomba si sigue encendida (si no la apagó antes el hilo 20 min).
            if self.pump_control.estado_actual():
                self.pump_control.desactivar()
            self.is_busy = False

    # --------------------------------------------------------------------------
    # Bomba con temporizador de 20 min
    # --------------------------------------------------------------------------
    def _encender_bomba_contemporizada(self) -> None:
        self.pump_control.activar()
        hilo_apagado = threading.Thread(target=self._apagar_bomba_despues_de_20, daemon=True)
        hilo_apagado.start()

    def _apagar_bomba_despues_de_20(self) -> None:
        time.sleep(20 * 60)  # 20 min
        with self.lock:
            if self.pump_control.estado_actual():
                self.pump_control.desactivar()
                logging.info("Bomba desactivada automáticamente después de 20 min.")

    def _dosificar_fertilizante(self, porcentaje: float) -> None:
        logging.info(f"Dosificando fertilizante al {porcentaje}%.")
        time.sleep(2)
        logging.info("Dosificación completada.")

    # =========================================================================
    # DOMINGO ESPECIAL
    # =========================================================================
    def _programar_domingo_especial(self) -> None:
        schedule.clear('domingo_especial')
        schedule.every().day.at("08:00").do(self._check_domingo_especial).tag('domingo_especial')
        logging.info("Domingo especial (fertilizante/entrenamiento) verificado a las 08:00h.")

    def _check_domingo_especial(self):
        w = datetime.now().weekday()
        if w == 6:
            self._evento_domingo()

    def _evento_domingo(self) -> None:
        hoy = datetime.now().date()
        semana_del_mes = (hoy.day - 1) // 7 + 1
        if semana_del_mes in [1, 3]:
            logging.info("Domingo de fertilizante.")
            self._funcion_dia_fertilizante()
        else:
            logging.info("Domingo de entrenamiento.")
            self._funcion_dia_entrenamiento()

    def _funcion_dia_fertilizante(self) -> None:
        logging.info("Secuencia: vaciar->recargar->fertilizar")
        if self.control_mode == 'automatic':
            # 1) Vaciar tanque
            self.valve_control.abrir_valvula('desague_1')
            self.valve_control.abrir_valvula('desague_2')
            logging.info("Desagües abiertos para vaciar. Esperando nivel <10.")
            start_time = datetime.now()
            timeout = timedelta(minutes=5)  # Máximo tiempo para vaciar el tanque

            while True:
                lvl_current = self.level_sensor.leer()
                if lvl_current is None:
                    logging.error("Sensor nivel no disponible. Abortando fertilización.")
                    return
                if lvl_current < 10:
                    break
                if datetime.now() - start_time > timeout:
                    logging.warning("Timeout al intentar vaciar el tanque. Continuando...")
                    break
                time.sleep(5)

            logging.info("Tanque vacío. Cerrando desagües.")
            self.valve_control.cerrar_valvula('desague_1')
            self.valve_control.cerrar_valvula('desague_2')

            # 2) Recargar tanque
            logging.info("Abriendo suministro_pucp. Esperando nivel >80.")
            self.valve_control.abrir_valvula('suministro_pucp')
            start_time = datetime.now()
            timeout = timedelta(minutes=10)  # Máximo tiempo para llenar el tanque

            while True:
                lvl_current = self.level_sensor.leer()
                if lvl_current is None:
                    logging.error("Sensor nivel no disponible. Abortando fertilización.")
                    return
                if lvl_current > 80:
                    break
                if datetime.now() - start_time > timeout:
                    logging.warning("Timeout al intentar llenar el tanque. Continuando...")
                    break
                time.sleep(5)

            logging.info("Cerrando suministro_pucp.")
            self.valve_control.cerrar_valvula('suministro_pucp')
        else:
            # Modo manual => sugerencias
            self.gui.mostrar_sugerencia("Abra desagües para vaciar el tanque. Confirme cuando esté vacío.")
            while not self.gui.recibir_confirmacion("tanque_vacio"):
                time.sleep(5)

            self.gui.mostrar_sugerencia("Abra suministro_pucp para recargar. Confirme cuando supere 80%.")
            while not self.gui.recibir_confirmacion("tanque_lleno"):
                time.sleep(5)
                
        # 3) Revisar pH, CE, NPK
        sensor_values = self._leer_sensores_global()
        ph = sensor_values.get('ph', 6.5)
        ce = sensor_values.get('ce', 2.0)
        N_val = sensor_values.get('N', 15.0)
        P_val = sensor_values.get('P', 15.0)
        K_val = sensor_values.get('K', 15.0)

        fuera_de_rango = False
        if not (5.5 <= ph <= 7.5):
            fuera_de_rango = True
        if ce > 3.0:
            fuera_de_rango = True
        if N_val > 50 or P_val > 50 or K_val > 50:
            fuera_de_rango = True

        if fuera_de_rango:
            logging.warning("pH/CE/NPK fuera de rango => no fertilizar, riego corto.")
            decision_emergente = {
                'activar_bomba': True,
                'abrir_valvula_riego': True,
                'inyectar_fertilizante': False,
                'abrir_valvula_suministro': False,
                'porcentaje_fertilizante': 0,
                'cantidad_agua': 5
            }
            self._ejecutar_decision(decision_emergente)
            return

        # Si está en rango => día de fertilizante normal
        sensor_values['dia_fertilizante'] = True
        decision = self.decision_engine.evaluar(sensor_values)
        if not self._verificar_coherencia(sensor_values, decision):
            logging.warning("Coherencia fallida en día fertil => fallback.")
            decision = self._decisiones_coherencia_fallback(sensor_values)
        if not self._decision_dentro_umbrales(sensor_values, decision):
            logging.warning("Decisión fertilizante fuera de umbrales => corrección.")
            self._corrigir_fuera_umbrales(sensor_values)
        else:
            self._ejecutar_decision(decision)

    def _funcion_dia_entrenamiento(self) -> None:
        if self._verificar_conexion_internet():
            ok = self.cloud_sync.sincronizar_con_nube()
            if ok:
                self.decision_engine.cargar_modelo()
        else:
            pass

    def _verificar_conexion_internet(self) -> bool:
        try:
            requests.get("https://www.google.com", timeout=5)
            return True
        except:
            return False

    # =========================================================================
    # MONITOREO ALMACENAMIENTO
    # =========================================================================
    def _monitorear_almacenamiento(self) -> None:
        try:
            total, used, free = shutil.disk_usage("/")
            used_perc = (used / total) * 100
            if used_perc > 80:
                logging.warning("Almacenamiento >80%. Eliminando datos antiguos.")
                self._eliminar_datos_antiguos()
        except Exception:
            logging.exception("Error al monitorear almacenamiento:")

    def _eliminar_datos_antiguos(self) -> None:
        try:
            self._backup_data_files()
            self._filtrar_csv_por_modo_control('data/sensor_data.csv')
            self._filtrar_csv_por_modo_control('data/decision_data.csv')
        except Exception:
            logging.exception("Error al eliminar datos antiguos:")

    def _filtrar_csv_por_modo_control(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            return
        df = pd.read_csv(file_path)
        if 'timestamp' not in df.columns or 'modo_control' not in df.columns:
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        now = datetime.now()
        cutoff_auto = now - timedelta(days=90)
        cutoff_manual = now - timedelta(days=180)

        mask_auto = (df['modo_control'] == 'automatic')
        mask_manual = (df['modo_control'] == 'manual')

        df_auto = df[mask_auto & (df['timestamp'] >= cutoff_auto)]
        df_manual = df[mask_manual & (df['timestamp'] >= cutoff_manual)]
        df_new = pd.concat([df_auto, df_manual], ignore_index=True)
        df_new.sort_values('timestamp', inplace=True)

        df_new.to_csv(file_path, index=False)
        logging.info(f"Filtrados datos antiguos en {file_path}, 90d auto y 180d manual.")

    def _backup_data_files(self) -> None:
        try:
            backup_dir = 'data/backup'
            os.makedirs(backup_dir, exist_ok=True)
            now_str = datetime.now().strftime('%Y%m%d_%H%M%S')

            for fpath in ['data/sensor_data.csv', 'data/decision_data.csv']:
                if os.path.isfile(fpath):
                    base_name = os.path.basename(fpath).replace('.csv', '')
                    bkp_name = f"{base_name}_{now_str}.csv.gz"
                    bkp_path = os.path.join(backup_dir, bkp_name)

                    with open(fpath, 'rb') as f_in:
                        with gzip.open(bkp_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    logging.info(f"Backup creado: {bkp_path}")

        except Exception:
            logging.exception("Error en backup_data_files:")

    # =========================================================================
    # REGISTRO DE DECISIONES
    # =========================================================================
    def _guardar_decision_csv(self, decision: Dict[str, Any]) -> None:
        file_path = 'data/decision_data.csv'
        file_exists = os.path.isfile(file_path)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = {**decision, 'timestamp': now_str}
        row['modo_control'] = self.control_mode

        for key in [
            'activar_bomba','abrir_valvula_riego',
            'inyectar_fertilizante','abrir_valvula_suministro',
            'porcentaje_fertilizante','cantidad_agua'
        ]:
            if key not in row:
                if 'abrir_valvula' in key or 'inyectar' in key or 'activar_bomba' in key:
                    row[key] = False
                else:
                    row[key] = 0

        campos = list(row.keys())
        if 'timestamp' not in campos:
            campos.insert(0, 'timestamp')

        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=campos)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            logging.info("Decisión registrada en decision_data.csv.")
        except Exception:
            logging.exception("Error al guardar decision_data.csv:")

    # =========================================================================
    # OBTENER ESTADO DE ACTUADORES
    # =========================================================================
    def get_actuators_state(self) -> Dict[str, bool]:
        """
        Retorna un dict con estado actual de la bomba y cada válvula.
        """
        estados: Dict[str, bool] = {}
        estados['bomba_dc'] = self.pump_control.estado_actual()
        for nombre_valvula in self.valve_control.valvulas.keys():
            estados[nombre_valvula] = self.valve_control.estado_actual(nombre_valvula)
        return estados

    # =========================================================================
    # FIN
    # =========================================================================
