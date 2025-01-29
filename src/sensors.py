# sensors.py
# -------------------------------------------------------------------------------
# Módulo de sensores para el sistema de riego inteligente.
# Incluye:
#   - SoilSensor7en1 (Modbus RS485) => (Temperatura, Humedad, CE, pH, N, P, K)
#   - LevelSensor (US-100 Trigger/Echo)
#   - FlowSensor (FS300A)
#   - Manejo de GPIO y simulación en Windows.
# -------------------------------------------------------------------------------

import time
import logging
import threading
import random
import sys

# Para Modbus RTU sobre RS485
try:
    from pymodbus.client import ModbusSerialClient
except ImportError:
    logging.warning("Librería pymodbus no disponible. Modo simulado para sensores Modbus.")
    ModbusSerialClient = None

# Simular RPi.GPIO si no estamos en Raspberry Pi
if sys.platform == "win32":
    from unittest.mock import MagicMock
    GPIO = MagicMock()
    logging.warning("GPIO no disponible en este entorno (Windows). Modo simulado.")
else:
    try:
        import RPi.GPIO as GPIO
    except ImportError:
        logging.warning("RPi.GPIO no disponible en este entorno. Modo simulado.")
        GPIO = None

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -------------------------------------------------------------------------------
# Configuración de pines (en caso de usar Raspberry Pi real)
# -------------------------------------------------------------------------------
if GPIO is not None:
    GPIO.setmode(GPIO.BOARD)

    # Pines RS485 (MAX485: DE/RE) - Modo half-duplex
    DE_RE_PIN = 27  # GPIO 27 para controlar transmisión/recepción
    GPIO.setup(DE_RE_PIN, GPIO.OUT)

    # Pines para sensor de flujo
    FLOW_SENSOR_PIN = 17

    # Pines para sensor US-100
    TRIG_PIN = 18
    ECHO_PIN = 24


# -------------------------------------------------------------------------------
# Clase base de sensores Modbus RS485 (SensorBase)
# -------------------------------------------------------------------------------
class SensorBase:
    """
    Clase base para sensores Modbus (RS485). Cada instancia manejará su propio 
    cliente ModbusSerialClient y su propio lock, permitiendo hilos en paralelo 
    cuando se dispone de puertos físicos separados (p.ej. /dev/ttyUSB0, /dev/ttyUSB1, etc.).
    """

    def __init__(self, address, serial_port='/dev/ttyUSB0', calibration_params=None):
        """
        :param address: Dirección (ID) Modbus del sensor.
        :param serial_port: Puerto serie donde está conectado el sensor 
                            (ej. '/dev/ttyUSB0', '/dev/ttyUSB1', etc.).
        :param calibration_params: Opcional, dict con parámetros de calibración específicos.
        """
        self.address = address
        self.serial_port = serial_port
        self.calibration_params = calibration_params or {}
        self.last_reading = None
        self.simulation_mode = (sys.platform == "win32")  # Modo simulado en Windows
        
        # Cada sensor tendrá su propio cliente y su propio lock
        self.modbus_client = None
        self.modbus_lock = threading.Lock()

        self.lock = threading.Lock()  # Para proteger self.last_reading
        if not self.simulation_mode and GPIO is not None:
            self._initialize_modbus_client()
        else:
            logging.info(f"Sensor Modbus en dirección {self.address} iniciado en modo simulado.")

    def _initialize_modbus_client(self):
        """
        Inicializa el cliente ModbusSerialClient para este sensor.
        Ajustar parámetros según el manual del sensor (4800 baudios, 8N1).
        """
        if not GPIO:
            logging.error("GPIO no disponible. No se puede inicializar Modbus en modo real.")
            return

        self.modbus_client = ModbusSerialClient(
            method='rtu',
            port=self.serial_port,
            baudrate=4800,  # Según especificación por defecto del sensor SoilSensor7en1
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=1,
        )
        if self.modbus_client.connect():
            logging.info(f"Cliente Modbus conectado correctamente en {self.serial_port}, addr={self.address}.")
        else:
            logging.error(f"Fallo al conectar el cliente Modbus en {self.serial_port}, addr={self.address}.")

    def leer_modbus(self, register_address, register_count):
        """
        Lectura genérica de registros Modbus (holding registers).
        - Retorna la lista de valores leídos o None si falla.
        - Hasta 3 intentos. Si da error, se cierra y re-conecta el cliente,
          se hace un pequeño back-off (0.2 s), y se reintenta.

        IMPORTANTE: Este método usa self.modbus_lock para
        garantizar el acceso ordenado al bus RS485 en modo half-duplex.
        """
        if not self.simulation_mode and self.modbus_client is not None:
            for attempt in range(3):
                try:
                    with self.modbus_lock:
                        # Modo transmisión
                        GPIO.output(DE_RE_PIN, GPIO.HIGH)
                        time.sleep(0.01)

                        result = self.modbus_client.read_holding_registers(
                            address=register_address,
                            count=register_count,
                            unit=self.address
                        )

                        # Modo recepción
                        GPIO.output(DE_RE_PIN, GPIO.LOW)

                    if not result.isError():
                        logging.debug(f"Modbus OK (addr={self.address}): {result.registers}")
                        return result.registers
                    else:
                        logging.error(f"Error Modbus addr={self.address}: {result}")
                        self.modbus_client.close()
                        time.sleep(0.2)  # pequeño delay antes de reconectar

                        if not self.modbus_client.connect():
                            logging.error("No se pudo reconectar al ModbusClient.")
                            if attempt < 2:
                                time.sleep(0.2)
                            else:
                                logging.error("Error al leer registros Modbus tras 3 intentos.")
                                return None
                        else:
                            # Reintento inmediato tras reconexión
                            with self.modbus_lock:
                                GPIO.output(DE_RE_PIN, GPIO.HIGH)
                                time.sleep(0.01)
                                second_result = self.modbus_client.read_holding_registers(
                                    address=register_address,
                                    count=register_count,
                                    unit=self.address
                                )
                                GPIO.output(DE_RE_PIN, GPIO.LOW)

                            if not second_result.isError():
                                logging.debug(f"Reintento exitoso (addr={self.address}): {second_result.registers}")
                                return second_result.registers
                            else:
                                logging.error(f"Error tras reconexión Modbus addr={self.address}: {second_result}")
                                if attempt < 2:
                                    time.sleep(0.2)
                                else:
                                    logging.error("Error al leer registros Modbus tras 3 intentos (con reconexión).")
                                    return None

                except Exception:
                    logging.exception(f"Excepción al leer Modbus addr={self.address} (intento {attempt + 1}/3):")
                    if attempt < 2:
                        time.sleep(0.2)
                    else:
                        return None

            return None
        else:
            # Modo simulado: generamos valores aleatorios
            return [random.randint(0, 1000) for _ in range(register_count)]

    def calibrar(self, raw_value):
        """
        Se implementa en las subclases si se necesita calibración específica.
        """
        raise NotImplementedError("calibrar() debe implementarse en subclases.")

    def leer(self):
        """
        Método genérico de lectura. Se implementa en cada subclase.
        """
        raise NotImplementedError("leer() debe implementarse en subclases.")

    def detectar_fallo(self):
        """
        Verifica si hay fallo en la lectura (cada subclase define sus criterios).
        """
        return False


# -------------------------------------------------------------------------------
# SoilSensor7en1: sensor RS485 con 7 parámetros (Temp, Humedad, CE, pH, N, P, K)
# -------------------------------------------------------------------------------
class SoilSensor7en1(SensorBase):
    """
    Sensor 7 en 1 RS485 que lee:
      - Temperatura
      - Humedad
      - CE (Conductividad eléctrica)
      - pH
      - N (nitrógeno)
      - P (fósforo)
      - K (potasio)

    Emplea registros holding Modbus. 
    """

    def __init__(self, address, serial_port='/dev/ttyUSB0', calibration_params=None):
        super().__init__(address, serial_port, calibration_params)
        self.register_address = 0x0000  # Registro inicial (según datasheet)
        self.register_count = 7         # 7 parámetros
        logging.info(f"Sensor de Suelo 7 en 1 inicializado (addr={address}, port={serial_port}).")

    def leer(self):
        """
        Lee los 7 valores y los almacena en self.last_reading.
        Retorna un dict con las claves:
          { 'humidity', 'temperature', 'ce', 'ph', 'N', 'P', 'K' }
        o None si falla.
        """
        if not self.simulation_mode and GPIO is not None:
            try:
                raw_values = self.leer_modbus(self.register_address, self.register_count)
                if raw_values and len(raw_values) == 7:
                    h_raw = raw_values[0]
                    t_raw = raw_values[1]
                    ce_raw = raw_values[2]
                    ph_raw = raw_values[3]
                    n_raw = raw_values[4]
                    p_raw = raw_values[5]
                    k_raw = raw_values[6]

                    # Conversión/calibración simples
                    humidity = h_raw / 10.0
                    temperature = self._calibrar_temperatura(t_raw)
                    ce = float(ce_raw)  # μS/cm
                    ph = ph_raw / 10.0
                    n_val = float(n_raw)
                    p_val = float(p_raw)
                    k_val = float(k_raw)

                    with self.lock:
                        self.last_reading = {
                            'humidity':    humidity,
                            'temperature': temperature,
                            'ce':          ce,
                            'ph':          ph,
                            'N':           n_val,
                            'P':           p_val,
                            'K':           k_val
                        }

                    logging.debug(f"SoilSensor7en1 => {self.last_reading}")
                    if self.detectar_fallo():
                        return None
                    return self.last_reading
                else:
                    logging.error("No se obtuvieron 7 registros válidos del sensor 7en1.")
                    return None
            except Exception:
                logging.exception("Error al leer SoilSensor7en1:")
                return None
        else:
            # Modo simulado
            with self.lock:
                self.last_reading = {
                    'humidity':    random.uniform(30, 70),
                    'temperature': random.uniform(10, 30),
                    'ce':          random.uniform(1.0, 2.5),
                    'ph':          random.uniform(5.5, 7.5),
                    'N':           random.uniform(10, 60),
                    'P':           random.uniform(5, 40),
                    'K':           random.uniform(10, 50)
                }
            logging.debug(f"SoilSensor7en1 (sim) => {self.last_reading}")
            return self.last_reading

    def _calibrar_temperatura(self, raw_val):
        """
        Manejo de signo para temperatura si el valor viene en complemento a 2.
        """
        if raw_val >= 0x8000:  # bit 15 en 1 => valor negativo
            return -(0x10000 - raw_val) / 10.0
        else:
            return raw_val / 10.0

    def detectar_fallo(self):
        """
        Chequea rangos básicos (humedad, temp, CE, pH, N, P, K).
        ADVERTENCIA: pH > 9 se considera fallo. Ajustar si tu rango real es mayor.
        """
        with self.lock:
            if self.last_reading is None:
                logging.error("No hay lectura disponible en SoilSensor7en1.")
                return True

            h = self.last_reading['humidity']
            t = self.last_reading['temperature']
            c = self.last_reading['ce']
            pH = self.last_reading['ph']
            n_ = self.last_reading['N']
            p_ = self.last_reading['P']
            k_ = self.last_reading['K']

            # Humedad
            if not (0 <= h <= 100):
                logging.error(f"Humedad fuera de rango: {h}")
                return True
            # Temperatura
            if not (-40 <= t <= 80):
                logging.error(f"Temperatura fuera de rango: {t}")
                return True
            # CE (μS/cm)
            if not (0 <= c <= 20000):
                logging.error(f"CE fuera de rango: {c}")
                return True
            # pH
            if not (3 <= pH <= 9):
                logging.error(f"pH fuera de rango: {pH}")
                return True
            # N, P, K
            if not (0 <= n_ <= 2000):
                logging.error(f"N fuera de rango: {n_}")
                return True
            if not (0 <= p_ <= 2000):
                logging.error(f"P fuera de rango: {p_}")
                return True
            if not (0 <= k_ <= 2000):
                logging.error(f"K fuera de rango: {k_}")
                return True

        return False


# -------------------------------------------------------------------------------
# LevelSensor: sensor de nivel (US-100) con Trigger/Echo
# -------------------------------------------------------------------------------
class LevelSensor:
    """
    Sensor de nivel (US-100) con Trigger/Echo para medir distancia en cm
    y convertir a % de nivel, dada una altura nominal 'tank_height'.

    Mejoras/Advertencias:
      - _medir_distancia() usa un while bloqueante con un timeout de ~1s.
      - En sistemas no deterministas, si hay interrupciones largas, puede dar
        falsos timeouts o lecturas inconsistentes.
    """

    def __init__(self, trig_pin, echo_pin, calibration_params=None):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.calibration_params = calibration_params or {}
        self.last_reading = None
        self.simulation_mode = (sys.platform == "win32")
        self.simulated_water_level = 79.0  # Valor de arranque simulado en %

        self.lock = threading.Lock()

        if not self.simulation_mode and GPIO is not None:
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            logging.info("LevelSensor (US-100) inicializado en modo real.")
        else:
            logging.info("LevelSensor en modo simulado.")

    def leer(self, bomba_activa=False, num_samples=5):
        """
        Toma 'num_samples' mediciones para reducir ruido y calcula el promedio.
        Convierte la distancia a porcentaje de nivel (con 'tank_height').

        :param bomba_activa: bool, si la bomba está encendida (afecta simulación).
        :param num_samples: número de muestras para promediar.
        :return: Porcentaje de nivel (0..100) o None si fallo.
        """
        if not self.simulation_mode and GPIO is not None:
            try:
                distances = []
                for _ in range(num_samples):
                    dist = self._medir_distancia()
                    if dist is not None:
                        distances.append(dist)
                    time.sleep(1)  # Pequeño delay entre muestras

                if distances:
                    avg_dist = sum(distances) / len(distances)
                    tank_height = self.calibration_params.get('tank_height', 80)
                    # Convertir distancia a % nivel
                    nivel = (tank_height - avg_dist) / tank_height * 100
                    nivel = max(0, min(100, nivel))  # Limitar a 0..100

                    with self.lock:
                        self.last_reading = nivel

                    logging.debug(f"LevelSensor => {nivel:.2f}%")
                    if self.detectar_fallo():
                        return None
                    return nivel
                else:
                    logging.error("No se pudo medir distancia del sensor ultrasónico en ninguna muestra.")
                    return None
            except Exception:
                logging.exception("Error leyendo LevelSensor:")
                return None

        else:
            # Modo simulado
            with self.lock:
                if bomba_activa:
                    # Si la bomba está activa, bajamos el nivel
                    dec = 0.5
                    self.simulated_water_level = max(self.simulated_water_level - dec, 0.0)
                else:
                    # Recuperación lenta (llenado)
                    inc = 0.1
                    self.simulated_water_level = min(self.simulated_water_level + inc, 100.0)

                self.last_reading = self.simulated_water_level

            logging.debug(f"LevelSensor(sim) => {self.last_reading:.2f}%")
            if self.detectar_fallo():
                return None
            return self.last_reading

    def _medir_distancia(self):
        """
        Envía pulso TRIG, mide duración del pulso ECHO y convierte a cm.
        Retorna la distancia o None en caso de timeout/error.
        """
        try:
            GPIO.output(self.trig_pin, False)
            time.sleep(0.0002)  # Pequeño delay

            GPIO.output(self.trig_pin, True)
            time.sleep(0.00001)
            GPIO.output(self.trig_pin, False)

            timeout = time.time() + 1
            while GPIO.input(self.echo_pin) == 0:
                inicio = time.time()
                if inicio > timeout:
                    logging.error("Timeout esperando inicio pulso ECHO (LOW->HIGH).")
                    return None

            while GPIO.input(self.echo_pin) == 1:
                fin = time.time()
                if fin > timeout:
                    logging.error("Timeout esperando fin pulso ECHO (HIGH->LOW).")
                    return None

            duracion = fin - inicio
            distancia_cm = (duracion * 34300) / 2
            return distancia_cm
        except Exception:
            logging.exception("Error midiendo distancia en LevelSensor:")
            return None

    def detectar_fallo(self, expected_flow=None):
        """
        Chequeos mínimos de falla:
          - last_reading > 105% => sospechoso
        """
        with self.lock:
            if self.last_reading is None:
                logging.warning("LevelSensor sin lectura previa.")
                return True
            if self.last_reading > 105:
                logging.error(f"Lectura de nivel sospechosa: {self.last_reading:.2f}%")
                return True
        return False


# -------------------------------------------------------------------------------
# FlowSensor (FS300A)
# -------------------------------------------------------------------------------
class FlowSensor:
    """
    Sensor de flujo (FS300A) que cuenta pulsos mediante interrupciones en un pin GPIO.
    - self.flow_frequency se incrementa en cada pulso (callback).
    - Se realiza un conteo durante 1 segundo en leer().
    - La conversión a L/min se hace con un factor 'conversion_factor'.

    Advertencia:
      - En caudales muy altos, podría dispararse la interrupción con demasiada
        frecuencia y saturar la CPU en sistemas de baja capacidad.
      - Se recomienda un hardware o software adicional para "dividir" pulsos
        si el caudal esperado es muy elevado.
    """

    def __init__(self, gpio_pin, calibration_params=None):
        self.gpio_pin = gpio_pin
        self.calibration_params = calibration_params or {}
        self.last_reading = None
        self.simulation_mode = (sys.platform == "win32")

        self.flow_frequency = 0
        # Ejemplo: 5.5 pulsos/L => factor = 5.5 => 1 pulso/seg => 1/5.5 L/s => 10.9 L/min
        self.conversion_factor = self.calibration_params.get('factor', 5.5)
        self.simulated_flow_rate = 0.0

        self.lock = threading.Lock()

        if not self.simulation_mode and GPIO is not None:
            GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(self.gpio_pin, GPIO.RISING, callback=self._count_pulse)
            logging.info("FlowSensor inicializado (FS300A) en modo real.")
        else:
            logging.info("FlowSensor en modo simulado.")

    def _count_pulse(self, channel):
        """
        Callback para la interrupción de RISING en el pin del sensor de flujo.
        Incrementa el contador de pulsos de forma thread-safe.
        """
        with self.lock:
            self.flow_frequency += 1

    def leer(self, bomba_activa=False, valvula_riego_abierta=False):
        """
        Retorna el caudal en L/min. En modo real:
          - Se limpia el contador, se espera 1 segundo, se lee el contador.
          - flow_rate = (frequency / conversion_factor).
        En modo simulado:
          - Ajusta flow_rate según estado de bomba y válvula.

        :param bomba_activa: bool, si la bomba está encendida.
        :param valvula_riego_abierta: bool, si la válvula de riego está abierta.
        :return: Caudal en L/min o None si falla/detectar_fallo = True
        """
        if not self.simulation_mode and GPIO is not None:
            try:
                with self.lock:
                    self.flow_frequency = 0
                # Conteo de pulsos durante 1 segundo
                time.sleep(1)
                with self.lock:
                    frequency = self.flow_frequency

                flow_rate = frequency / self.conversion_factor
                with self.lock:
                    self.last_reading = flow_rate

                logging.debug(f"FlowSensor => {flow_rate:.3f} L/min")
                if self.detectar_fallo():
                    return None
                return flow_rate
            except Exception:
                logging.exception("Error en FlowSensor:")
                return None
        else:
            # Modo simulado
            with self.lock:
                # Ajuste de caudal según si la bomba y la válvula están activas
                if bomba_activa and valvula_riego_abierta:
                    max_flow = 60.0  # un tope arbitrario en L/min para simulación
                    inc = 5.0
                    self.simulated_flow_rate = min(self.simulated_flow_rate + inc, max_flow)
                else:
                    dec = 5.0
                    self.simulated_flow_rate = max(self.simulated_flow_rate - dec, 0.0)

                flow_rate = self.simulated_flow_rate
                self.last_reading = flow_rate

            logging.debug(f"FlowSensor(sim) => {flow_rate:.3f} L/min")
            if self.detectar_fallo():
                return None
            return flow_rate

    def detectar_fallo(self, expected_flow=None):
        """
        Detecta anomalías (fuga u obstrucción) si se pasa un expected_flow,
        además de verificar caudales excesivamente altos (p.e. > 200 L/min => fallo).

        :param expected_flow: Caudal esperado (L/min). Si None, se omite esta parte.
        """
        with self.lock:
            if self.last_reading is None:
                logging.warning("FlowSensor sin lectura previa.")
                return True

            # Chequeo básico: si el caudal > 200 L/min, se considera muy alto => posible sensor dañado
            if self.last_reading > 200:
                logging.error(f"FlowSensor lectura sospechosamente alta: {self.last_reading:.2f} L/min")
                return True

            if expected_flow is not None:
                tol = self.calibration_params.get('tolerance', 0.2)
                min_thr = self.calibration_params.get('minimal_flow_threshold', 0.5)

                if expected_flow == 0:
                    # No se espera flujo => si lo hay, podría ser fuga
                    if self.last_reading > min_thr:
                        logging.error("Fuga detectada: flujo cuando no debería haber.")
                        return True
                else:
                    # Se verifica si está por debajo o por encima del flujo esperado (± tolerancia)
                    if self.last_reading < expected_flow * (1 - tol):
                        logging.error("Posible bloqueo: flujo < esperado.")
                        return True
                    elif self.last_reading > expected_flow * (1 + tol):
                        logging.error("Posible fuga: flujo > esperado.")
                        return True

        return False
