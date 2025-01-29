# actuators.py

import time
import logging
import sys
import threading  # Importamos threading para el lock

# Simular RPi.GPIO solo si no estamos en una Raspberry Pi
if sys.platform == "win32":
    from unittest.mock import MagicMock
    GPIO = MagicMock()
    logging.warning("Librerías de hardware no disponibles. Ejecutando en modo simulado.")
else:
    try:
        import RPi.GPIO as GPIO
    except ImportError:
        logging.warning("Librerías de RPi.GPIO no disponibles en este entorno.")
        GPIO = None

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Sección de control GPIO global para evitar repeticiones

# _gpio_mode_set se utiliza como bandera para asegurarnos de que GPIO.setmode(GPIO.BCM)
# sólo se invoque una vez en toda la ejecución (por buenas prácticas de la librería RPi.GPIO).
_gpio_mode_set = False

# _gpio_lock se usa para proteger las llamadas a GPIO en entornos multi-hilo,
# evitando que dos hilos realicen setup o output simultáneamente.
_gpio_lock = threading.Lock()

# Limpieza global

def limpiar_gpio_global():
    """
    Realiza la limpieza global de GPIO al finalizar la aplicación.
    Recomendado para dejar la Raspberry Pi en un estado limpio.
    """
    with _gpio_lock:
        GPIO.cleanup()
    logging.info("Limpieza global de GPIO ejecutada correctamente.")

class ActuatorBase:
    """
    Clase base para los actuadores. Proporciona métodos para:
      - Inicialización del pin GPIO.
      - Activación y desactivación (control de salida digital).
      - Registro del estado interno (True/False).
    
    Maneja la lógica 'active_low' (algunos relés o transistores se activan poniendo
    la salida en LOW).
    """

    def __init__(self, gpio_pin, nombre_actuador="Actuador", active_low=False):
        """
        Inicializa el actuador en el pin GPIO especificado.

        :param gpio_pin: Número del pin GPIO al que está conectado el actuador (BCM).
        :param nombre_actuador: Nombre descriptivo del actuador (para logs).
        :param active_low: Indica si el actuador se activa con una señal baja (GPIO.LOW).
        """
        self.gpio_pin = gpio_pin
        self.nombre_actuador = nombre_actuador
        self.estado = False            # Estado inicial: desactivado
        self.active_low = active_low   # Control de lógica inversa

        # En modo simulado (Windows o sin RPi.GPIO), no se hace setup real.
        self.simulation_mode = (sys.platform == "win32" or GPIO is None)

        if not self.simulation_mode:
            with _gpio_lock:
                global _gpio_mode_set
                if not _gpio_mode_set:
                    # Se configura BCM solo la primera vez
                    GPIO.setmode(GPIO.BCM)
                    _gpio_mode_set = True

                # Intentamos configurar el pin en modo salida
                try:
                    GPIO.setup(self.gpio_pin, GPIO.OUT)
                except (RuntimeError, ValueError) as e:
                    logging.error(
                        f"No se pudo configurar el pin GPIO {self.gpio_pin} "
                        f"para {self.nombre_actuador}. Error: {str(e)}"
                    )
                    raise

                # Establecer el estado inicial (desactivado)
                if self.active_low:
                    GPIO.output(self.gpio_pin, GPIO.HIGH)
                else:
                    GPIO.output(self.gpio_pin, GPIO.LOW)

            logging.info(f"{self.nombre_actuador} inicializado en pin GPIO {self.gpio_pin}.")
        else:
            logging.info(f"{self.nombre_actuador} inicializado en modo simulado.")

    def activar(self):
        """
        Activa el actuador si no está ya activado.
        Cambia el estado interno a True y pone el pin en LOW/HIGH según la lógica.
        """
        if not self.estado:
            self.estado = True
            if not self.simulation_mode:
                with _gpio_lock:
                    if self.active_low:
                        GPIO.output(self.gpio_pin, GPIO.LOW)
                    else:
                        GPIO.output(self.gpio_pin, GPIO.HIGH)
            logging.info(f"{self.nombre_actuador} activado.")
        else:
            logging.debug(f"{self.nombre_actuador} ya estaba activado.")

    def desactivar(self):
        """
        Desactiva el actuador si está activado.
        Cambia el estado interno a False y pone el pin en LOW/HIGH según la lógica.
        """
        if self.estado:
            self.estado = False
            if not self.simulation_mode:
                with _gpio_lock:
                    if self.active_low:
                        GPIO.output(self.gpio_pin, GPIO.HIGH)
                    else:
                        GPIO.output(self.gpio_pin, GPIO.LOW)
            logging.info(f"{self.nombre_actuador} desactivado.")
        else:
            logging.debug(f"{self.nombre_actuador} ya estaba desactivado.")

    def estado_actual(self):
        """
        Devuelve el estado actual del actuador.

        :return: True si está activado, False si está desactivado.
        """
        return self.estado

    def limpiar(self):
        """
        Marca el pin como listo para limpieza global. No se ejecuta GPIO.cleanup(pin).
        """
        if not self.simulation_mode:
            logging.info(f"Pin GPIO {self.gpio_pin} listo para limpieza global.")
        else:
            logging.info(f"Simulación de limpieza del GPIO para {self.nombre_actuador}.")

class PumpControl(ActuatorBase):
    """
    Control de la bomba DC.
    Se asume que la bomba se activa con una señal alta (active_low=False).
    """
    def __init__(self, gpio_pin):
        super().__init__(
            gpio_pin=gpio_pin,
            nombre_actuador="Bomba DC",
            active_low=False
        )

class ValveControl:
    """
    Control de las electroválvulas. Maneja múltiples válvulas (ej. 'desague_1', 'mora', etc.)
    Cada válvula se representa con una instancia de ActuatorBase.

    Las válvulas se asumen 'active_low' (muchos relés se activan poniendo la salida en LOW).
    """
    def __init__(self, valvulas_config):
        """
        Inicializa las electroválvulas según la configuración (dict).

        :param valvulas_config: Diccionario: nombre_valvula -> pin_GPIO
               Ej: { 'desague_1': 5, 'mora': 13, ... }
        """
        self.valvulas = {}
        # Lock para asegurar que no haya colisiones en abrir/cerrar válvulas simultáneamente.
        self.lock = threading.Lock()

        for nombre, pin in valvulas_config.items():
            self.valvulas[nombre] = ActuatorBase(
                pin,
                nombre_actuador=f"Válvula {nombre}",
                active_low=True
            )
        logging.info("Control de Electroválvulas inicializado.")

    def abrir_valvula(self, nombre_valvula):
        """
        Abre la válvula especificada si no está ya abierta.
        Usa un lock para evitar conflictos en accesos simultáneos.

        :param nombre_valvula: Nombre de la válvula a abrir (debe existir en self.valvulas).
        """
        with self.lock:
            valvula = self.valvulas.get(nombre_valvula)
            if valvula is not None:
                valvula.activar()
            else:
                logging.error(f"No se encontró la válvula '{nombre_valvula}' para abrir.")

    def cerrar_valvula(self, nombre_valvula):
        """
        Cierra la válvula especificada si está abierta.
        """
        with self.lock:
            valvula = self.valvulas.get(nombre_valvula)
            if valvula is not None:
                valvula.desactivar()
            else:
                logging.error(f"No se encontró la válvula '{nombre_valvula}' para cerrar.")

    def estado_actual(self, nombre_valvula):
        """
        Devuelve el estado actual de la válvula especificada:
          True = abierta, False = cerrada.

        :raises ValueError: si la válvula no existe.
        """
        valvula = self.valvulas.get(nombre_valvula)
        if valvula is None:
            raise ValueError(f"La válvula '{nombre_valvula}' no se encuentra registrada.")

        return valvula.estado_actual()

    def limpiar(self):
        """
        Llama a limpiar() en cada válvula controlada. De forma análoga a ActuatorBase,
        no se ejecuta GPIO.cleanup(pin) individualmente.
        """
        for valvula in self.valvulas.values():
            valvula.limpiar()
        logging.info("Configuración de electroválvulas lista para limpieza global.")
