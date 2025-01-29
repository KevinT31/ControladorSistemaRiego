# gui.py
# -------------------------------------------------------------------------------
# Interfaz gráfica (Node-RED) para el sistema de riego inteligente.
# Permite:
#   - Enviar datos de sensores y estado de control a Node-RED.
#   - Recibir comandos manuales.
#   - Monitorear si Node-RED está disponible (health endpoint).
#   - Manejar modo de control (automatic/manual).
#   - Enviar sugerencias al operador (modo manual sugerido).
# -------------------------------------------------------------------------------

import requests
import logging
import threading
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class NodeRedInterface:
    """
    Clase para manejar la comunicación con la interfaz gráfica en Node-RED.
    Permite:
      - Enviar datos de sensores y el modo de control a un endpoint Node-RED (/update).
      - Recibir el modo de control actual (/control_mode).
      - Recibir comandos manuales (/manual_commands).
      - Enviar sugerencias al operador (/suggestions).
      - Monitorear periódicamente si Node-RED está disponible (/health).

    Uso típico:
      1) Se instancia la clase con la URL base de Node-RED.
      2) Se llama a actualizar_interfaz(sensor_values, control_mode, decision)
         para mandar info al dashboard.
      3) Se llama a obtener_modo_control() para saber si es 'automatic' o 'manual'.
      4) En modo 'manual', se usa recibir_comandos() para obtener órdenes (activar bomba, etc.).
      5) Se puede mostrar_sugerencias(...) para enviar recomendaciones (en modo manual).
    """

    def __init__(self, node_red_url="http://localhost:1880"):
        """
        :param node_red_url: Dirección base del servidor Node-RED.
                             Ej: 'http://192.168.0.10:1880'
        """
        self.node_red_url = node_red_url
        self.control_mode = 'automatic'  # 'automatic' o 'manual'
        self.lock = threading.Lock()
        self.node_red_available = False
        self.last_message_sent = None  # Para evitar spam de sugerencias repetidas

        # Flag para detener el hilo si se cierra la app
        self._stop_monitor = False

        # Iniciar un hilo para monitorear la disponibilidad de Node-RED
        self._monitor_thread = threading.Thread(target=self.monitor_interface, daemon=True)
        self._monitor_thread.start()

    def __del__(self):
        """
        Destructor. Tratamos de detener el hilo de monitor de forma ordenada.
        (En Python, no se garantiza que __del__ se ejecute siempre, por ejemplo
         si el intérprete se cierra abruptamente. Se sugiere llamar a shutdown()).
        """
        self.shutdown()

    def shutdown(self):
        """
        Método para detener el hilo de monitorización de Node-RED si se desea
        un cierre ordenado. Ajusta la variable _stop_monitor a True.
        """
        self._stop_monitor = True

    def actualizar_interfaz(self, sensor_values, control_mode, decision=None):
        """
        Envía los datos de los sensores, el modo de control y (opcional) la decisión
        a Node-RED para su visualización (endpoint /update).

        :param sensor_values: dict con los valores de los sensores.
        :param control_mode: 'automatic' o 'manual'
        :param decision: dict con la última decisión tomada (opcional).
        """
        try:
            data = {
                'sensor_values': sensor_values,
                'control_mode': control_mode,
                'decision': decision
            }
            requests.post(f"{self.node_red_url}/update", json=data, timeout=5)
            logging.info("Interfaz Node-RED actualizada con los últimos datos.")
        except requests.RequestException:
            logging.warning("No se pudo actualizar la interfaz Node-RED (posible desconexión).")

    def obtener_modo_control(self, retries=3):
        """
        Consulta a Node-RED (endpoint /control_mode) para saber el modo de control actual.
        Retorna 'automatic' o 'manual'. Intenta 'retries' veces en caso de error.

        :param retries: Número de reintentos si hay fallo de conexión.
        :return: Cadena 'automatic' o 'manual'
        """
        for attempt in range(retries):
            try:
                response = requests.get(f"{self.node_red_url}/control_mode", timeout=5)
                if response.status_code == 200:
                    datos = response.json()
                    self.control_mode = datos.get('control_mode', 'automatic')
                    logging.info(f"Modo de control recibido de Node-RED: {self.control_mode}")
                    return self.control_mode
                else:
                    logging.warning(f"Respuesta inesperada al obtener modo de control: {response.status_code}")
            except requests.RequestException:
                logging.warning(f"Fallo al obtener modo de control (intento {attempt+1}).")
                time.sleep(1)

        logging.error("No se pudo conectar con Node-RED tras varios intentos. Se mantiene modo_control actual.")
        return self.control_mode

    def recibir_comandos(self, retries=3):
        """
        Obtiene comandos manuales desde Node-RED (endpoint /manual_commands).
        En modo manual, el usuario puede solicitar activar bomba, abrir válvulas, etc.

        :param retries: Número de reintentos
        :return: Diccionario con los comandos, ej. {"activar_bomba": True, "abrir_valvula_mora": True, ...}
                 En caso de falla, retorna {}.
        """
        for attempt in range(retries):
            try:
                response = requests.get(f"{self.node_red_url}/manual_commands", timeout=5)
                if response.status_code == 200:
                    commands = response.json()
                    logging.info(f"Comandos recibidos desde Node-RED: {commands}")
                    return commands
                else:
                    logging.warning(f"Respuesta inesperada al obtener comandos manuales: {response.status_code}")
            except requests.RequestException:
                logging.warning(f"Fallo al recibir comandos manuales (intento {attempt+1}).")
                time.sleep(1)

        logging.error("No se pudo obtener comandos manuales de Node-RED tras varios intentos.")
        return {}

    def mostrar_sugerencias(self, sugerencias, retries=3):
        """
        Envía sugerencias (en modo manual) a Node-RED (endpoint /suggestions).
        Si Node-RED no está disponible, las registra localmente en el log.
        Para evitar spam, si el mensaje es idéntico al anterior, no se reenvía.

        NOTA: Si la misma sugerencia se repite pero en realidad es importante
        (por ejemplo, el estado interno cambió), se omite. Ajustar la lógica
        si se requiere reenviar mensajes repetidos.

        :param sugerencias: dict con sugerencias, ej:
                {
                  'accion_recomendada': 'Abrir riego',
                  'agua_sugerida(L)': 10,
                  'fertilizar': 'Sí',
                  'dosis_fertilizante(%)': 2.5
                }
        :param retries: Número de reintentos en caso de fallo de red.
        """
        # Si Node-RED no está disponible, sólo guardamos en log
        if not self.node_red_available:
            logging.info(f"Sugerencias (offline): {sugerencias}")
            return

        # Evitar enviar el mismo mensaje repetidamente
        if sugerencias == self.last_message_sent:
            logging.info("Mensaje de sugerencia repetido, se omite el reenvío.")
            return
        else:
            self.last_message_sent = sugerencias

        for attempt in range(retries):
            try:
                requests.post(f"{self.node_red_url}/suggestions", json=sugerencias, timeout=5)
                logging.info("Sugerencias enviadas a Node-RED correctamente.")
                return
            except requests.RequestException:
                logging.warning(f"Fallo al enviar sugerencias a Node-RED (intento {attempt+1}).")
                time.sleep(1)

        logging.error("No se pudo enviar sugerencias a Node-RED tras varios intentos.")

    def monitor_interface(self):
        """
        Hilo que verifica periódicamente si Node-RED está activo (endpoint /health).
        Si el status_code == 200 => node_red_available=True; si no => False.

        Se detiene si self._stop_monitor se pone a True (vía shutdown()).
        """
        while not self._stop_monitor:
            try:
                response = requests.get(f"{self.node_red_url}/health", timeout=5)
                if response.status_code == 200:
                    self.node_red_available = True
                    logging.info("Node-RED está disponible (health=200).")
                else:
                    self.node_red_available = False
                    logging.warning("Node-RED responde, pero con estado != 200.")
            except requests.RequestException:
                self.node_red_available = False
                logging.warning("No se pudo conectar con Node-RED (health check fallido).")

            # Revisa cada 60 seg
            for _ in range(60):
                if self._stop_monitor:
                    break
                time.sleep(1)
