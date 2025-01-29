import time
import logging
import sys

# Importar clases de sensors.py
from sensors import SoilSensor7en1, LevelSensor, FlowSensor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def test_soil_sensor7en1():
    """
    Prueba rápida del SoilSensor7en1 en la dirección 0x01 y en /dev/ttyUSB0.

    """
    logging.info("Iniciando test de SoilSensor7en1...")

    sensor_modbus_address = 1
    sensor_serial_port = '/dev/ttyUSB0'

    # Crear instancia del sensor
    soil_sensor = SoilSensor7en1(address=sensor_modbus_address,
                                 serial_port=sensor_serial_port)
    # Leer
    logging.info(f"Leyendo sensor 7en1 (addr={sensor_modbus_address}, port={sensor_serial_port})...")
    reading = soil_sensor.leer()
    if reading is not None:
        logging.info(f"Lectura SoilSensor7en1 => {reading}")
    else:
        logging.error("No se pudo leer el sensor de suelo 7 en 1 (retornó None).")


def test_level_sensor():
    """
    Prueba rápida del sensor de nivel US-100 (LevelSensor).
    - Mide el % de nivel si estás en Raspberry con Trigger/Echo reales.
    - En Windows (simulado), retorna valores simulados.
    """
    logging.info("Iniciando test de LevelSensor...")

    # Ajusta los pines o calibration_params según tu circuito
    TRIG_PIN = 18
    ECHO_PIN = 24
    calibration = {'tank_height': 80}  # Ejemplo: tanque de 80 cm

    level_sensor = LevelSensor(trig_pin=TRIG_PIN,
                               echo_pin=ECHO_PIN,
                               calibration_params=calibration)

    # Leer varias veces para ver variación
    for i in range(3):
        nivel = level_sensor.leer(bomba_activa=False, num_samples=3)
        if nivel is not None:
            logging.info(f"Lectura LevelSensor => {nivel:.2f}% (sample {i+1})")
        else:
            logging.error("No se pudo leer el sensor de nivel (retornó None).")
        time.sleep(1)


def test_flow_sensor():
    """
    Prueba rápida del FlowSensor (FS300A).
    - En Raspberry, contará pulsos reales (si el caudal existe).
    - En Windows (simulado), mostrará un caudal que depende de bomba_activa y valvula_riego_abierta.
    """
    logging.info("Iniciando test de FlowSensor...")

    FLOW_SENSOR_PIN = 17
    calibration = {
        'factor': 5.5,             # Pulsos/L para FS300A (ejemplo)
        'tolerance': 0.2,
        'minimal_flow_threshold': 0.5
    }

    flow_sensor = FlowSensor(gpio_pin=FLOW_SENSOR_PIN,
                             calibration_params=calibration)

    # Simulamos lectura con la bomba y la válvula:
    for i in range(3):
        # Cambia estas banderas para ver cómo varían las lecturas
        bomba_activa = True if i == 1 else False
        valvula_abierta = True if i == 1 else False

        caudal = flow_sensor.leer(bomba_activa=bomba_activa, valvula_riego_abierta=valvula_abierta)
        if caudal is not None:
            logging.info(f"Lectura FlowSensor => {caudal:.3f} L/min (sample {i+1})")
        else:
            logging.error("No se pudo leer el sensor de flujo (retornó None).")
        time.sleep(1)


def main():
    """
    Ejecuta todos los tests de sensores.
    Ajusta o descomenta según lo que quieras probar.
    """
    logging.info("==== INICIANDO TEST DE TODOS LOS SENSORES ====")

    # Test SoilSensor7en1 (Modbus RS485)
    test_soil_sensor7en1()

    # Test LevelSensor (US-100)
    test_level_sensor()

    # Test FlowSensor (FS300A)
    test_flow_sensor()

    logging.info("==== FIN DE TEST DE SENSORES ====")


if __name__ == "__main__":
    main()
