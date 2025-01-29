import logging
from controller import ControladorSistemaRiego

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    """
    Test de ControladorSistemaRiego en modo simulación.
    Prueba sensores, actuadores y lógica de decisión.
    """
    # Inicializar el controlador
    controlador = ControladorSistemaRiego()

    # Simular lectura de sensores
    logging.info("=== Leyendo sensores en modo simulación ===")
    sensores = controlador._leer_sensores_global()
    logging.info(f"Sensores leídos: {sensores}")

    # Simular decisión del motor de decisiones
    logging.info("=== Probando el motor de decisiones ===")
    decision = controlador.decision_engine.evaluar(sensores)
    logging.info(f"Decisión generada: {decision}")

    # Simular ejecución de la decisión
    logging.info("=== Ejecutando la decisión ===")
    controlador._ejecutar_decision(decision)

    # Verificar estado de actuadores
    logging.info("=== Estado actual de los actuadores ===")
    estados = controlador.get_actuators_state()
    logging.info(f"Estados: {estados}")

    # Simular control manual desde Node-RED (en modo manual)
    logging.info("=== Simulando control manual desde Node-RED ===")
    controlador.control_mode = 'manual'
    comandos_manual = {'activar_bomba': True, 'abrir_valvula_mora': True}
    logging.info(f"Comandos recibidos: {comandos_manual}")
    if comandos_manual.get('activar_bomba'):
        controlador.pump_control.activar()
    if comandos_manual.get('abrir_valvula_mora'):
        controlador.valve_control.abrir_valvula('mora')

    # Verificar nuevamente estado de actuadores
    logging.info("=== Estado de los actuadores después del control manual ===")
    estados = controlador.get_actuators_state()
    logging.info(f"Estados: {estados}")

    # Finalizar
    logging.info("=== Test completado ===")

if __name__ == "__main__":
    main()