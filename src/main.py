# Main
import logging
import sys
from controller import ControladorSistemaRiego

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    """
    Punto de entrada principal del sistema de riego inteligente.
    Inicializa el controlador principal y gestiona el bucle principal de operaciones.
    """
    try:
        logging.info("Iniciando el sistema de riego automático...")

        # 1) Inicializar el controlador
        controlador = ControladorSistemaRiego()

        # 2) Iniciar el bucle principal del controlador
        controlador.iniciar()

    except KeyboardInterrupt:
        # Manejo de la interrupción del teclado (Ctrl + C)
        logging.info("Sistema de riego detenido manualmente por el usuario.")

    except Exception as e:
        # Captura de cualquier excepción no prevista
        logging.error(f"Error inesperado en el sistema: {e}", exc_info=True)

    finally:
        logging.info("Sistema de riego finalizado.")

if __name__ == "__main__":
    main()
