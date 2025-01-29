# signal_conditioning.py

import logging
import numpy as np

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SignalConditioning:
    """
    Clase para el acondicionamiento de señales de entrada y salida.
    Proporciona métodos para filtrar ruido, eliminar interferencias y ajustar niveles
    antes de la conversión ADC/DAC o interacción con sensores y actuadores.

    Se usan dos tipos de filtros sencillos:
      - 'median': Toma la mediana de las últimas N muestras.
      - 'moving_average': Toma la media de las últimas N muestras.
    Por defecto, N=5, aunque se podría personalizar.
    """

    def __init__(self, fs=1.0, filter_type='median', debug=False):
        """
        Inicializa el módulo de acondicionamiento de señal con la frecuencia de muestreo (fs)
        y el tipo de filtro ('median' o 'moving_average').

        :param fs: Frecuencia de muestreo en Hz (por defecto 1 Hz para señales lentas).
        :param filter_type: Tipo de filtro a aplicar ('median' o 'moving_average').
        :param debug: Si True, hace logging.debug al filtrar. Si False, reduce los mensajes de debug.
        """
        self.fs = fs
        self.nyq = 0.5 * fs  # Frecuencia de Nyquist (no se usa en estos filtros, pero puede ser útil en el futuro).
        self.filter_type = filter_type
        self.debug = debug

        # Historiales para cada tipo de señal que deseas filtrar.
        # Se podría parametrizar N si se quiere más muestras en la ventana.
        self.buffer_size = 5
        self.historicos = {
            'humedad': [],
            'ph': [],
            'ce': [],
            'nivel': [],
            'temperatura': []
            # Ejemplo: si quisieras filtrar también N, P, K, podrías añadir:
            # 'N': [], 'P': [], 'K': []
        }

        logging.info("Módulo de Acondicionamiento de Señal inicializado.")

    def acondicionar_humedad(self, valor):
        """Acondiciona la señal de humedad (por ejemplo, filtra ruido)."""
        return self._filtrar_valor(valor, 'humedad')

    def acondicionar_ph(self, valor):
        """Acondiciona la señal de pH."""
        return self._filtrar_valor(valor, 'ph')

    def acondicionar_ce(self, valor):
        """Acondiciona la señal de conductividad eléctrica (CE)."""
        return self._filtrar_valor(valor, 'ce')

    def acondicionar_nivel(self, valor):
        """Acondiciona la señal de nivel de agua del tanque (p. ej. ultrasonido)."""
        return self._filtrar_valor(valor, 'nivel')

    def acondicionar_temperatura(self, valor):
        """Acondiciona la señal de temperatura."""
        return self._filtrar_valor(valor, 'temperatura')

    def _filtrar_valor(self, valor, tipo_sensor):
        """
        Aplica el filtro indicado ('median' o 'moving_average') a la señal especificada.

        :param valor: Valor actual del sensor.
        :param tipo_sensor: Clave que indica de qué sensor se trata.
        :return: Valor filtrado (sea mediana o media de las últimas N muestras).
        """
        historial = self.historicos.get(tipo_sensor, [])
        historial.append(valor)

        # Mantener solo los últimos buffer_size valores
        if len(historial) > self.buffer_size:
            historial.pop(0)

        self.historicos[tipo_sensor] = historial

        if self.filter_type == 'median':
            valor_filtrado = float(np.median(historial))
        elif self.filter_type == 'moving_average':
            valor_filtrado = float(np.mean(historial))
        else:
            # Si se especifica otro filtro no soportado, por defecto usamos mediana
            valor_filtrado = float(np.median(historial))

        if self.debug:
            logging.debug(
                f"[SignalConditioning] Filtro={self.filter_type}, "
                f"Sensor={tipo_sensor}, ValorEntrada={valor}, ValorFiltrado={valor_filtrado}, "
                f"Historial={historial}"
            )
        return valor_filtrado
