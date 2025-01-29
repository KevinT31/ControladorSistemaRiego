# generate_synthetic_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def generate_synthetic_data(num_samples=4320):
    """
    Genera datos sintéticos realistas para entrenar el modelo de Machine Learning.
    - Por defecto: 90 días, con 2 muestras/hora => 48 muestras/día => 4320 totales.
    - Cada muestra corresponde aproximadamente a 30 minutos de datos.

    Param:
    -------
    num_samples : int
        Número total de muestras a generar. Por defecto 4320 (90 días * 48).
    """
    data = []
    # Fecha inicial ~ hace 90 días
    start_date = datetime.now() - timedelta(days=90)
    current_time = start_date

    # Variables iniciales (semi-random) para simular evolución
    last_humidity = np.random.uniform(30, 70)       # % de humedad del suelo
    last_temperature = np.random.uniform(15, 25)    # °C
    last_ph = np.random.uniform(5.5, 7.5)
    last_ce = np.random.uniform(1.0, 2.5)           # Conductividad eléctrica
    last_water_level = np.random.uniform(40, 80)    # 0-100, representando %
    last_flow_rate = 0.0                            # L/min
    # Nutrientes N, P, K en ppm (arbitrario)
    last_n = np.random.uniform(10, 50)
    last_p = np.random.uniform(5, 30)
    last_k = np.random.uniform(10, 40)

    for _ in range(num_samples):
        # Incrementar timestamp en 30 min
        current_time += timedelta(minutes=30)

        # Determinar estación según mes
        month = current_time.month
        if month in [12, 1, 2]:
            season = 'summer'
            base_temp = 27
            temp_var = 5
        elif month in [3, 4, 5]:
            season = 'autumn'
            base_temp = 20
            temp_var = 4
        elif month in [6, 7, 8]:
            season = 'winter'
            base_temp = 12
            temp_var = 3
        else:
            season = 'spring'
            base_temp = 22
            temp_var = 4

        # Simular temperatura actual
        temperature = base_temp + np.random.normal(0, temp_var)
        temperature = np.clip(temperature, 5, 40)

        # La humedad del suelo depende del riego y la evapotranspiración
        # Si hubo riego recientemente (flow_rate > 0), sube la humedad
        # Sino, va bajando lentamente
        if last_flow_rate > 0:
            humidity = last_humidity + np.random.uniform(4, 10)
        else:
            humidity = last_humidity - np.random.uniform(0, 2)
        humidity = np.clip(humidity, 5, 95)

        # pH con ligeras variaciones
        ph = last_ph + np.random.normal(0, 0.05)
        ph = np.clip(ph, 4.5, 8.5)

        # CE varía con fertilizaciones y drenajes
        ce = last_ce + np.random.normal(0, 0.05)
        ce = np.clip(ce, 0.5, 3.5)

        # Nutrientes N, P, K (aumentan si se inyecta fertilizante, sino bajan)
        n = last_n
        p = last_p
        k = last_k

        # Lógica de acciones (bomba, riego, fertilizante, etc.)
        activar_bomba = False
        abrir_valvula_riego = False
        inyectar_fertilizante = False
        abrir_valvula_suministro = False
        abrir_juego_desague = False

        # Por defecto, no hay fertilizante ni agua
        porcentaje_fertilizante = 0.0
        cantidad_agua = 0.0
        flow_rate = 0.0

        # Ajustar water_level
        # Si la bomba está prendida y se riega, baja el nivel
        # Si se abre el suministro, sube el nivel
        water_level = last_water_level

        # REGLAS de “experto manual” (simplificadas):
        # 1) Si humidity < 35 o temp > 30 => regar
        # 2) Si CE < 0.9 => inyectar fertilizante
        # 3) Si water_level < 20 => abrir suminstro
        # 4) Si water_level > 90 => abrir desagüe
        # 5) flow_rate simulado en [20..60] L/min si se riega

        # Regla 1: riego
        if humidity < 35 or temperature > 30:
            activar_bomba = True
            abrir_valvula_riego = True
            cantidad_agua = np.random.uniform(10, 25)
            flow_rate = np.random.uniform(20, 60)

        # Regla 2: fertilizar si CE baja
        if ce < 0.9:
            inyectar_fertilizante = True
            porcentaje_fertilizante = np.random.uniform(1, 5)

        # Ajustar N, P, K si se fertiliza
        if inyectar_fertilizante:
            # Incrementar N, P, K en un rango
            dn = np.random.uniform(5, 12)
            dp = np.random.uniform(2, 8)
            dk = np.random.uniform(3, 10)
            n = last_n + dn
            p = last_p + dp
            k = last_k + dk
        else:
            # Disminuyen un poco con el tiempo
            n = last_n - np.random.uniform(0, 1)
            p = last_p - np.random.uniform(0, 0.5)
            k = last_k - np.random.uniform(0, 0.5)
        n = np.clip(n, 0, 150)
        p = np.clip(p, 0, 80)
        k = np.clip(k, 0, 120)

        # Regla 3: Suministro
        if water_level < 20:
            abrir_valvula_suministro = True
            # Simular relleno
            water_level += np.random.uniform(5, 15)

        # Regla 4: Desagüe
        if water_level > 90:
            abrir_juego_desague = True
            # Simular drenaje
            water_level -= np.random.uniform(5, 15)

        # Ajustar water_level si bomba y riego => se reduce
        if activar_bomba and abrir_valvula_riego:
            # Q se retira del tanque
            water_level -= np.random.uniform(1, 3)

        water_level = np.clip(water_level, 0, 100)

        # Recalcular CE (sube un poco si fertilizas, baja si drenas)
        if inyectar_fertilizante:
            ce += np.random.uniform(0.1, 0.4)
        if abrir_juego_desague:
            ce -= np.random.uniform(0.1, 0.3)
        ce = np.clip(ce, 0.5, 3.5)

        # Actualizar states para la siguiente iteración
        last_humidity = humidity
        last_temperature = temperature
        last_ph = ph
        last_ce = ce
        last_water_level = water_level
        last_flow_rate = flow_rate
        last_n, last_p, last_k = n, p, k

        # Construir registro
        sample = {
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'humidity': round(humidity, 1),
            'temperature': round(temperature, 1),
            'ph': round(ph, 2),
            'ce': round(ce, 2),
            'N': round(n, 2),
            'P': round(p, 2),
            'K': round(k, 2),
            'water_level': round(water_level, 1),
            'flow_rate': round(flow_rate, 1),
            'season': season,

            'activar_bomba': activar_bomba,
            'abrir_valvula_riego': abrir_valvula_riego,
            'inyectar_fertilizante': inyectar_fertilizante,
            'abrir_valvula_suministro': abrir_valvula_suministro,
            'abrir_juego_desague': abrir_juego_desague,

            'porcentaje_fertilizante': round(porcentaje_fertilizante, 1),
            'cantidad_agua': round(cantidad_agua, 1)
        }
        data.append(sample)

    df = pd.DataFrame(data)

    # Crear carpeta 'data' si no existe
    os.makedirs('data', exist_ok=True)

    output_path = 'data/decision_data.csv'
    df.to_csv(output_path, index=False)
    logging.info(f"Datos sintéticos generados y guardados en '{output_path}'. Filas: {len(df)}")

if __name__ == "__main__":
    generate_synthetic_data(num_samples=4320)
