import unittest
from sensors import HumiditySensor, PhSensor

class TestSensors(unittest.TestCase):

    def test_humidity_sensor(self):
        sensor = HumiditySensor(channel=0, calibration_params={'min_adc': 5000, 'max_adc': 25000})
        sensor_value = sensor.leer()
        self.assertIsNotNone(sensor_value)

    def test_ph_sensor(self):
        sensor = PhSensor(channel=1, calibration_params={'min_adc': 0, 'max_adc': 32767})
        sensor_value = sensor.leer()
        self.assertIsNotNone(sensor_value)

if __name__ == '__main__':
    unittest.main()
